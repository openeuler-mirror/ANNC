#include "fusion_meta.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace annc::converter_v2 {
namespace {

std::string missingFieldError(const std::string &field) {
  return "fusion metadata missing required field '" + field + "'";
}

// ── JSON reading (mirrors FusionMetadataJson.cpp) ──────────────────────────

template <typename T>
std::vector<T> jsonValue(const nlohmann::json &json, const char *key) {
  return json.value(key, std::vector<T>{});
}

Result<FusionInfo> fusionInfoFromJson(const nlohmann::json &json) {
  Result<FusionInfo> result;
  if (!json.is_object()) {
    result.error = "fusion metadata entry is not an object";
    return result;
  }

  FusionInfo info;
  try {
    info.name = json.value("name", "");
    info.pattern = json.value("pattern", "");
    info.executionMode = json.value("execution_mode", "aot");
    info.kernelName = json.value("kernel_name", "");
    info.templateFingerprint = json.value("template_fingerprint", "");
    info.patternAttrs =
        json.value("pattern_attrs", std::map<std::string, std::string>{});

    if (json.contains("args") && json["args"].is_array()) {
      for (const auto &item : json["args"]) {
        FusionArg arg;
        arg.role = item.value("role", "");
        arg.tfName = item.value("tf_name", "");
        arg.shape = jsonValue<int64_t>(item, "shape");
        arg.rank = item.value("rank", int64_t{-1});
        arg.dtype = item.value("dtype", "");
        info.args.push_back(std::move(arg));
      }
    }

    if (json.contains("outputs") && json["outputs"].is_array()) {
      for (const auto &item : json["outputs"]) {
        FusionArg output;
        output.role = item.value("role", "");
        output.tfName = item.value("tf_name", "");
        output.shape = jsonValue<int64_t>(item, "shape");
        output.rank = item.value("rank", int64_t{-1});
        output.dtype = item.value("dtype", "");
        info.outputs.push_back(std::move(output));
      }
    }

    info.abi = json.value("abi", "mlir_ciface");
    if (info.abi.empty()) info.abi = "mlir_ciface";
    info.memberNodeNames =
        json.value("member_node_names", std::vector<std::string>{});
    info.dynamicDims = jsonValue<int64_t>(json, "dynamic_dims");
    info.kernelArgOrder = jsonValue<int64_t>(json, "kernel_arg_order");
    info.contextI64Attrs = jsonValue<int64_t>(json, "context_i64_attrs");
    info.symbolicSignature = json.value("symbolic_signature", "");
    info.fallbackFunction = json.value("fallback_function", "");
  } catch (const std::exception &e) {
    result.error = std::string("failed to parse fusion metadata JSON: ") +
                   e.what();
    return result;
  }

  Status validation = validateFusionInfo(info);
  if (!validation.ok()) {
    result.error = validation.error;
    return result;
  }
  result.value = std::move(info);
  return result;
}

}  // namespace

// ── TensorEndpoint (mirrors TensorEndpoint.cpp) ────────────────────────────

std::string TensorEndpoint::canonicalDataName() const {
  return node + ":" + std::to_string(port);
}

Result<TensorEndpoint> parseTensorEndpoint(std::string_view input) {
  Result<TensorEndpoint> result;
  if (input.empty()) {
    result.error = "tensor endpoint is empty";
    return result;
  }

  TensorEndpoint endpoint;
  std::string_view value = input;
  if (value.front() == '^') {
    endpoint.control = true;
    value.remove_prefix(1);
    if (value.empty() || value.find(':') != std::string_view::npos) {
      result.error = "control endpoint must not contain a data port: " +
                     std::string(input);
      return result;
    }
    endpoint.node = std::string(value);
    result.value = std::move(endpoint);
    return result;
  }

  size_t colon = value.find(':');
  if (colon == std::string_view::npos) {
    endpoint.node = std::string(value);
    result.value = std::move(endpoint);
    return result;
  }
  if (colon == 0 || value.find(':', colon + 1) != std::string_view::npos ||
      colon + 1 == value.size()) {
    result.error = "malformed tensor endpoint: " + std::string(input);
    return result;
  }
  endpoint.node = std::string(value.substr(0, colon));
  std::string_view portText = value.substr(colon + 1);
  for (char c : portText) {
    if (c < '0' || c > '9') {
      result.error =
          "malformed tensor endpoint port: " + std::string(input);
      return result;
    }
  }
  errno = 0;
  char *end = nullptr;
  std::string portString(portText);
  long long parsed = std::strtoll(portString.c_str(), &end, 10);
  if (errno == ERANGE || end != portString.c_str() + portString.size() ||
      parsed < 0) {
    result.error = "invalid tensor endpoint port: " + std::string(input);
    return result;
  }
  endpoint.port = parsed;
  result.value = std::move(endpoint);
  return result;
}

// ── Validation (mirrors validateFusionInfo) ────────────────────────────────

Status validateFusionInfo(const FusionInfo &info) {
  Status status;
  if (info.name.empty()) {
    status.error = missingFieldError("name");
    return status;
  }
  if (info.pattern.empty()) {
    status.error = missingFieldError("pattern");
    return status;
  }
  if (info.executionMode != "aot" && info.executionMode != "jit") {
    status.error =
        "unsupported fusion execution mode: " + info.executionMode;
    return status;
  }
  if (info.kernelName.empty()) {
    status.error = missingFieldError("kernel_name");
    return status;
  }
  if (!info.templateFingerprint.empty()) {
    bool validFingerprint =
        info.templateFingerprint.size() == 64 &&
        std::all_of(info.templateFingerprint.begin(),
                    info.templateFingerprint.end(), [](unsigned char c) {
                      return std::isdigit(c) || (c >= 'a' && c <= 'f');
                    });
    if (!validFingerprint) {
      status.error =
          "fusion metadata template_fingerprint must be 64 lowercase hex "
          "characters";
      return status;
    }
  }
  if (info.executionMode == "jit" && info.templateFingerprint.empty()) {
    status.error = missingFieldError("template_fingerprint for JIT fusion");
    return status;
  }
  if (info.args.empty()) {
    status.error = missingFieldError("args");
    return status;
  }
  if (info.outputs.empty()) {
    status.error = missingFieldError("outputs");
    return status;
  }
  if (info.abi != "mlir_ciface" && info.abi != "annc_execution_v2" &&
      info.abi != "annc_context_v3") {
    status.error = "unsupported fusion ABI: " + info.abi;
    return status;
  }
  if (info.abi == "annc_execution_v2" && !info.kernelArgOrder.empty()) {
    status.error = "annc_execution_v2 must not persist output argument order";
    return status;
  }
  if (info.abi == "annc_context_v3" && !info.kernelArgOrder.empty()) {
    // Context v3 transports inputs by role groups and records the inverse
    // mapping in context_input_order; a persisted output-argument order would
    // be a second, conflicting contract.
    status.error = "annc_context_v3 does not accept kernel_arg_order";
    return status;
  }

  static const std::unordered_set<std::string> inputRoles = {
      "constant", "fixed", "dynamic"};
  for (const FusionArg &arg : info.args) {
    if (!inputRoles.count(arg.role)) {
      status.error = "fusion input '" + arg.tfName + "' has invalid role '" +
                     arg.role + "'";
      return status;
    }
    if (arg.tfName.empty()) {
      status.error = missingFieldError("args.tf_name");
      return status;
    }
    if (arg.rank < 0 || arg.rank != static_cast<int64_t>(arg.shape.size())) {
      status.error = "fusion input '" + arg.tfName +
                     "' rank does not match its shape";
      return status;
    }
    if (arg.dtype.empty()) {
      status.error = missingFieldError("args.dtype");
      return status;
    }
    auto endpoint = parseTensorEndpoint(arg.tfName);
    if (!endpoint.ok()) {
      status.error = endpoint.error;
      return status;
    }
    if (endpoint.value.control) {
      status.error = "fusion input '" + arg.tfName + "' is a control endpoint";
      return status;
    }
  }
  std::unordered_set<std::string> seenOutputs;
  for (const FusionArg &output : info.outputs) {
    if (output.role != "output") {
      status.error = "fusion output '" + output.tfName +
                     "' has invalid role '" + output.role + "'";
      return status;
    }
    if (output.tfName.empty()) {
      status.error = missingFieldError("outputs.tf_name");
      return status;
    }
    if (output.rank < 0 ||
        output.rank != static_cast<int64_t>(output.shape.size())) {
      status.error = "fusion output '" + output.tfName +
                     "' rank does not match its shape";
      return status;
    }
    if (output.dtype.empty()) {
      status.error = missingFieldError("outputs.dtype");
      return status;
    }
    auto endpoint = parseTensorEndpoint(output.tfName);
    if (!endpoint.ok()) {
      status.error = endpoint.error;
      return status;
    }
    if (endpoint.value.control) {
      status.error =
          "fusion output '" + output.tfName + "' is a control endpoint";
      return status;
    }
    if (!seenOutputs.insert(endpoint.value.canonicalDataName()).second) {
      status.error = "duplicate fusion output endpoint '" +
                     endpoint.value.canonicalDataName() + "'";
      return status;
    }
  }
  return status;
}

// ── JSON source (mirrors readFusionMetadataJson / fusionInfosFromJson) ─────

Result<std::vector<FusionInfo>> readFusionMetadataJson(
    const std::string &path) {
  Result<std::vector<FusionInfo>> result;
  std::ifstream in(path);
  if (!in.is_open()) {
    result.error = "cannot open metadata JSON '" + path + "'";
    return result;
  }
  nlohmann::json json;
  try {
    in >> json;
  } catch (const std::exception &e) {
    result.error = "failed to parse metadata JSON '" + path + "': " + e.what();
    return result;
  }

  std::vector<FusionInfo> infos;
  if (json.is_object() && json.contains("fusions") &&
      json["fusions"].is_array()) {
    for (const auto &item : json["fusions"]) {
      auto info = fusionInfoFromJson(item);
      if (!info.ok()) {
        result.error = info.error;
        return result;
      }
      infos.push_back(std::move(info.value));
    }
    result.value = std::move(infos);
    return result;
  }

  auto info = fusionInfoFromJson(json);
  if (!info.ok()) {
    result.error = info.error;
    return result;
  }
  infos.push_back(std::move(info.value));
  result.value = std::move(infos);
  return result;
}

}  // namespace annc::converter_v2
