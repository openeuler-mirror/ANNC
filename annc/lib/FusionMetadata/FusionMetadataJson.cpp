#include "FusionMetadata/FusionMetadataJson.h"

#include <exception>
#include <fstream>
#include <system_error>

namespace annc::fusion {
namespace {

template <typename T>
std::vector<T> jsonValue(const nlohmann::json &json, const char *key,
                         std::vector<T> fallback = {}) {
  return json.value(key, fallback);
}

}  // namespace

nlohmann::json fusionInfoToJson(const FusionInfo &info) {
  nlohmann::json json;
  json["schema_version"] = info.schemaVersion;
  json["name"] = info.name;
  json["pattern"] = info.pattern;
  json["kernel_name"] = info.kernelName;
  json["output_tensor"] = info.outputTensor;
  json["original_nodes"] = info.originalNodes;
  json["args"] = nlohmann::json::array();
  for (const auto &arg : info.args) {
    json["args"].push_back({
        {"role", arg.role},
        {"tf_name", arg.tfName},
        {"shape", arg.shape},
        {"rank", arg.rank},
        {"dtype", arg.dtype},
    });
  }
  json["outputs"] = nlohmann::json::array();
  for (const auto &output : info.outputs) {
    json["outputs"].push_back({
        {"role", output.role},
        {"tf_name", output.tfName},
        {"shape", output.shape},
        {"rank", output.rank},
        {"dtype", output.dtype},
    });
  }
  json["pattern_attrs"] = info.patternAttrs;

  // Legacy fields consumed by the current GraphDef rewrite path.
  json["inputs"] = info.inputs;
  json["input_shapes"] = info.inputShapes;
  json["output_shape"] = info.outputShape;
  json["abi"] = info.abi.empty() ? "mlir_ciface" : info.abi;
  json["n_constants"] = info.nConstants;
  json["n_fixed"] = info.nFixed;
  json["n_dynamic"] = info.nDynamic;
  json["num_outputs"] = info.numOutputs;
  json["output_ranks"] = info.outputRanks;
  json["input_ranks"] = info.inputRanks;
  json["dynamic_dims"] = info.dynamicDims;
  json["kernel_arg_order"] = info.kernelArgOrder;
  json["symbolic_signature"] = info.symbolicSignature;
  json["fallback_function"] = info.fallbackFunction;
  return json;
}

nlohmann::json fusionInfosToJson(const std::vector<FusionInfo> &infos) {
  nlohmann::json result;
  result["schema_version"] = 1;
  result["fusions"] = nlohmann::json::array();
  for (const auto &info : infos) {
    result["fusions"].push_back(fusionInfoToJson(info));
  }
  return result;
}

llvm::Expected<FusionInfo> fusionInfoFromJson(const nlohmann::json &json) {
  if (!json.is_object()) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "fusion metadata entry is not an object");
  }

  FusionInfo info;
  try {
    info.schemaVersion = json.value("schema_version", 1);
    info.name = json.value("name", "");
    info.pattern = json.value("pattern", "");
    info.kernelName = json.value("kernel_name", "");
    info.outputTensor = json.value("output_tensor", "");
    info.originalNodes = jsonValue<std::string>(json, "original_nodes");
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

    info.inputs = jsonValue<std::string>(json, "inputs");
    info.inputShapes =
        json.value("input_shapes", std::vector<std::vector<int64_t>>{});
    info.outputShape = jsonValue<int64_t>(json, "output_shape");
    info.abi = json.value("abi", "mlir_ciface");
    if (info.abi.empty()) info.abi = "mlir_ciface";
    info.nConstants = json.value("n_constants", int64_t{0});
    info.nFixed = json.value("n_fixed", int64_t{0});
    info.nDynamic = json.value("n_dynamic", int64_t{0});
    info.numOutputs = json.value("num_outputs", int64_t{1});
    info.outputRanks = jsonValue<int64_t>(json, "output_ranks");
    info.inputRanks = jsonValue<int64_t>(json, "input_ranks");
    info.dynamicDims = jsonValue<int64_t>(json, "dynamic_dims");
    info.kernelArgOrder = jsonValue<int64_t>(json, "kernel_arg_order");
    info.symbolicSignature = json.value("symbolic_signature", "");
    info.fallbackFunction = json.value("fallback_function", "");
    normalizeFusionInfo(info);
  } catch (const std::exception &e) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "failed to parse fusion metadata JSON: %s",
                                   e.what());
  }

  if (auto err = validateFusionInfo(info)) return std::move(err);
  return info;
}

llvm::Expected<std::vector<FusionInfo>> fusionInfosFromJson(
    const nlohmann::json &json) {
  std::vector<FusionInfo> infos;
  if (json.is_object() && json.contains("fusions") &&
      json["fusions"].is_array()) {
    for (const auto &item : json["fusions"]) {
      auto info = fusionInfoFromJson(item);
      if (!info) return info.takeError();
      infos.push_back(std::move(*info));
    }
    return infos;
  }

  auto info = fusionInfoFromJson(json);
  if (!info) return info.takeError();
  infos.push_back(std::move(*info));
  return infos;
}

llvm::Error writeFusionMetadataJson(const std::vector<FusionInfo> &infos,
                                    const std::string &path) {
  std::ofstream out(path);
  if (!out.is_open()) {
    return llvm::createStringError(std::errc::io_error,
                                   "cannot open output file '%s'",
                                   path.c_str());
  }
  out << fusionInfosToJson(infos).dump(2) << "\n";
  return llvm::Error::success();
}

llvm::Expected<std::vector<FusionInfo>> readFusionMetadataJson(
    const std::string &path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    return llvm::createStringError(std::errc::io_error,
                                   "cannot open metadata JSON '%s'",
                                   path.c_str());
  }
  nlohmann::json json;
  try {
    in >> json;
  } catch (const std::exception &e) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "failed to parse metadata JSON '%s': %s",
                                   path.c_str(), e.what());
  }
  return fusionInfosFromJson(json);
}

}  // namespace annc::fusion
