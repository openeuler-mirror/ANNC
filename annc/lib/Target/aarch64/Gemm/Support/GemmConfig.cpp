#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>

#include "../GemmPlan.h"

namespace annc::aarch64::gemm {
namespace {

llvm::Error configError(llvm::StringRef path, llvm::StringRef message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid GEMM tuning config '%s': %s",
                                 path.str().c_str(), message.str().c_str());
}

llvm::Expected<int64_t> getConfigI64(const nlohmann::json &object,
                                     llvm::StringRef path,
                                     llvm::StringRef name) {
  auto it = object.find(name.str());
  if (it == object.end() || !it->is_number_integer() || it->get<int64_t>() <= 0)
    return configError(path, ("requires positive integer field " + name).str());
  return it->get<int64_t>();
}

llvm::Expected<double> getConfigF64(const nlohmann::json &object,
                                    llvm::StringRef path,
                                    llvm::StringRef name) {
  auto it = object.find(name.str());
  if (it == object.end() || !it->is_number())
    return configError(path, ("requires numeric field " + name).str());
  const double value = it->get<double>();
  if (!std::isfinite(value) || value <= 0.0)
    return configError(path, ("requires finite positive field " + name).str());
  return value;
}

llvm::Expected<std::string> getConfigString(const nlohmann::json &object,
                                            llvm::StringRef path,
                                            llvm::StringRef name) {
  auto it = object.find(name.str());
  if (it == object.end() || !it->is_string())
    return configError(path, ("requires string field " + name).str());
  return it->get<std::string>();
}

llvm::Expected<nlohmann::json> getConfigObject(const nlohmann::json &object,
                                               llvm::StringRef path,
                                               llvm::StringRef name) {
  auto it = object.find(name.str());
  if (it == object.end() || !it->is_object())
    return configError(path, ("requires object field " + name).str());
  return *it;
}

llvm::Expected<GemmTarget> parseConfigTarget(const nlohmann::json &config,
                                             llvm::StringRef path) {
  auto value = config.find("target_arch");
  if (value == config.end() || !value->is_string())
    return configError(path, "requires string field target_arch");
  const std::string target = value->get<std::string>();
  if (auto parsed = parseGemmTargetName(target)) return *parsed;
  return configError(path, "has unsupported target_arch " + target);
}

llvm::Expected<GemmIsa> parseConfigIsa(const nlohmann::json &config,
                                       llvm::StringRef path) {
  auto value = config.find("isa");
  if (value == config.end() || !value->is_string())
    return configError(path, "requires string field isa");
  const std::string isa = value->get<std::string>();
  if (auto parsed = parseGemmIsaName(isa)) return *parsed;
  return configError(path, "has unsupported isa " + isa);
}

llvm::Expected<GemmDataType> parseConfigDataType(const nlohmann::json &config,
                                                 llvm::StringRef path) {
  auto value = config.find("data_type");
  if (value == config.end() || !value->is_string())
    return configError(path, "requires string field data_type");
  const std::string dataType = value->get<std::string>();
  if (auto parsed = parseGemmDataTypeName(dataType)) return *parsed;
  return configError(path, "has unsupported data_type " + dataType);
}

}  // namespace

std::optional<GemmTarget> parseGemmTargetName(llvm::StringRef name) {
  if (name == "hip12") return GemmTarget::kHip12;
  if (name == "hip09") return GemmTarget::kHip09;
  if (name == "kp920b") return GemmTarget::kKp920b;
  if (name == "kp950") return GemmTarget::kKp950;
  return std::nullopt;
}

std::optional<GemmIsa> parseGemmIsaName(llvm::StringRef name) {
  if (name == "neon") return GemmIsa::kNeon;
  if (name == "sve") return GemmIsa::kSve;
  return std::nullopt;
}

std::optional<GemmDataType> parseGemmDataTypeName(llvm::StringRef name) {
  if (name == "f32") return GemmDataType::kF32;
  return std::nullopt;
}

llvm::Expected<GemmTuningConfig> loadGemmTuningConfig(llvm::StringRef path) {
  std::ifstream input(path.str());
  if (!input) return configError(path, "cannot open file");

  nlohmann::json config;
  try {
    input >> config;
    if (!config.is_object()) return configError(path, "root must be an object");
    const int64_t version = config.value("version", 0);
    if (version != kTuningConfigVersion)
      return configError(path, "requires version = 3");
    auto target = parseConfigTarget(config, path);
    auto isa = parseConfigIsa(config, path);
    auto dataType = parseConfigDataType(config, path);
    if (!target || !isa || !dataType) {
      llvm::Error error = llvm::Error::success();
      if (!target)
        error = llvm::joinErrors(std::move(error), target.takeError());
      if (!isa) error = llvm::joinErrors(std::move(error), isa.takeError());
      if (!dataType)
        error = llvm::joinErrors(std::move(error), dataType.takeError());
      return error;
    }

    auto cache = getConfigObject(config, path, "cache");
    auto kernel = getConfigObject(config, path, "kernel");
    if (!cache || !kernel)
      return llvm::joinErrors(cache.takeError(), kernel.takeError());
    auto cacheName = getConfigString(*cache, path, "strategy");
    auto kernelName = getConfigString(*kernel, path, "strategy");
    if (!cacheName || !kernelName)
      return llvm::joinErrors(cacheName.takeError(), kernelName.takeError());
    auto cacheKind = parseGemmCacheStrategyName(*cacheName);
    auto kernelKind = parseGemmKernelStrategyName(*kernelName);
    if (!cacheKind)
      return configError(path, "has unsupported cache strategy " + *cacheName);
    if (!kernelKind)
      return configError(path,
                         "has unsupported kernel strategy " + *kernelName);

    auto cacheParams = getConfigObject(*cache, path, *cacheName);
    auto kernelParams = getConfigObject(*kernel, path, *kernelName);
    if (!cacheParams || !kernelParams)
      return llvm::joinErrors(cacheParams.takeError(),
                              kernelParams.takeError());
    auto mc = getConfigI64(*cacheParams, path, "mc");
    auto nc = getConfigI64(*cacheParams, path, "nc");
    auto kc = getConfigI64(*cacheParams, path, "kc");
    auto mr = getConfigI64(*kernelParams, path, "mr");
    auto panelLanes = getConfigI64(*kernelParams, path, "panel_lanes");
    if (!mc || !nc || !kc || !mr || !panelLanes) {
      llvm::Error error = llvm::Error::success();
      for (auto *value : {&mc, &nc, &kc, &mr, &panelLanes})
        if (!*value)
          error = llvm::joinErrors(std::move(error), value->takeError());
      return error;
    }
    auto costModel = getConfigObject(config, path, "cost_model");
    if (!costModel) return costModel.takeError();
    auto load = getConfigF64(*costModel, path, "load_cycles_per_byte");
    auto store = getConfigF64(*costModel, path, "store_cycles_per_byte");
    auto compute = getConfigF64(*costModel, path, "compute_cycle_scale");
    auto gemvCompute =
        getConfigF64(*costModel, path, "gemv_compute_cycle_scale");
    auto microkernelCall =
        getConfigF64(*costModel, path, "microkernel_call_cycles");
    auto packBytes = getConfigF64(*costModel, path, "pack_cycles_per_byte");
    auto packCall = getConfigF64(*costModel, path, "pack_call_cycles");
    if (!load || !store || !compute || !gemvCompute || !microkernelCall ||
        !packBytes || !packCall)
      return configError(path, "has invalid cost_model");
    return GemmTuningConfig{
        *target,
        *isa,
        *dataType,
        GemmCacheStrategy{*cacheKind, {*mc, *nc, *kc}},
        GemmKernelStrategy{*kernelKind, {*mr, *panelLanes}},
        GemmThreadPlannerCostModel{*load, *store, *compute, *gemvCompute,
                                   *microkernelCall, *packBytes, *packCall}};
  } catch (const std::exception &error) {
    return configError(path, error.what());
  }
}

}  // namespace annc::aarch64::gemm
