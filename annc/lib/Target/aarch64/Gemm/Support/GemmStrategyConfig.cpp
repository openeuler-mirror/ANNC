#include "GemmStrategyConfig.h"

#include "../GemmPlan.h"

namespace annc::aarch64::gemm {
namespace {

llvm::Error configError(llvm::StringRef configPath, llvm::StringRef message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid GEMM tuning config '%s': %s",
                                 configPath.str().c_str(), message.str().c_str());
}

llvm::Expected<int64_t> getPositiveInteger(const nlohmann::json &parameters,
                                           llvm::StringRef configPath,
                                           llvm::StringRef name) {
  auto value = parameters.find(name.str());
  if (value == parameters.end() || !value->is_number_integer() ||
      value->get<int64_t>() <= 0)
    return configError(configPath,
                       ("requires positive integer field " + name).str());
  return value->get<int64_t>();
}

}  // namespace

namespace {

mlir::FailureOr<int64_t> getPositiveParameter(mlir::Operation *op,
                                               mlir::DictionaryAttr parameters,
                                               llvm::StringRef name) {
  auto value = parameters.getAs<mlir::IntegerAttr>(name);
  if (!value || value.getInt() <= 0) {
    op->emitOpError() << "requires positive strategy parameter " << name;
    return mlir::failure();
  }
  return value.getInt();
}

}  // namespace

mlir::DictionaryAttr encodeCacheStrategyParameters(
    mlir::Builder &builder, const GemmCacheStrategy &strategy) {
  mlir::NamedAttrList parameters;
  switch (strategy.kind) {
    case GemmCacheStrategyKind::kFixed: {
      const auto &fixed = getFixedCacheParameters(strategy);
      parameters.append("mc", builder.getI64IntegerAttr(fixed.mc));
      parameters.append("nc", builder.getI64IntegerAttr(fixed.nc));
      parameters.append("kc", builder.getI64IntegerAttr(fixed.kc));
      break;
    }
    default:
      llvm_unreachable("unknown GEMM cache strategy");
  }
  return parameters.getDictionary(builder.getContext());
}

mlir::DictionaryAttr encodeKernelStrategyParameters(
    mlir::Builder &builder, const GemmKernelStrategy &strategy) {
  mlir::NamedAttrList parameters;
  switch (strategy.kind) {
    case GemmKernelStrategyKind::kFixed: {
      const auto &fixed = getFixedKernelParameters(strategy);
      parameters.append("mr", builder.getI64IntegerAttr(fixed.mr));
      parameters.append("panel_lanes",
                        builder.getI64IntegerAttr(fixed.panelLanes));
      break;
    }
    default:
      llvm_unreachable("unknown GEMM kernel strategy");
  }
  return parameters.getDictionary(builder.getContext());
}

mlir::FailureOr<GemmCacheStrategy> decodeCacheStrategyParameters(
    mlir::Operation *op, GemmCacheStrategyKind kind,
    mlir::DictionaryAttr parameters) {
  switch (kind) {
    case GemmCacheStrategyKind::kFixed: {
      auto mc = getPositiveParameter(op, parameters, "mc");
      auto nc = getPositiveParameter(op, parameters, "nc");
      auto kc = getPositiveParameter(op, parameters, "kc");
      if (mlir::failed(mc) || mlir::failed(nc) || mlir::failed(kc))
        return mlir::failure();
      return GemmCacheStrategy{kind, FixedCacheParameters{*mc, *nc, *kc}};
    }
  }
  llvm_unreachable("unknown GEMM cache strategy");
}

mlir::FailureOr<GemmKernelStrategy> decodeKernelStrategyParameters(
    mlir::Operation *op, GemmKernelStrategyKind kind,
    mlir::DictionaryAttr parameters) {
  switch (kind) {
    case GemmKernelStrategyKind::kFixed: {
      auto mr = getPositiveParameter(op, parameters, "mr");
      auto panelLanes = getPositiveParameter(op, parameters, "panel_lanes");
      if (mlir::failed(mr) || mlir::failed(panelLanes))
        return mlir::failure();
      return GemmKernelStrategy{kind, FixedKernelParameters{*mr, *panelLanes}};
    }
  }
  llvm_unreachable("unknown GEMM kernel strategy");
}

llvm::Expected<GemmCacheStrategy> parseGemmCacheStrategyConfig(
    llvm::StringRef name, const nlohmann::json &parameters,
    llvm::StringRef configPath) {
  if (name != "fixed")
    return configError(configPath,
                       ("has unsupported cache strategy " + name).str());
  if (!parameters.is_object())
    return configError(configPath,
                       "cache strategy parameters must be an object");
  auto mc = getPositiveInteger(parameters, configPath, "mc");
  auto nc = getPositiveInteger(parameters, configPath, "nc");
  auto kc = getPositiveInteger(parameters, configPath, "kc");
  if (!mc || !nc || !kc) {
    llvm::Error error = llvm::Error::success();
    if (!mc) error = llvm::joinErrors(std::move(error), mc.takeError());
    if (!nc) error = llvm::joinErrors(std::move(error), nc.takeError());
    if (!kc) error = llvm::joinErrors(std::move(error), kc.takeError());
    return error;
  }
  return GemmCacheStrategy{GemmCacheStrategyKind::kFixed,
                           FixedCacheParameters{*mc, *nc, *kc}};
}

llvm::Expected<GemmKernelStrategy> parseGemmKernelStrategyConfig(
    llvm::StringRef name, const nlohmann::json &parameters,
    llvm::StringRef configPath) {
  if (name != "fixed")
    return configError(configPath,
                       ("has unsupported kernel strategy " + name).str());
  if (!parameters.is_object())
    return configError(configPath,
                       "kernel strategy parameters must be an object");
  auto mr = getPositiveInteger(parameters, configPath, "mr");
  auto panelLanes = getPositiveInteger(parameters, configPath, "panel_lanes");
  if (!mr || !panelLanes) {
    llvm::Error error = llvm::Error::success();
    if (!mr) error = llvm::joinErrors(std::move(error), mr.takeError());
    if (!panelLanes)
      error = llvm::joinErrors(std::move(error), panelLanes.takeError());
    return error;
  }
  return GemmKernelStrategy{GemmKernelStrategyKind::kFixed,
                            FixedKernelParameters{*mr, *panelLanes}};
}

}  // namespace annc::aarch64::gemm
