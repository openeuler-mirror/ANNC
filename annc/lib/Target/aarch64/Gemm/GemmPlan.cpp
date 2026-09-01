#include "GemmPlan.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Diagnostics.h"

namespace annc::aarch64::gemm {
namespace {

llvm::Error configError(llvm::StringRef path, llvm::StringRef message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid GEMM tuning config '%s': %s",
                                 path.str().c_str(), message.str().c_str());
}

std::optional<GemmTarget> parseTargetName(llvm::StringRef name) {
  if (name == "hip12") return GemmTarget::kHip12;
  if (name == "hip09") return GemmTarget::kHip09;
  return std::nullopt;
}

std::optional<GemmIsa> parseIsaName(llvm::StringRef name) {
  if (name == "neon") return GemmIsa::kNeon;
  if (name == "sve") return GemmIsa::kSve;
  return std::nullopt;
}

std::optional<GemmDataType> parseDataTypeName(llvm::StringRef name) {
  if (name == "f32") return GemmDataType::kF32;
  return std::nullopt;
}

std::optional<GemmExecutionKind> parseExecutionKindName(llvm::StringRef name) {
  if (name == "gemm") return GemmExecutionKind::kGemm;
  if (name == "gemv-ab") return GemmExecutionKind::kGemvAB;
  return std::nullopt;
}

std::optional<RhsPacking> parseRhsPackingName(llvm::StringRef name) {
  if (name == "direct") return RhsPacking::kDirect;
  if (name == "packed") return RhsPacking::kPacked;
  if (name == "row_major") return RhsPacking::kRowMajor;
  return std::nullopt;
}

std::optional<RhsPackSource> parseRhsPackSourceName(llvm::StringRef name) {
  if (name == "none") return RhsPackSource::kNone;
  if (name == "generated") return RhsPackSource::kGenerated;
  if (name == "prepacked") return RhsPackSource::kPrepacked;
  return std::nullopt;
}

llvm::Expected<int64_t> getConfigI64(const nlohmann::json &object,
                                     llvm::StringRef path,
                                     llvm::StringRef name) {
  auto it = object.find(name.str());
  if (it == object.end() || !it->is_number_integer() || it->get<int64_t>() <= 0)
    return configError(path, ("requires positive integer field " + name).str());
  return it->get<int64_t>();
}

llvm::Expected<nlohmann::json> getConfigObject(const nlohmann::json &object,
                                               llvm::StringRef path,
                                               llvm::StringRef name) {
  auto it = object.find(name.str());
  if (it == object.end() || !it->is_object())
    return configError(path, ("requires object field " + name).str());
  return *it;
}

mlir::FailureOr<int64_t> getPositiveI64(mlir::Operation *op,
                                        mlir::DictionaryAttr dictionary,
                                        llvm::StringRef name) {
  auto value = dictionary.getAs<mlir::IntegerAttr>(name);
  if (!value || value.getInt() <= 0) {
    op->emitOpError() << "requires positive integer field " << name;
    return mlir::failure();
  }
  return value.getInt();
}

mlir::LogicalResult requireString(mlir::Operation *op,
                                  mlir::DictionaryAttr dictionary,
                                  llvm::StringRef name,
                                  llvm::StringRef expected) {
  auto value = dictionary.getAs<mlir::StringAttr>(name);
  if (!value || value.getValue() != expected) {
    return op->emitOpError()
           << "requires string field " << name << " = \"" << expected << "\"";
  }
  return mlir::success();
}

mlir::LogicalResult requireThreadPartition(mlir::Operation *op,
                                           mlir::DictionaryAttr dictionary,
                                           bool allowLegacySerial) {
  auto value = dictionary.getAs<mlir::StringAttr>("thread_partition");
  const bool isStatic2D = value && value.getValue() == "static-2d";
  const bool isLegacySerial =
      allowLegacySerial && value && value.getValue() == "serial";
  if (!isStatic2D && !isLegacySerial) {
    return op->emitOpError()
           << "requires string field thread_partition = \"static-2d\"";
  }
  return mlir::success();
}

llvm::Expected<GemmTarget> parseConfigTarget(const nlohmann::json &config,
                                             llvm::StringRef path) {
  auto value = config.find("target_arch");
  if (value == config.end() || !value->is_string())
    return configError(path, "requires string field target_arch");
  const std::string target = value->get<std::string>();
  if (auto parsed = parseTargetName(target)) return *parsed;
  return configError(path, "has unsupported target_arch " + target);
}

llvm::Expected<GemmIsa> parseConfigIsa(const nlohmann::json &config,
                                       llvm::StringRef path) {
  auto value = config.find("isa");
  if (value == config.end() || !value->is_string())
    return configError(path, "requires string field isa");
  const std::string isa = value->get<std::string>();
  if (auto parsed = parseIsaName(isa)) return *parsed;
  return configError(path, "has unsupported isa " + isa);
}

llvm::Expected<GemmDataType> parseConfigDataType(const nlohmann::json &config,
                                                 llvm::StringRef path) {
  auto value = config.find("data_type");
  if (value == config.end() || !value->is_string())
    return configError(path, "requires string field data_type");
  const std::string dataType = value->get<std::string>();
  if (auto parsed = parseDataTypeName(dataType)) return *parsed;
  return configError(path, "has unsupported data_type " + dataType);
}

mlir::FailureOr<GemmTarget> parseTarget(mlir::Operation *op,
                                        mlir::DictionaryAttr dictionary,
                                        llvm::StringRef stateName) {
  auto value = dictionary.getAs<mlir::StringAttr>("target_arch");
  if (!value) {
    op->emitOpError() << "requires string field target_arch in " << stateName;
    return mlir::failure();
  }
  if (auto parsed = parseTargetName(value.getValue())) return *parsed;
  op->emitOpError() << "has unsupported target_arch " << value.getValue()
                    << " in " << stateName;
  return mlir::failure();
}

mlir::FailureOr<GemmIsa> parseIsa(mlir::Operation *op,
                                  mlir::DictionaryAttr dictionary,
                                  llvm::StringRef stateName) {
  auto value = dictionary.getAs<mlir::StringAttr>("isa");
  if (!value) {
    op->emitOpError() << "requires string field isa in " << stateName;
    return mlir::failure();
  }
  if (auto parsed = parseIsaName(value.getValue())) return *parsed;
  op->emitOpError() << "has unsupported isa " << value.getValue() << " in "
                    << stateName;
  return mlir::failure();
}

mlir::FailureOr<GemmDataType> parseDataType(mlir::Operation *op,
                                            mlir::DictionaryAttr dictionary,
                                            llvm::StringRef stateName) {
  auto value = dictionary.getAs<mlir::StringAttr>("data_type");
  if (!value) {
    op->emitOpError() << "requires string field data_type in " << stateName;
    return mlir::failure();
  }
  if (auto parsed = parseDataTypeName(value.getValue())) return *parsed;
  op->emitOpError() << "has unsupported data_type " << value.getValue()
                    << " in " << stateName;
  return mlir::failure();
}

}  // namespace

const GemmKernelABI &getGemmKernelABI(GemmTarget target, GemmIsa isa,
                                      GemmDataType dataType,
                                      GemmExecutionKind executionKind) {
  (void)target;
  (void)dataType;
  static const GemmKernelABI neon{"annc-neon-f32-v1", 16, 6, 4, 1};
  static const GemmKernelABI sve{"annc-sve-f32-v1", 32, 6, 4, 0};
  static const GemmKernelABI gemv{"annc-neon-gemv-ab-f32-v1", 16, 4, 1,
                                  4};
  if (executionKind == GemmExecutionKind::kGemvAB &&
      isa == GemmIsa::kNeon)
    return gemv;
  return isa == GemmIsa::kSve ? sve : neon;
}

llvm::FailureOr<int64_t> getGemmKScalarUnroll(const GemmKernelABI &abi,
                                              GemmDataType dataType) {
  const int64_t elementBytes = dataType == GemmDataType::kF32 ? 4 : 0;
  if (elementBytes == 0 || abi.vectorLengthBytes <= 0 ||
      abi.vectorLengthBytes % elementBytes != 0 || abi.kVectorUnroll <= 0)
    return mlir::failure();
  return abi.kVectorUnroll * (abi.vectorLengthBytes / elementBytes);
}

llvm::StringRef getGemmDataTypeName(GemmDataType dataType) {
  (void)dataType;
  return "f32";
}

llvm::FailureOr<int64_t> getGemmNr(const GemmKernelTile &kernelTile,
                                   int64_t vectorLengthBytes,
                                   GemmDataType dataType,
                                   GemmExecutionKind executionKind) {
  const int64_t elementBytes = dataType == GemmDataType::kF32 ? 4 : 0;
  if (elementBytes == 0 || vectorLengthBytes <= 0 ||
      vectorLengthBytes % elementBytes != 0 || kernelTile.panelLanes <= 0)
    return mlir::failure();
  if (executionKind == GemmExecutionKind::kGemvAB) {
    if (kernelTile.panelLanes != 1) return mlir::failure();
    return int64_t{1};
  }
  return kernelTile.panelLanes * (vectorLengthBytes / elementBytes);
}

llvm::Expected<GemmTuningConfig> loadGemmTuningConfig(llvm::StringRef path) {
  std::ifstream input(path.str());
  if (!input) return configError(path, "cannot open file");

  nlohmann::json config;
  try {
    input >> config;
    if (!config.is_object()) return configError(path, "root must be an object");
    if (config.value("version", 0) != 1)
      return configError(path, "requires version = 1");
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

    auto cache = getConfigObject(config, path, "cache_tile");
    auto kernel = getConfigObject(config, path, "kernel_tile");
    if (!cache || !kernel)
      return llvm::joinErrors(cache.takeError(), kernel.takeError());
    auto mc = getConfigI64(*cache, path, "mc");
    auto nc = getConfigI64(*cache, path, "nc");
    auto kc = getConfigI64(*cache, path, "kc");
    auto mr = getConfigI64(*kernel, path, "mr");
    auto panelLanes = getConfigI64(*kernel, path, "panel_lanes");
    if (!mc || !nc || !kc || !mr || !panelLanes) {
      llvm::Error error = llvm::Error::success();
      for (auto *value : {&mc, &nc, &kc, &mr, &panelLanes})
        if (!*value)
          error = llvm::joinErrors(std::move(error), value->takeError());
      return error;
    }
    return GemmTuningConfig{*target, *isa, *dataType,
                            GemmCacheTile{*mc, *nc, *kc},
                            GemmKernelTile{*mr, *panelLanes}};
  } catch (const std::exception &error) {
    return configError(path, error.what());
  }
}

bool isGemmAnchor(mlir::Operation *op) {
  return llvm::isa<mlir::linalg::MatmulOp>(op) ||
         (llvm::isa<mlir::linalg::GenericOp>(op) && op->hasAttr(kGemmAttrName));
}

mlir::Value getGemmInput(mlir::Operation *op, unsigned index) {
  if (auto matmul = llvm::dyn_cast<mlir::linalg::MatmulOp>(op))
    return matmul.getInputs()[index];
  return llvm::cast<mlir::linalg::GenericOp>(op).getInputs()[index];
}

mlir::Value getGemmOutput(mlir::Operation *op, unsigned index) {
  if (auto matmul = llvm::dyn_cast<mlir::linalg::MatmulOp>(op))
    return matmul.getOutputs()[index];
  return llvm::cast<mlir::linalg::GenericOp>(op).getOutputs()[index];
}

mlir::FailureOr<GemmProblem> readProblem(mlir::Operation *op) {
  auto problem = op->getAttrOfType<mlir::DictionaryAttr>(kProblemAttrName);
  if (!problem) {
    op->emitOpError() << "requires " << kProblemAttrName;
    return mlir::failure();
  }

  static constexpr llvm::StringLiteral fields[] = {"version", "m",   "n",  "k",
                                                   "lda",     "ldb", "ldc"};
  llvm::SmallVector<int64_t> values;
  values.reserve(std::size(fields));
  for (llvm::StringRef field : fields) {
    mlir::FailureOr<int64_t> value = getPositiveI64(op, problem, field);
    if (mlir::failed(value)) return mlir::failure();
    values.push_back(*value);
  }
  if (values[0] != kPlanVersion) {
    op->emitOpError() << "requires " << kProblemAttrName
                      << ".version = " << kPlanVersion;
    return mlir::failure();
  }
  return GemmProblem{values[0], values[1], values[2], values[3],
                     values[4], values[5], values[6]};
}

mlir::FailureOr<GemmCandidate> readCandidate(mlir::Operation *op) {
  auto candidate = op->getAttrOfType<mlir::DictionaryAttr>(kCandidateAttrName);
  if (!candidate) {
    op->emitOpError() << "requires " << kCandidateAttrName;
    return mlir::failure();
  }

  static constexpr llvm::StringLiteral fields[] = {
      "version", "mc", "nc", "kc", "mr", "panel_lanes", "thread_count"};
  llvm::SmallVector<int64_t> values;
  values.reserve(std::size(fields));
  for (llvm::StringRef field : fields) {
    mlir::FailureOr<int64_t> value = getPositiveI64(op, candidate, field);
    if (mlir::failed(value)) return mlir::failure();
    values.push_back(*value);
  }
  if (values[0] != kPlanVersion) {
    op->emitOpError() << "requires " << kCandidateAttrName
                      << ".version = " << kPlanVersion;
    return mlir::failure();
  }
  mlir::FailureOr<GemmTarget> target =
      parseTarget(op, candidate, kCandidateAttrName);
  mlir::FailureOr<GemmIsa> isa =
      parseIsa(op, candidate, kCandidateAttrName);
  mlir::FailureOr<GemmDataType> dataType =
      parseDataType(op, candidate, kCandidateAttrName);
  if (mlir::failed(target) || mlir::failed(isa) || mlir::failed(dataType))
    return mlir::failure();
  GemmExecutionKind executionKind = GemmExecutionKind::kGemm;
  if (auto value = candidate.getAs<mlir::StringAttr>(kExecutionKindAttrName)) {
    auto parsed = parseExecutionKindName(value.getValue());
    if (!parsed) {
      op->emitOpError() << "has unsupported execution_kind "
                        << value.getValue();
      return mlir::failure();
    }
    executionKind = *parsed;
  }
  const GemmKernelABI &abi =
      getGemmKernelABI(*target, *isa, *dataType, executionKind);
  if (mlir::failed(
          requireString(op, candidate, "kernel_family", abi.family)) ||
      mlir::failed(requireThreadPartition(op, candidate,
                                          /*allowLegacySerial=*/false))) {
    return mlir::failure();
  }
  GemmKernelTile kernelTile{values[4], values[5]};
  if (kernelTile.mr > abi.maxMr ||
      kernelTile.panelLanes > abi.maxPanelLanes ||
      mlir::failed(getGemmNr(kernelTile, abi.vectorLengthBytes, *dataType,
                             executionKind)) ||
      (executionKind == GemmExecutionKind::kGemvAB &&
       (kernelTile.mr != 4 || kernelTile.panelLanes != 1))) {
    op->emitOpError("has a candidate unsupported by the selected target and ISA");
    return mlir::failure();
  }
  return GemmCandidate{values[0],
                       *target,
                       *isa,
                       *dataType,
                       GemmCacheTile{values[1], values[2], values[3]},
                       kernelTile,
                       values[6],
                       executionKind};
}

mlir::FailureOr<GemmTilingPlan> readTilingPlan(mlir::Operation *op) {
  auto plan = op->getAttrOfType<mlir::DictionaryAttr>(kPlanAttrName);
  if (!plan) {
    op->emitOpError() << "requires " << kPlanAttrName;
    return mlir::failure();
  }

  static constexpr llvm::StringLiteral fields[] = {
      "version", "m",  "n",  "k",  "lda", "ldb", "ldc",
      "mc",      "nc", "kc", "mr", "panel_lanes",
      "vector_length_bytes", "thread_count"};
  llvm::SmallVector<int64_t> values;
  values.reserve(std::size(fields));
  for (llvm::StringRef field : fields) {
    mlir::FailureOr<int64_t> value = getPositiveI64(op, plan, field);
    if (mlir::failed(value)) return mlir::failure();
    values.push_back(*value);
  }

  if (values[0] != kPlanVersion) {
    op->emitOpError() << "requires " << kPlanAttrName
                      << ".version = " << kPlanVersion;
    return mlir::failure();
  }
  if (mlir::failed(requireString(op, plan, "macro_order", "mkn")) ||
      mlir::failed(requireString(op, plan, "first_kc_mode", "overwrite")) ||
      mlir::failed(requireString(op, plan, "next_kc_mode", "accumulate")) ||
      mlir::failed(requireThreadPartition(op, plan,
                                          /*allowLegacySerial=*/true))) {
    return mlir::failure();
  }
  GemmExecutionKind executionKind = GemmExecutionKind::kGemm;
  if (auto value = plan.getAs<mlir::StringAttr>(kExecutionKindAttrName)) {
    auto parsed = parseExecutionKindName(value.getValue());
    if (!parsed) {
      op->emitOpError() << "has unsupported execution_kind "
                        << value.getValue();
      return mlir::failure();
    }
    executionKind = *parsed;
  }
  RhsPacking rhsPacking = RhsPacking::kPacked;
  if (auto value = plan.getAs<mlir::StringAttr>(kRhsPackingAttrName)) {
    auto parsed = parseRhsPackingName(value.getValue());
    if (!parsed) {
      op->emitOpError() << "has unsupported rhs_packing " << value.getValue();
      return mlir::failure();
    }
    rhsPacking = *parsed;
  }
  RhsPackSource rhsPackSource = rhsPacking == RhsPacking::kPacked
                                    ? RhsPackSource::kGenerated
                                    : RhsPackSource::kNone;
  if (auto value = plan.getAs<mlir::StringAttr>(kRhsPackSourceAttrName)) {
    auto parsed = parseRhsPackSourceName(value.getValue());
    if (!parsed) {
      op->emitOpError() << "has unsupported rhs_pack_source "
                        << value.getValue();
      return mlir::failure();
    }
    rhsPackSource = *parsed;
  }
  if ((rhsPacking != RhsPacking::kPacked &&
       rhsPackSource != RhsPackSource::kNone) ||
      (rhsPacking == RhsPacking::kPacked &&
       rhsPackSource == RhsPackSource::kNone) ||
      (executionKind == GemmExecutionKind::kGemvAB &&
       (rhsPacking != RhsPacking::kDirect ||
        rhsPackSource != RhsPackSource::kNone)) ||
      (executionKind != GemmExecutionKind::kGemm &&
       rhsPacking == RhsPacking::kRowMajor)) {
    op->emitOpError("has an invalid execution/RHS representation combination");
    return mlir::failure();
  }
  mlir::FailureOr<GemmDataType> dataType =
      parseDataType(op, plan, kPlanAttrName);
  if (mlir::failed(dataType)) return mlir::failure();
  GemmKernelTile kernelTile{values[10], values[11]};
  if (mlir::failed(getGemmNr(kernelTile, values[12], *dataType,
                             executionKind))) {
    op->emitOpError("has an invalid GEMM kernel tile shape");
    return mlir::failure();
  }

  return GemmTilingPlan{
      values[0],
      values[1],
      values[2],
      values[3],
      values[4],
      values[5],
      values[6],
      GemmCacheTile{values[7], values[8], values[9]},
      kernelTile,
      *dataType,
      values[12],
      values[13],
      KcMode::kOverwrite,
      KcMode::kAccumulate,
      executionKind,
      rhsPacking,
      rhsPackSource};
}

mlir::FailureOr<GemmPlan> readPlan(mlir::Operation *op) {
  mlir::FailureOr<GemmTilingPlan> tiling = readTilingPlan(op);
  if (mlir::failed(tiling)) return mlir::failure();
  auto plan = op->getAttrOfType<mlir::DictionaryAttr>(kPlanAttrName);
  mlir::FailureOr<GemmTarget> target =
      parseTarget(op, plan, kPlanAttrName);
  mlir::FailureOr<GemmIsa> isa = parseIsa(op, plan, kPlanAttrName);
  if (mlir::failed(target) || mlir::failed(isa)) return mlir::failure();
  const GemmKernelABI &abi = getGemmKernelABI(
      *target, *isa, tiling->dataType, tiling->executionKind);
  if (tiling->vectorLengthBytes != abi.vectorLengthBytes ||
      tiling->kernelTile.mr > abi.maxMr ||
      tiling->kernelTile.panelLanes > abi.maxPanelLanes) {
    op->emitOpError("has a plan unsupported by the selected kernel ABI");
    return mlir::failure();
  }
  if (mlir::failed(requireString(op, plan, "kernel_family", abi.family)))
    return mlir::failure();
  if (tiling->executionKind == GemmExecutionKind::kGemvAB &&
      (tiling->kernelTile.mr != 4 || tiling->kernelTile.panelLanes != 1 ||
       *isa != GemmIsa::kNeon)) {
    op->emitOpError("has an invalid GEMV-AB kernel tile or ISA");
    return mlir::failure();
  }
  return GemmPlan{*tiling, *target, *isa};
}

llvm::StringRef getGemmTargetName(GemmTarget target) {
  return target == GemmTarget::kHip09 ? "hip09" : "hip12";
}

llvm::StringRef getGemmIsaName(GemmIsa isa) {
  return isa == GemmIsa::kSve ? "sve" : "neon";
}

llvm::StringRef getGemmExecutionKindName(GemmExecutionKind kind) {
  return kind == GemmExecutionKind::kGemvAB ? "gemv-ab" : "gemm";
}

llvm::StringRef getPackBAsmSymbol(GemmTarget target, GemmIsa isa) {
  (void)target;
  return isa == GemmIsa::kSve ? kSvePackBAsmSymbol : kNeonPackBAsmSymbol;
}

llvm::StringRef getKcModeName(KcMode mode) {
  return mode == KcMode::kOverwrite ? "overwrite" : "accumulate";
}

mlir::LogicalResult requireStage(mlir::Operation *op,
                                 llvm::StringRef expected) {
  auto stage = op->getAttrOfType<mlir::StringAttr>(kStageAttrName);
  if (!stage || stage.getValue() != expected) {
    return op->emitOpError()
           << "requires " << kStageAttrName << " = \"" << expected << "\"";
  }
  return mlir::success();
}

bool hasStage(mlir::Operation *op, llvm::StringRef expected) {
  auto stage = op->getAttrOfType<mlir::StringAttr>(kStageAttrName);
  return stage && stage.getValue() == expected;
}

void setStage(mlir::Operation *op, mlir::Builder &builder,
              llvm::StringRef stage) {
  op->setDiscardableAttr(kStageAttrName, builder.getStringAttr(stage));
}

}  // namespace annc::aarch64::gemm
