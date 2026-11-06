#include <array>
#include <cmath>
#include <limits>

#include "../GemmPlan.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Diagnostics.h"

namespace annc::aarch64::gemm {
namespace {

std::optional<GemmExecutionKind> parseExecutionKindName(llvm::StringRef name) {
  if (name == "gemm") return GemmExecutionKind::kGemm;
  if (name == "matrix-vector") return GemmExecutionKind::kMatrixVector;
  if (name == "vector-matrix") return GemmExecutionKind::kVectorMatrix;
  return std::nullopt;
}

std::optional<RhsPacking> parseRhsPackingName(llvm::StringRef name) {
  if (name == "direct") return RhsPacking::kDirect;
  if (name == "packed") return RhsPacking::kPacked;
  if (name == "prepacked") return RhsPacking::kPrepacked;
  return std::nullopt;
}

std::optional<RhsPackSource> parseRhsPackSourceName(llvm::StringRef name) {
  if (name == "none") return RhsPackSource::kNone;
  if (name == "generated") return RhsPackSource::kGenerated;
  if (name == "prepacked") return RhsPackSource::kPrepacked;
  return std::nullopt;
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

mlir::FailureOr<GemmTarget> parseTarget(mlir::Operation *op,
                                        mlir::DictionaryAttr dictionary,
                                        llvm::StringRef stateName) {
  auto value = dictionary.getAs<mlir::StringAttr>("target_arch");
  if (!value) {
    op->emitOpError() << "requires string field target_arch in " << stateName;
    return mlir::failure();
  }
  if (auto parsed = parseGemmTargetName(value.getValue())) return *parsed;
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
  if (auto parsed = parseGemmIsaName(value.getValue())) return *parsed;
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
  if (auto parsed = parseGemmDataTypeName(value.getValue())) return *parsed;
  op->emitOpError() << "has unsupported data_type " << value.getValue()
                    << " in " << stateName;
  return mlir::failure();
}

mlir::FailureOr<GemmExecutionKind> parseExecutionKind(
    mlir::Operation *op, mlir::DictionaryAttr dictionary,
    llvm::StringRef stateName) {
  auto value = dictionary.getAs<mlir::StringAttr>(kExecutionKindAttrName);
  if (!value) {
    op->emitOpError() << "requires string field " << kExecutionKindAttrName
                      << " in " << stateName;
    return mlir::failure();
  }
  if (auto parsed = parseExecutionKindName(value.getValue())) return *parsed;
  op->emitOpError() << "has unsupported execution_kind " << value.getValue()
                    << " in " << stateName;
  return mlir::failure();
}

}  // namespace

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

mlir::FailureOr<GemmTuningStrategy> readTuningStrategy(
    mlir::Operation *op, mlir::DictionaryAttr candidate) {
  static constexpr llvm::StringLiteral fields[] = {
      "version", "mc", "nc", "kc", "mr", "panel_lanes", "max_thread_count"};
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
  auto target = parseTarget(op, candidate, kCandidateAttrName);
  auto isa = parseIsa(op, candidate, kCandidateAttrName);
  auto dataType = parseDataType(op, candidate, kCandidateAttrName);
  auto executionKind = parseExecutionKind(op, candidate, kCandidateAttrName);
  if (mlir::failed(target) || mlir::failed(isa) || mlir::failed(dataType) ||
      mlir::failed(executionKind))
    return mlir::failure();
  const GemmKernelABI &abi =
      getGemmKernelABI(*target, *isa, *dataType, *executionKind);
  if (mlir::failed(requireString(op, candidate, "kernel_family", abi.family)))
    return mlir::failure();
  GemmKernelTile kernelTile{values[4], values[5]};
  if (kernelTile.mr > abi.maxMr || kernelTile.panelLanes > abi.maxPanelLanes ||
      mlir::failed(getGemmNr(kernelTile, abi.vectorLengthBytes, *dataType,
                             *executionKind)) ||
      (*executionKind == GemmExecutionKind::kMatrixVector &&
       (kernelTile.mr != 4 || kernelTile.panelLanes != 1 ||
        *isa != GemmIsa::kNeon)) ||
      (*executionKind == GemmExecutionKind::kVectorMatrix &&
       (kernelTile.mr != 1 || kernelTile.panelLanes != 4 ||
        *isa != GemmIsa::kNeon))) {
    op->emitOpError("has a tuning strategy unsupported by the kernel ABI");
    return mlir::failure();
  }
  RhsPacking rhsPacking = RhsPacking::kPacked;
  if (auto value = candidate.getAs<mlir::StringAttr>(kRhsPackingAttrName)) {
    auto parsed = parseRhsPackingName(value.getValue());
    if (!parsed) {
      op->emitOpError() << "has unsupported rhs_packing " << value.getValue();
      return mlir::failure();
    }
    rhsPacking = *parsed;
  }
  if (*executionKind != GemmExecutionKind::kGemm &&
      rhsPacking != RhsPacking::kDirect) {
    op->emitOpError("non-GEMM execution requires direct RHS");
    return mlir::failure();
  }
  return GemmTuningStrategy{*target,
                            *isa,
                            *dataType,
                            *executionKind,
                            GemmCacheTile{values[1], values[2], values[3]},
                            kernelTile,
                            values[6],
                            rhsPacking};
}

mlir::FailureOr<GemmPlanningPolicy> readPlanningPolicy(mlir::Operation *op) {
  auto candidate = op->getAttrOfType<mlir::DictionaryAttr>(kCandidateAttrName);
  if (!candidate) {
    op->emitOpError() << "requires " << kCandidateAttrName;
    return mlir::failure();
  }
  if (mlir::failed(requireString(op, candidate, "planning_state", "policy")))
    return mlir::failure();
  auto strategy = readTuningStrategy(op, candidate);
  if (mlir::failed(strategy)) return mlir::failure();

  static constexpr llvm::StringLiteral costFields[] = {
      "load_cycles_per_byte",    "store_cycles_per_byte",
      "compute_cycle_scale",     "gemv_compute_cycle_scale",
      "microkernel_call_cycles", "pack_cycles_per_byte",
      "pack_call_cycles"};
  std::array<double, 7> cost{};
  for (std::size_t index = 0; index < std::size(costFields); ++index) {
    auto value = candidate.getAs<mlir::FloatAttr>(costFields[index]);
    if (!value || !std::isfinite(value.getValueAsDouble()) ||
        value.getValueAsDouble() <= 0.0) {
      op->emitOpError() << "requires finite positive field "
                        << costFields[index];
      return mlir::failure();
    }
    cost[index] = value.getValueAsDouble();
  }
  return GemmPlanningPolicy{
      *strategy, GemmThreadPlannerCostModel{cost[0], cost[1], cost[2], cost[3],
                                            cost[4], cost[5], cost[6]}};
}

mlir::FailureOr<GemmCandidate> readCandidate(mlir::Operation *op) {
  auto candidate = op->getAttrOfType<mlir::DictionaryAttr>(kCandidateAttrName);
  if (!candidate) {
    op->emitOpError() << "requires " << kCandidateAttrName;
    return mlir::failure();
  }
  auto strategy = readTuningStrategy(op, candidate);
  if (mlir::failed(strategy)) return mlir::failure();

  static constexpr llvm::StringLiteral fields[] = {"thread_count", "tasks_m",
                                                   "tasks_n"};
  llvm::SmallVector<int64_t> values;
  values.reserve(std::size(fields));
  for (llvm::StringRef field : fields) {
    mlir::FailureOr<int64_t> value = getPositiveI64(op, candidate, field);
    if (mlir::failed(value)) return mlir::failure();
    values.push_back(*value);
  }
  auto shard = candidate.getAs<mlir::StringAttr>("shard_direction");
  const bool shardByColumns = shard && shard.getValue() == "columns";
  if (!shard || (!shardByColumns && shard.getValue() != "rows") ||
      strategy->maxThreadCount < values[0] ||
      values[1] > std::numeric_limits<int64_t>::max() / values[2] ||
      values[1] * values[2] != values[0] ||
      mlir::failed(
          requireString(op, candidate, "thread_partition", "static-2d"))) {
    op->emitOpError("has an inconsistent static task topology");
    return mlir::failure();
  }
  return GemmCandidate{*strategy, values[0], values[1], values[2],
                       shardByColumns};
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

  for (llvm::StringRef field :
       {llvm::StringRef("tasks_m"), llvm::StringRef("tasks_n")}) {
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
      mlir::failed(requireString(op, plan, "thread_partition", "static-2d"))) {
    return mlir::failure();
  }
  auto shard = plan.getAs<mlir::StringAttr>("shard_direction");
  const bool shardByColumns = shard && shard.getValue() == "columns";
  if (!shard || (!shardByColumns && shard.getValue() != "rows") ||
      values[14] > values[1] || values[15] > values[2] ||
      values[14] > std::numeric_limits<int64_t>::max() / values[15] ||
      values[14] * values[15] != values[13]) {
    op->emitOpError("has an inconsistent finalized task topology");
    return mlir::failure();
  }
  mlir::FailureOr<GemmExecutionKind> executionKind =
      parseExecutionKind(op, plan, kPlanAttrName);
  if (mlir::failed(executionKind)) return mlir::failure();
  auto rhsPackingValue = plan.getAs<mlir::StringAttr>(kRhsPackingAttrName);
  if (!rhsPackingValue) {
    op->emitOpError() << "requires string field " << kRhsPackingAttrName
                      << " in " << kPlanAttrName;
    return mlir::failure();
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
                                    : rhsPacking == RhsPacking::kPrepacked
                                          ? RhsPackSource::kPrepacked
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
  const bool sourceMatches =
      (rhsPacking == RhsPacking::kDirect &&
       rhsPackSource == RhsPackSource::kNone) ||
      (rhsPacking == RhsPacking::kPacked &&
       rhsPackSource == RhsPackSource::kGenerated) ||
      (rhsPacking == RhsPacking::kPrepacked &&
       rhsPackSource == RhsPackSource::kPrepacked);
  if (!sourceMatches ||
      (executionKind != GemmExecutionKind::kGemm &&
       rhsPacking != RhsPacking::kDirect)) {
    op->emitOpError("has an invalid execution/RHS representation combination");
    return mlir::failure();
  }
  mlir::FailureOr<GemmDataType> dataType =
      parseDataType(op, plan, kPlanAttrName);
  if (mlir::failed(dataType)) return mlir::failure();
  GemmKernelTile kernelTile{values[10], values[11]};
  if (mlir::failed(
          getGemmNr(kernelTile, values[12], *dataType, *executionKind))) {
    op->emitOpError("has an invalid GEMM kernel tile shape");
    return mlir::failure();
  }

  return GemmTilingPlan{
      values[0],          values[1],
      values[2],          values[3],
      values[4],          values[5],
      values[6],          GemmCacheTile{values[7], values[8], values[9]},
      kernelTile,         *dataType,
      values[12],         values[13],
      values[14],         values[15],
      shardByColumns,
      KcMode::kOverwrite, KcMode::kAccumulate,
      *executionKind,     rhsPacking,
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
  std::string prepackedSymbol;
  int64_t prepackedElements = 0;
  if (tiling->rhsPacking == RhsPacking::kPrepacked) {
    auto symbol = plan.getAs<mlir::StringAttr>("rhs_data_symbol");
    auto elements = plan.getAs<mlir::IntegerAttr>("rhs_data_elements");
    if (!symbol || symbol.getValue().empty() || !elements ||
        elements.getInt() <= 0) {
      op->emitOpError(
          "prepacked RHS requires rhs_data_symbol and rhs_data_elements");
      return mlir::failure();
    }
    prepackedSymbol = symbol.getValue().str();
    prepackedElements = elements.getInt();
  }
  if (tiling->executionKind == GemmExecutionKind::kMatrixVector &&
      (tiling->kernelTile.mr != 4 || tiling->kernelTile.panelLanes != 1 ||
       *isa != GemmIsa::kNeon)) {
    op->emitOpError("has an invalid matrix-vector kernel tile or ISA");
    return mlir::failure();
  }
  if (tiling->executionKind == GemmExecutionKind::kVectorMatrix &&
      (tiling->m != 1 || tiling->kernelTile.mr != 1 ||
       tiling->kernelTile.panelLanes != 4 || *isa != GemmIsa::kNeon)) {
    op->emitOpError(
        "has an invalid vector-matrix problem, kernel tile, or ISA");
    return mlir::failure();
  }
  return GemmPlan{*tiling, *target, *isa, std::move(prepackedSymbol),
                  prepackedElements};
}

}  // namespace annc::aarch64::gemm
