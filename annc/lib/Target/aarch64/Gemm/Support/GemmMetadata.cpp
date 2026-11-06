#include <array>
#include <cmath>
#include <limits>

#include "../GemmPlan.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"

namespace annc::aarch64::gemm {
namespace {

mlir::Operation *metadataOwner(mlir::Operation *op) {
  if (llvm::isa<mlir::func::FuncOp, mlir::ModuleOp>(op)) return op;
  if (auto function = op->getParentOfType<mlir::func::FuncOp>())
    return function;
  return op->getParentOfType<mlir::ModuleOp>().getOperation();
}

mlir::StringAttr intern(mlir::Operation *owner, llvm::StringRef tableName,
                        llvm::StringRef prefix, mlir::DictionaryAttr value) {
  mlir::NamedAttrList table(
      owner->getAttrOfType<mlir::DictionaryAttr>(tableName));
  for (auto entry : table)
    if (entry.getValue() == value) return entry.getName();
  unsigned index = 0;
  std::string id;
  do {
    id = (prefix + llvm::Twine(index++)).str();
  } while (table.get(id));
  mlir::Builder builder(owner->getContext());
  table.set(id, value);
  owner->setAttr(tableName, table.getDictionary(owner->getContext()));
  return builder.getStringAttr(id);
}

mlir::DictionaryAttr lookup(mlir::Operation *op, llvm::StringRef tableName,
                            mlir::StringAttr reference) {
  mlir::Operation *owner = metadataOwner(op);
  auto table = owner ? owner->getAttrOfType<mlir::DictionaryAttr>(tableName)
                     : mlir::DictionaryAttr();
  auto result = table && reference
                    ? table.getAs<mlir::DictionaryAttr>(reference.getValue())
                    : mlir::DictionaryAttr();
  if (!result)
    op->emitOpError() << "invalid GEMM reference " << reference << " in "
                      << tableName;
  return result;
}

mlir::DictionaryAttr planDictionary(mlir::Operation *op) {
  if (auto legacy = op->getAttrOfType<mlir::DictionaryAttr>(kPlanAttrName))
    return legacy;
  auto plan = lookup(op, kPlansAttrName,
                     op->getAttrOfType<mlir::StringAttr>(kPlanAttrName));
  if (!plan) return {};
  auto strategy =
      lookup(op, kStrategiesAttrName, plan.getAs<mlir::StringAttr>("strategy"));
  if (!strategy) return {};
  mlir::NamedAttrList merged(strategy);
  for (auto attr : plan)
    if (attr.getName() != "strategy")
      merged.set(attr.getName(), attr.getValue());
  return merged.getDictionary(op->getContext());
}

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

mlir::FailureOr<GemmCacheStrategyKind> parseCacheStrategy(
    mlir::Operation *op, mlir::DictionaryAttr dictionary,
    llvm::StringRef stateName, bool allowLegacyFixed = false) {
  auto value = dictionary.getAs<mlir::StringAttr>("cache_strategy");
  if (!value && allowLegacyFixed) return GemmCacheStrategyKind::kFixed;
  if (!value) {
    op->emitOpError() << "requires string field cache_strategy in "
                      << stateName;
    return mlir::failure();
  }
  if (auto parsed = parseGemmCacheStrategyName(value.getValue()))
    return *parsed;
  op->emitOpError() << "has unsupported cache_strategy " << value.getValue()
                    << " in " << stateName;
  return mlir::failure();
}

mlir::FailureOr<GemmKernelStrategyKind> parseKernelStrategy(
    mlir::Operation *op, mlir::DictionaryAttr dictionary,
    llvm::StringRef stateName, bool allowLegacyFixed = false) {
  auto value = dictionary.getAs<mlir::StringAttr>("kernel_strategy");
  if (!value && allowLegacyFixed) return GemmKernelStrategyKind::kFixed;
  if (!value) {
    op->emitOpError() << "requires string field kernel_strategy in "
                      << stateName;
    return mlir::failure();
  }
  if (auto parsed = parseGemmKernelStrategyName(value.getValue()))
    return *parsed;
  op->emitOpError() << "has unsupported kernel_strategy " << value.getValue()
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
  if (values[0] != kProblemVersion) {
    op->emitOpError() << "requires " << kProblemAttrName
                      << ".version = " << kProblemVersion;
    return mlir::failure();
  }
  return GemmProblem{values[0], values[1], values[2], values[3],
                     values[4], values[5], values[6]};
}

mlir::FailureOr<GemmTuningStrategy> readTuningStrategy(
    mlir::Operation *op, mlir::DictionaryAttr candidate) {
  if (!candidate) return mlir::failure();
  static constexpr llvm::StringLiteral fields[] = {
      "version", "mc", "nc", "kc", "mr", "panel_lanes"};
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
  auto cacheStrategy = parseCacheStrategy(op, candidate, kCandidateAttrName);
  auto kernelStrategy = parseKernelStrategy(op, candidate, kCandidateAttrName);
  auto executionKind = parseExecutionKind(op, candidate, kCandidateAttrName);
  if (mlir::failed(target) || mlir::failed(isa) || mlir::failed(dataType) ||
      mlir::failed(cacheStrategy) || mlir::failed(kernelStrategy) ||
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
      (*executionKind != GemmExecutionKind::kGemm &&
       (kernelTile.mr != abi.maxMr ||
        kernelTile.panelLanes != abi.maxPanelLanes ||
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
  return GemmTuningStrategy{
      *target,
      *isa,
      *dataType,
      *executionKind,
      GemmCacheStrategy{*cacheStrategy, {values[1], values[2], values[3]}},
      GemmKernelStrategy{*kernelStrategy, kernelTile},
      rhsPacking};
}

mlir::FailureOr<GemmThreadPlanningInput> readThreadPlanningInput(
    mlir::Operation *op) {
  auto candidate = op->getAttrOfType<mlir::DictionaryAttr>(kCandidateAttrName);
  if (!candidate) {
    op->emitOpError() << "requires " << kCandidateAttrName;
    return mlir::failure();
  }
  if (mlir::failed(requireString(op, candidate, "planning_state", "input")))
    return mlir::failure();
  auto strategy = readTuningStrategy(
      op, lookup(op, kStrategiesAttrName,
                 candidate.getAs<mlir::StringAttr>("strategy")));
  if (mlir::failed(strategy)) return mlir::failure();
  auto maxThreadCount = getPositiveI64(op, candidate, "max_thread_count");
  if (mlir::failed(maxThreadCount)) return mlir::failure();

  auto costModel = lookup(op, kCostModelsAttrName,
                          candidate.getAs<mlir::StringAttr>("cost_model"));
  if (!costModel) return mlir::failure();

  static constexpr llvm::StringLiteral costFields[] = {
      "load_cycles_per_byte",    "store_cycles_per_byte",
      "compute_cycle_scale",     "gemv_compute_cycle_scale",
      "microkernel_call_cycles", "pack_cycles_per_byte",
      "pack_call_cycles"};
  std::array<double, 7> cost{};
  for (std::size_t index = 0; index < std::size(costFields); ++index) {
    auto value = costModel.getAs<mlir::FloatAttr>(costFields[index]);
    if (!value || !std::isfinite(value.getValueAsDouble()) ||
        value.getValueAsDouble() <= 0.0) {
      op->emitOpError() << "requires finite positive field "
                        << costFields[index];
      return mlir::failure();
    }
    cost[index] = value.getValueAsDouble();
  }
  return GemmThreadPlanningInput{
      *strategy,
      GemmThreadPlannerCostModel{cost[0], cost[1], cost[2], cost[3], cost[4],
                                 cost[5], cost[6]},
      *maxThreadCount};
}

mlir::FailureOr<GemmCandidate> readCandidate(mlir::Operation *op) {
  auto candidate = op->getAttrOfType<mlir::DictionaryAttr>(kCandidateAttrName);
  if (!candidate) {
    op->emitOpError() << "requires " << kCandidateAttrName;
    return mlir::failure();
  }
  if (mlir::failed(requireString(op, candidate, "planning_state", "planned")))
    return mlir::failure();
  auto strategy = readTuningStrategy(
      op, lookup(op, kStrategiesAttrName,
                 candidate.getAs<mlir::StringAttr>("strategy")));
  if (mlir::failed(strategy)) return mlir::failure();
  auto maxThreadCount = getPositiveI64(op, candidate, "max_thread_count");
  if (mlir::failed(maxThreadCount)) return mlir::failure();

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
      *maxThreadCount < values[0] ||
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
  auto plan = planDictionary(op);
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

  const bool legacyPlan =
      op->getAttrOfType<mlir::DictionaryAttr>(kPlanAttrName) != nullptr;
  const bool supportsLegacyPlan = legacyPlan && values[0] == 1;
  if (values[0] != kPlanVersion && !supportsLegacyPlan) {
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
  mlir::FailureOr<GemmCacheStrategyKind> cacheStrategy =
      parseCacheStrategy(op, plan, kPlanAttrName, supportsLegacyPlan);
  mlir::FailureOr<GemmKernelStrategyKind> kernelStrategy =
      parseKernelStrategy(op, plan, kPlanAttrName, supportsLegacyPlan);
  if (mlir::failed(cacheStrategy) || mlir::failed(kernelStrategy))
    return mlir::failure();
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
      values[6],          *cacheStrategy,
      GemmCacheTile{values[7], values[8], values[9]}, *kernelStrategy,
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
  auto plan = planDictionary(op);
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
  GemmPlan result{*tiling, *target, *isa, std::move(prepackedSymbol),
                  prepackedElements};
  result.prepackedRhsFromInput = plan.get("rhs_data_input") != nullptr;
  return result;
}

mlir::FailureOr<GemmTuningStrategy> readStrategy(mlir::Operation *op) {
  auto reference = op->getAttrOfType<mlir::StringAttr>(kStrategyAttrName);
  if (!reference) {
    if (auto candidate =
            op->getAttrOfType<mlir::DictionaryAttr>(kCandidateAttrName))
      reference = candidate.getAs<mlir::StringAttr>("strategy");
    else if (auto plan = op->getAttrOfType<mlir::StringAttr>(kPlanAttrName)) {
      auto entry = lookup(op, kPlansAttrName, plan);
      if (entry) reference = entry.getAs<mlir::StringAttr>("strategy");
    }
  }
  if (!reference && op->getAttrOfType<mlir::DictionaryAttr>(kPlanAttrName)) {
    auto plan = readPlan(op);
    if (mlir::failed(plan)) return mlir::failure();
    return GemmTuningStrategy{
        plan->target,
        plan->isa,
        plan->dataType,
        plan->executionKind,
        {plan->cacheStrategyKind, plan->cacheTile},
        {plan->kernelStrategyKind, plan->kernelTile},
        plan->rhsPacking};
  }
  auto strategy = lookup(op, kStrategiesAttrName, reference);
  if (!strategy) return mlir::failure();
  return readTuningStrategy(op, strategy);
}

void writeProblem(mlir::Operation *op, const GemmProblem &problem) {
  mlir::Builder builder(op->getContext());
  op->setAttr(
      kProblemAttrName,
      builder.getDictionaryAttr(
          {builder.getNamedAttr("version",
                                builder.getI64IntegerAttr(kProblemVersion)),
           builder.getNamedAttr("m", builder.getI64IntegerAttr(problem.m)),
           builder.getNamedAttr("n", builder.getI64IntegerAttr(problem.n)),
           builder.getNamedAttr("k", builder.getI64IntegerAttr(problem.k)),
           builder.getNamedAttr("lda", builder.getI64IntegerAttr(problem.lda)),
           builder.getNamedAttr("ldb", builder.getI64IntegerAttr(problem.ldb)),
           builder.getNamedAttr("ldc",
                                builder.getI64IntegerAttr(problem.ldc))}));
}

void writeThreadPlanningInput(mlir::Operation *op,
                              const GemmThreadPlanningInput &input) {
  mlir::Builder builder(op->getContext());
  const auto &strategy = input.strategy;
  const auto &abi = getGemmKernelABI(strategy);
  auto strategyAttr = builder.getDictionaryAttr(
      {builder.getNamedAttr("version", builder.getI64IntegerAttr(kPlanVersion)),
       builder.getNamedAttr(
           "target_arch",
           builder.getStringAttr(getGemmTargetName(strategy.target))),
       builder.getNamedAttr(
           "isa", builder.getStringAttr(getGemmIsaName(strategy.isa))),
       builder.getNamedAttr(
           "data_type",
           builder.getStringAttr(getGemmDataTypeName(strategy.dataType))),
       builder.getNamedAttr("cache_strategy",
                            builder.getStringAttr(
                                getGemmCacheStrategyName(strategy.cache.kind))),
       builder.getNamedAttr("kernel_strategy",
                            builder.getStringAttr(getGemmKernelStrategyName(
                                strategy.kernel.kind))),
       builder.getNamedAttr("kernel_family", builder.getStringAttr(abi.family)),
       builder.getNamedAttr("mc",
                            builder.getI64IntegerAttr(strategy.cache.tile.mc)),
       builder.getNamedAttr("nc",
                            builder.getI64IntegerAttr(strategy.cache.tile.nc)),
       builder.getNamedAttr("kc",
                            builder.getI64IntegerAttr(strategy.cache.tile.kc)),
       builder.getNamedAttr("mr",
                            builder.getI64IntegerAttr(strategy.kernel.tile.mr)),
       builder.getNamedAttr(
           "panel_lanes",
           builder.getI64IntegerAttr(strategy.kernel.tile.panelLanes)),
       builder.getNamedAttr("execution_kind",
                            builder.getStringAttr(getGemmExecutionKindName(
                                strategy.executionKind))),
       builder.getNamedAttr(
           "rhs_packing",
           builder.getStringAttr(
               strategy.rhsPacking == RhsPacking::kDirect   ? "direct"
               : strategy.rhsPacking == RhsPacking::kPacked ? "packed"
                                                            : "prepacked"))});
  const auto &cost = input.costModel;
  auto costAttr = builder.getDictionaryAttr(
      {builder.getNamedAttr("load_cycles_per_byte",
                            builder.getF64FloatAttr(cost.load_cycles_per_byte)),
       builder.getNamedAttr(
           "store_cycles_per_byte",
           builder.getF64FloatAttr(cost.store_cycles_per_byte)),
       builder.getNamedAttr("compute_cycle_scale",
                            builder.getF64FloatAttr(cost.compute_cycle_scale)),
       builder.getNamedAttr(
           "gemv_compute_cycle_scale",
           builder.getF64FloatAttr(cost.gemv_compute_cycle_scale)),
       builder.getNamedAttr(
           "microkernel_call_cycles",
           builder.getF64FloatAttr(cost.microkernel_call_cycles)),
       builder.getNamedAttr("pack_cycles_per_byte",
                            builder.getF64FloatAttr(cost.pack_cycles_per_byte)),
       builder.getNamedAttr("pack_call_cycles",
                            builder.getF64FloatAttr(cost.pack_call_cycles))});
  mlir::Operation *owner = metadataOwner(op);
  auto strategyRef = intern(owner, kStrategiesAttrName, "s", strategyAttr);
  auto costRef = intern(owner, kCostModelsAttrName, "c", costAttr);
  op->setAttr(kCandidateAttrName,
              builder.getDictionaryAttr(
                  {builder.getNamedAttr(
                       "version", builder.getI64IntegerAttr(kPlanVersion)),
                   builder.getNamedAttr("planning_state",
                                        builder.getStringAttr("input")),
                   builder.getNamedAttr("strategy", strategyRef),
                   builder.getNamedAttr("cost_model", costRef),
                   builder.getNamedAttr(
                       "max_thread_count",
                       builder.getI64IntegerAttr(input.maxThreadCount))}));
}

void writeCandidate(mlir::Operation *op,
                    const GemmThreadPlanningResult &result) {
  mlir::Builder builder(op->getContext());
  auto input = op->getAttrOfType<mlir::DictionaryAttr>(kCandidateAttrName);
  mlir::NamedAttrList candidate(input);
  candidate.set("planning_state", builder.getStringAttr("planned"));
  candidate.set("thread_count", builder.getI64IntegerAttr(result.thread_count));
  candidate.set("tasks_m", builder.getI64IntegerAttr(result.tasks_m));
  candidate.set("tasks_n", builder.getI64IntegerAttr(result.tasks_n));
  candidate.set(
      "shard_direction",
      builder.getStringAttr(result.shard_by_columns ? "columns" : "rows"));
  candidate.set("thread_partition", builder.getStringAttr("static-2d"));
  op->setAttr(kCandidateAttrName, candidate.getDictionary(op->getContext()));
}

void setCandidateRhsPacking(mlir::Operation *op, RhsPacking packing) {
  auto input = op->getAttrOfType<mlir::DictionaryAttr>(kCandidateAttrName);
  if (!input) return;
  auto strategy =
      readTuningStrategy(op, lookup(op, kStrategiesAttrName,
                                    input.getAs<mlir::StringAttr>("strategy")));
  if (mlir::failed(strategy)) return;
  strategy->rhsPacking = packing;
  mlir::Builder builder(op->getContext());
  const auto &abi = getGemmKernelABI(*strategy);
  auto strategyAttr = builder.getDictionaryAttr(
      {builder.getNamedAttr("version", builder.getI64IntegerAttr(kPlanVersion)),
       builder.getNamedAttr(
           "target_arch",
           builder.getStringAttr(getGemmTargetName(strategy->target))),
       builder.getNamedAttr(
           "isa", builder.getStringAttr(getGemmIsaName(strategy->isa))),
       builder.getNamedAttr(
           "data_type",
           builder.getStringAttr(getGemmDataTypeName(strategy->dataType))),
       builder.getNamedAttr("cache_strategy",
                            builder.getStringAttr(getGemmCacheStrategyName(
                                strategy->cache.kind))),
       builder.getNamedAttr("kernel_strategy",
                            builder.getStringAttr(getGemmKernelStrategyName(
                                strategy->kernel.kind))),
       builder.getNamedAttr("kernel_family", builder.getStringAttr(abi.family)),
       builder.getNamedAttr("mc",
                            builder.getI64IntegerAttr(strategy->cache.tile.mc)),
       builder.getNamedAttr("nc",
                            builder.getI64IntegerAttr(strategy->cache.tile.nc)),
       builder.getNamedAttr("kc",
                            builder.getI64IntegerAttr(strategy->cache.tile.kc)),
       builder.getNamedAttr(
           "mr", builder.getI64IntegerAttr(strategy->kernel.tile.mr)),
       builder.getNamedAttr(
           "panel_lanes",
           builder.getI64IntegerAttr(strategy->kernel.tile.panelLanes)),
       builder.getNamedAttr("execution_kind",
                            builder.getStringAttr(getGemmExecutionKindName(
                                strategy->executionKind))),
       builder.getNamedAttr(
           "rhs_packing",
           builder.getStringAttr(packing == RhsPacking::kDirect ? "direct"
                                 : packing == RhsPacking::kPacked
                                     ? "packed"
                                     : "prepacked"))});
  auto reference =
      intern(metadataOwner(op), kStrategiesAttrName, "s", strategyAttr);
  mlir::NamedAttrList attrs(input);
  attrs.set("strategy", reference);
  op->setAttr(kCandidateAttrName, attrs.getDictionary(op->getContext()));
}

void writePlan(mlir::Operation *op, mlir::DictionaryAttr plan) {
  mlir::NamedAttrList planEntry(plan);
  if (auto candidate =
          op->getAttrOfType<mlir::DictionaryAttr>(kCandidateAttrName))
    if (auto strategy = candidate.get("strategy"))
      planEntry.set("strategy", strategy);
  for (llvm::StringRef field :
       {"target_arch", "isa", "data_type", "kernel_family", "execution_kind",
        "mc", "nc", "kc", "mr", "panel_lanes", "rhs_packing"})
    planEntry.erase(field);
  auto reference = intern(metadataOwner(op), kPlansAttrName, "p",
                          planEntry.getDictionary(op->getContext()));
  op->setAttr(kPlanAttrName, reference);
  op->removeAttr(kProblemAttrName);
  op->removeAttr(kCandidateAttrName);
  op->removeAttr(kPrepackedRhsAttrName);
  op->removeAttr(kRhsNameAttrName);
}

void copyStrategyReference(mlir::Operation *source, mlir::Operation *dest) {
  if (auto reference = source->getAttr(kStrategyAttrName)) {
    dest->setAttr(kStrategyAttrName, reference);
    return;
  }
  if (auto candidate =
          source->getAttrOfType<mlir::DictionaryAttr>(kCandidateAttrName)) {
    if (auto reference = candidate.get("strategy"))
      dest->setAttr(kStrategyAttrName, reference);
    return;
  }
  auto planRef = source->getAttrOfType<mlir::StringAttr>(kPlanAttrName);
  auto plan = planRef
                  ? lookup(source, kPlansAttrName, planRef)
                  : source->getAttrOfType<mlir::DictionaryAttr>(kPlanAttrName);
  if (plan)
    if (auto reference = plan.get("strategy"))
      dest->setAttr(kStrategyAttrName, reference);
}

void copyGemmMetadata(mlir::Operation *source, mlir::Operation *destination) {
  auto *sourceOwner = metadataOwner(source);
  auto *destinationOwner = metadataOwner(destination);
  if (!sourceOwner || !destinationOwner || sourceOwner == destinationOwner)
    return;
  for (llvm::StringRef name :
       {kStrategiesAttrName, kCostModelsAttrName, kPlansAttrName})
    if (auto attr = sourceOwner->getAttr(name))
      destinationOwner->setAttr(name, attr);
}

mlir::DictionaryAttr encodePrepackedRhsBinding(mlir::Operation *op, int64_t k,
                                               int64_t n,
                                               llvm::StringRef symbol,
                                               int64_t elements,
                                               bool fromInput) {
  mlir::Builder builder(op->getContext());
  mlir::NamedAttrList contract;
  contract.append("version", builder.getI64IntegerAttr(kPrepackedRhsVersion));
  if (auto candidate =
          op->getAttrOfType<mlir::DictionaryAttr>(kCandidateAttrName)) {
    auto strategy = lookup(op, kStrategiesAttrName,
                           candidate.getAs<mlir::StringAttr>("strategy"));
    if (auto nc = strategy.getAs<mlir::IntegerAttr>("nc"))
      contract.append("nc", nc);
    if (auto kc = strategy.getAs<mlir::IntegerAttr>("kc"))
      contract.append("kc", kc);
  }
  contract.append("k", builder.getI64IntegerAttr(k));
  contract.append("n", builder.getI64IntegerAttr(n));
  contract.append("data_symbol", builder.getStringAttr(symbol));
  contract.append("data_elements", builder.getI64IntegerAttr(elements));
  if (fromInput) contract.append("input", builder.getUnitAttr());
  return contract.getDictionary(builder.getContext());
}

void pruneGemmMetadata(mlir::Operation *owner) {
  llvm::StringSet<> plans, strategies, costs;
  auto retain = [](llvm::StringSet<> &ids, mlir::Attribute attr) {
    if (auto reference = llvm::dyn_cast_or_null<mlir::StringAttr>(attr))
      ids.insert(reference.getValue());
  };
  owner->walk([&](mlir::Operation *op) {
    if (metadataOwner(op) != owner) return;
    retain(plans, op->getAttr(kPlanAttrName));
    retain(strategies, op->getAttr(kStrategyAttrName));
    if (auto candidate =
            op->getAttrOfType<mlir::DictionaryAttr>(kCandidateAttrName)) {
      retain(strategies, candidate.get("strategy"));
      retain(costs, candidate.get("cost_model"));
    }
  });
  if (auto table = owner->getAttrOfType<mlir::DictionaryAttr>(kPlansAttrName))
    for (auto entry : table)
      if (plans.contains(entry.getName().getValue()))
        if (auto plan = llvm::dyn_cast<mlir::DictionaryAttr>(entry.getValue()))
          retain(strategies, plan.get("strategy"));
  for (auto [name, live] :
       {std::pair<llvm::StringRef, llvm::StringSet<> *>{kPlansAttrName, &plans},
        {kStrategiesAttrName, &strategies},
        {kCostModelsAttrName, &costs}}) {
    auto table = owner->getAttrOfType<mlir::DictionaryAttr>(name);
    if (!table) continue;
    mlir::NamedAttrList retained;
    for (auto entry : table)
      if (live->contains(entry.getName().getValue())) retained.append(entry);
    if (retained.empty())
      owner->removeAttr(name);
    else
      owner->setAttr(name, retained.getDictionary(owner->getContext()));
  }
}

void pruneGemmMetadata(mlir::ModuleOp module) {
  module.walk(
      [&](mlir::func::FuncOp function) { pruneGemmMetadata(function); });
  pruneGemmMetadata(module.getOperation());
}

}  // namespace annc::aarch64::gemm
