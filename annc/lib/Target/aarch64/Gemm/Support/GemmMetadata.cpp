#include <array>
#include <cmath>
#include <limits>

#include "../GemmPlan.h"
#include "../GemmPacking.h"
#include "GemmStrategyConfig.h"
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
    llvm::StringRef stateName) {
  auto value = dictionary.getAs<mlir::StringAttr>("cache_strategy");
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
    llvm::StringRef stateName) {
  auto value = dictionary.getAs<mlir::StringAttr>("kernel_strategy");
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
  auto version = getPositiveI64(op, candidate, "version");
  if (mlir::failed(version)) return mlir::failure();
  if (*version != kCandidateVersion) {
    op->emitOpError() << "requires " << kCandidateAttrName
                      << ".version = " << kCandidateVersion;
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
  auto cacheParameters = candidate.getAs<mlir::DictionaryAttr>("cache_parameters");
  auto kernelParameters = candidate.getAs<mlir::DictionaryAttr>("kernel_parameters");
  if (!cacheParameters || !kernelParameters) {
    op->emitOpError("requires strategy-specific parameter dictionaries");
    return mlir::failure();
  }
  auto cache = decodeCacheStrategyParameters(op, *cacheStrategy, cacheParameters);
  auto kernel = decodeKernelStrategyParameters(op, *kernelStrategy, kernelParameters);
  if (mlir::failed(cache) || mlir::failed(kernel)) return mlir::failure();
  if (!getGemmKernelTilingStrategy(kernel->kind)
           .isValidForExecution(*kernel, *executionKind, *isa, *dataType,
                                abi)) {
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
  if (rhsPacking == RhsPacking::kPrepacked &&
      !isRhsPrepackSupported(*cacheStrategy, *kernelStrategy)) {
    op->emitOpError(
        "prepacked RHS currently requires fixed cache and kernel strategies");
    return mlir::failure();
  }
  return GemmTuningStrategy{
      *target,
      *isa,
      *dataType,
      *executionKind,
      *cache,
      *kernel,
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

mlir::FailureOr<GemmExecutionContext> readExecutionContext(
    mlir::Operation *op) {
  auto problem = readProblem(op);
  auto strategy = readStrategy(op);
  if (mlir::failed(problem) || mlir::failed(strategy))
    return mlir::failure();

  GemmExecutionContext context{*problem, *strategy};
  if (context.strategy.rhsPacking == RhsPacking::kPrepacked) {
    auto binding = op->getAttrOfType<mlir::DictionaryAttr>(kPrepackedRhsAttrName);
    if (!binding) {
      op->emitOpError("prepacked RHS requires a valid data binding");
      return mlir::failure();
    }
    auto symbol = binding.getAs<mlir::StringAttr>("data_symbol");
    auto elements = binding.getAs<mlir::IntegerAttr>("data_elements");
    if (!symbol || symbol.getValue().empty() || !elements ||
        elements.getInt() <= 0) {
      op->emitOpError("prepacked RHS requires a valid data binding");
      return mlir::failure();
    }
    if (!isRhsPrepackSupported(context.strategy.cache.kind,
                               context.strategy.kernel.kind)) {
      op->emitOpError(
          "prepacked RHS currently requires fixed cache and kernel strategies");
      return mlir::failure();
    }
    const auto &cache = getFixedCacheParameters(context.strategy.cache);
    int64_t expectedElements = 0;
    const GemmKernelABI &abi = getGemmKernelABI(context.strategy);
    for (const auto &block : partitionCache2D(
             problem->k, problem->n, cache.kc, cache.nc))
      expectedElements +=
          getGemmPackedBElements(abi, block.kSize, block.nSize);
    const std::pair<llvm::StringRef, int64_t> fields[] = {
        {"version", kPrepackedRhsVersion},
        {"k", problem->k},
        {"n", problem->n},
        {"kc", cache.kc},
        {"nc", cache.nc},
        {"data_elements", expectedElements}};
    for (const auto &[name, expected] : fields) {
      auto value = binding.getAs<mlir::IntegerAttr>(name);
      if (!value || value.getInt() != expected) {
        op->emitOpError() << "prepacked RHS binding field " << name
                          << " does not match the selected strategy";
        return mlir::failure();
      }
    }
    context.prepackedRhsSymbol = symbol.getValue().str();
    context.prepackedRhsElements = elements.getInt();
    context.prepackedRhsFromInput = binding.get("input") != nullptr;
  }
  return context;
}

mlir::FailureOr<GemmTuningStrategy> readStrategy(mlir::Operation *op) {
  auto reference = op->getAttrOfType<mlir::StringAttr>(kStrategyAttrName);
  if (!reference) {
    if (auto candidate =
            op->getAttrOfType<mlir::DictionaryAttr>(kCandidateAttrName))
      reference = candidate.getAs<mlir::StringAttr>("strategy");
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
      {builder.getNamedAttr("version", builder.getI64IntegerAttr(kCandidateVersion)),
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
       builder.getNamedAttr("cache_parameters",
                            encodeCacheStrategyParameters(builder, strategy.cache)),
       builder.getNamedAttr("kernel_parameters",
                            encodeKernelStrategyParameters(builder, strategy.kernel)),
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
                       "version", builder.getI64IntegerAttr(kCandidateVersion)),
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
      {builder.getNamedAttr("version", builder.getI64IntegerAttr(kCandidateVersion)),
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
       builder.getNamedAttr("cache_parameters",
                            encodeCacheStrategyParameters(builder, strategy->cache)),
       builder.getNamedAttr("kernel_parameters",
                            encodeKernelStrategyParameters(builder, strategy->kernel)),
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
  // The task grid depends on the packing cost of the selected strategy.
  attrs.set("planning_state", builder.getStringAttr("input"));
  op->setAttr(kCandidateAttrName, attrs.getDictionary(op->getContext()));
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
}

void copyGemmMetadata(mlir::Operation *source, mlir::Operation *destination) {
  auto *sourceOwner = metadataOwner(source);
  auto *destinationOwner = metadataOwner(destination);
  if (!sourceOwner || !destinationOwner || sourceOwner == destinationOwner)
    return;
  for (llvm::StringRef name : {kStrategiesAttrName, kCostModelsAttrName})
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
  auto strategy = readStrategy(op);
  if (mlir::succeeded(strategy)) {
    if (strategy->cache.kind == GemmCacheStrategyKind::kFixed) {
      const auto &tile = getFixedCacheParameters(strategy->cache);
      contract.append("nc", builder.getI64IntegerAttr(tile.nc));
      contract.append("kc", builder.getI64IntegerAttr(tile.kc));
    }
  }
  contract.append("k", builder.getI64IntegerAttr(k));
  contract.append("n", builder.getI64IntegerAttr(n));
  contract.append("data_symbol", builder.getStringAttr(symbol));
  contract.append("data_elements", builder.getI64IntegerAttr(elements));
  if (fromInput) contract.append("input", builder.getUnitAttr());
  return contract.getDictionary(builder.getContext());
}

void pruneGemmMetadata(mlir::Operation *owner) {
  llvm::StringSet<> strategies, costs;
  auto retain = [](llvm::StringSet<> &ids, mlir::Attribute attr) {
    if (auto reference = llvm::dyn_cast_or_null<mlir::StringAttr>(attr))
      ids.insert(reference.getValue());
  };
  owner->walk([&](mlir::Operation *op) {
    if (metadataOwner(op) != owner) return;
    retain(strategies, op->getAttr(kStrategyAttrName));
    if (auto candidate =
            op->getAttrOfType<mlir::DictionaryAttr>(kCandidateAttrName)) {
      retain(strategies, candidate.get("strategy"));
      retain(costs, candidate.get("cost_model"));
    }
  });
  for (auto [name, live] :
       {std::pair<llvm::StringRef, llvm::StringSet<> *>{kStrategiesAttrName, &strategies},
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
