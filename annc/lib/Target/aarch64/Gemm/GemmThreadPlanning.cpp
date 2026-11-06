#include <string>

#include "GemmPlan.h"
#include "GemmThreadPlanner.h"
#include "Target/aarch64/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace annc {
namespace {

using aarch64::gemm::GemmCacheTile;
using aarch64::gemm::GemmPlanningPolicy;
using aarch64::gemm::GemmProblem;
using aarch64::gemm::GemmThreadPlannerConfig;
using aarch64::gemm::GemmThreadPlanningResult;

LogicalResult planGemmThreads(Operation *op) {
  FailureOr<GemmProblem> problem = aarch64::gemm::readProblem(op);
  FailureOr<GemmPlanningPolicy> policy = aarch64::gemm::readPlanningPolicy(op);
  if (failed(problem) || failed(policy)) return failure();

  const aarch64::gemm::GemmTuningStrategy &strategy = policy->strategy;
  const aarch64::gemm::GemmKernelABI &abi = aarch64::gemm::getGemmKernelABI(
      strategy.target, strategy.isa, strategy.dataType, strategy.executionKind);
  FailureOr<int64_t> nr = aarch64::gemm::getGemmNr(
      strategy.kernelTile, abi.vectorLengthBytes, strategy.dataType,
      strategy.executionKind);
  if (failed(nr))
    return op->emitOpError("cannot derive NR for GEMM thread planning");
  const aarch64::gemm::GemmDataTypeInfo &dataType =
      aarch64::gemm::getGemmDataTypeInfo(strategy.dataType);
  if (abi.vectorLengthBytes % dataType.rhsBytes != 0)
    return op->emitOpError("has a vector length incompatible with data type");

  const int64_t packetSize = abi.vectorLengthBytes / dataType.rhsBytes;
  // NEON emits one vector for a partial panel. The SVE packer emits four
  // predicated streams per K row ([panel][4][K][VL]), even when the final
  // output group uses fewer than four vectors.
  const int64_t rhsMemoryAlignment =
      strategy.isa == aarch64::gemm::GemmIsa::kSve ? 4 * packetSize
                                                   : packetSize;
  const bool packRhs =
      strategy.rhsPacking == aarch64::gemm::RhsPacking::kPacked;
  GemmThreadPlannerConfig plannerConfig{
      problem->m,
      problem->n,
      problem->k,
      strategy.maxThreadCount,
      strategy.kernelTile.mr,
      *nr,
      packetSize,
      rhsMemoryAlignment,
      packRhs,
      // Only MatVec vectorizes K. VecMat vectorizes output columns, like
      // ordinary GEMM, so only MatrixVector uses the reduction cost model.
      strategy.executionKind == aarch64::gemm::GemmExecutionKind::kMatrixVector,
      dataType.lhsBytes,
      dataType.rhsBytes,
      dataType.outputBytes,
      policy->costModel,
      GemmCacheTile{strategy.cacheTile.mc, strategy.cacheTile.nc,
                    strategy.cacheTile.kc}};
  GemmThreadPlanningResult result;
  std::string error;
  if (!aarch64::gemm::planGemmThreads(plannerConfig, result, error))
    return op->emitOpError() << "cannot compute AArch64 GEMM plan: " << error;

  Builder builder(op->getContext());
  auto candidate =
      op->getAttrOfType<DictionaryAttr>(aarch64::gemm::kCandidateAttrName);
  NamedAttrList planned(candidate);
  planned.set("planning_state", builder.getStringAttr("planned"));
  planned.set("thread_count", builder.getI64IntegerAttr(result.thread_count));
  planned.set("tasks_m", builder.getI64IntegerAttr(result.tasks_m));
  planned.set("tasks_n", builder.getI64IntegerAttr(result.tasks_n));
  planned.set(
      "shard_direction",
      builder.getStringAttr(result.shard_by_columns ? "columns" : "rows"));
  planned.set("thread_partition", builder.getStringAttr("static-2d"));
  op->setDiscardableAttr(aarch64::gemm::kCandidateAttrName,
                         planned.getDictionary(op->getContext()));
  return success();
}

class AArch64GemmThreadPlanning
    : public AArch64GemmThreadPlanningBase<AArch64GemmThreadPlanning> {
 public:
  using Base::Base;

  void runOnOperation() override {
    WalkResult result = getOperation().walk([](Operation *op) -> WalkResult {
      if (!aarch64::gemm::isGemmAnchor(op)) return WalkResult::advance();
      auto candidate =
          op->getAttrOfType<DictionaryAttr>(aarch64::gemm::kCandidateAttrName);
      if (!candidate) return WalkResult::advance();
      auto state = candidate.getAs<StringAttr>("planning_state");
      if (!state || state.getValue() != "policy") return WalkResult::advance();
      return failed(planGemmThreads(op)) ? WalkResult::interrupt()
                                         : WalkResult::advance();
    });
    if (result.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAArch64GemmThreadPlanning() {
  return std::make_unique<AArch64GemmThreadPlanning>();
}

}  // namespace annc
