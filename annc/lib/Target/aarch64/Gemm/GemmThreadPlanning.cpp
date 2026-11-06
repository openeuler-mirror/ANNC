#include <string>

#include "GemmPacking.h"
#include "GemmPlan.h"
#include "GemmThreadPlanner.h"
#include "Target/aarch64/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace annc {
namespace {

using aarch64::gemm::GemmProblem;
using aarch64::gemm::GemmThreadPlannerConfig;
using aarch64::gemm::GemmThreadPlanningInput;
using aarch64::gemm::GemmThreadPlanningResult;

LogicalResult planGemmThreads(Operation *op) {
  FailureOr<GemmProblem> problem = aarch64::gemm::readProblem(op);
  FailureOr<GemmThreadPlanningInput> policy =
      aarch64::gemm::readThreadPlanningInput(op);
  if (failed(problem) || failed(policy)) return failure();

  const aarch64::gemm::GemmTuningStrategy &strategy = policy->strategy;
  const aarch64::gemm::GemmKernelABI &abi =
      aarch64::gemm::getGemmKernelABI(strategy);
  FailureOr<int64_t> nr =
      aarch64::gemm::getGemmNr(strategy.kernel.tile, abi.vectorLengthBytes,
                               strategy.dataType, strategy.executionKind);
  if (failed(nr))
    return op->emitOpError("cannot derive NR for GEMM thread planning");
  const aarch64::gemm::GemmDataTypeInfo &dataType =
      aarch64::gemm::getGemmDataTypeInfo(strategy.dataType);
  if (abi.vectorLengthBytes % dataType.rhsBytes != 0)
    return op->emitOpError("has a vector length incompatible with data type");

  const int64_t packetSize = abi.vectorLengthBytes / dataType.rhsBytes;
  const int64_t rhsMemoryAlignment =
      aarch64::gemm::getGemmPackedBAlignment(abi);
  const bool packRhs =
      strategy.rhsPacking == aarch64::gemm::RhsPacking::kPacked;
  GemmThreadPlannerConfig plannerConfig{
      problem->m,
      problem->n,
      problem->k,
      policy->maxThreadCount,
      strategy.kernel.tile.mr,
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
      strategy.cache.tile};
  GemmThreadPlanningResult result;
  std::string error;
  if (!aarch64::gemm::planGemmThreads(plannerConfig, result, error))
    return op->emitOpError() << "cannot compute AArch64 GEMM plan: " << error;

  aarch64::gemm::writeCandidate(op, result);
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
      if (!state || state.getValue() != "input") return WalkResult::advance();
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
