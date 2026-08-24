#include "GemmPlan.h"
#include "Target/aarch64/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <cstdlib>
#include <cstring>

namespace annc {
namespace {

// Conservative automatic gate.  Larger shapes remain on the established
// packed path until target-specific benchmark data justifies widening it.
constexpr int64_t kRowMajorOperationLimit = 4500;

void appendI64(NamedAttrList &attrs, Builder &builder, llvm::StringRef name,
               int64_t value) {
  attrs.append(name, builder.getI64IntegerAttr(value));
}

void appendString(NamedAttrList &attrs, Builder &builder, llvm::StringRef name,
                  llvm::StringRef value) {
  attrs.append(name, builder.getStringAttr(value));
}

LogicalResult finalizeGemmPlan(Operation *op) {
  if (op->hasAttr(aarch64::gemm::kPlanAttrName))
    return op->emitOpError("already has a finalized AArch64 GEMM plan");

  FailureOr<aarch64::gemm::GemmProblem> problem =
      aarch64::gemm::readProblem(op);
  FailureOr<aarch64::gemm::GemmCandidate> candidate =
      aarch64::gemm::readCandidate(op);
  if (failed(problem) || failed(candidate)) return failure();

  Builder builder(op->getContext());
  NamedAttrList plan;
  appendI64(plan, builder, "version", aarch64::gemm::kPlanVersion);
  appendString(plan, builder, "target_arch",
               aarch64::gemm::getGemmTargetName(candidate->target));
  appendString(plan, builder, "isa",
               aarch64::gemm::getGemmIsaName(candidate->isa));
  const aarch64::gemm::GemmKernelABI &abi =
      aarch64::gemm::getGemmKernelABI(candidate->target, candidate->isa,
                                      candidate->dataType,
                                      candidate->executionKind);
  appendString(plan, builder, "data_type",
               aarch64::gemm::getGemmDataTypeName(candidate->dataType));
  appendI64(plan, builder, "vector_length_bytes", abi.vectorLengthBytes);
  appendString(plan, builder, "kernel_family", abi.family);
  appendString(plan, builder, aarch64::gemm::kExecutionKindAttrName,
               aarch64::gemm::getGemmExecutionKindName(
                   candidate->executionKind));
  appendI64(plan, builder, "m", problem->m);
  appendI64(plan, builder, "n", problem->n);
  appendI64(plan, builder, "k", problem->k);
  appendI64(plan, builder, "lda", problem->lda);
  appendI64(plan, builder, "ldb", problem->ldb);
  appendI64(plan, builder, "ldc", problem->ldc);
  appendI64(plan, builder, "mc", candidate->cacheTile.mc);
  appendI64(plan, builder, "nc", candidate->cacheTile.nc);
  appendI64(plan, builder, "kc", candidate->cacheTile.kc);
  appendI64(plan, builder, "mr", candidate->kernelTile.mr);
  appendI64(plan, builder, "panel_lanes", candidate->kernelTile.panelLanes);
  appendString(plan, builder, "macro_order", "mkn");
  appendString(plan, builder, "micro_order", "mn");
  if (candidate->executionKind == aarch64::gemm::GemmExecutionKind::kGemvAB) {
    appendString(plan, builder, aarch64::gemm::kRhsPackingAttrName, "direct");
    appendString(plan, builder, aarch64::gemm::kRhsPackSourceAttrName,
                 "none");
  } else {
    if (candidate->isa == aarch64::gemm::GemmIsa::kSve) {
      appendString(plan, builder, "pack_b_schema", "annc-sve-packed-b-v2");
    } else {
      appendString(plan, builder, "pack_b_schema", "annc-neon-packed-b-v1");
    }
    appendString(plan, builder, "pack_b_block_order", "pc-jc");
    appendString(plan, builder, "pack_b_execution", "full-then-compute");
    // Skip RHS packing for small B matrices.  The environment override is
    // intentionally kept for benchmark/debug reproducibility.
    bool useRowMajor = false;
    const char *forcePacking = std::getenv("ANNC_GEMM_RHS_PACKING");
    if (forcePacking && std::strcmp(forcePacking, "packed") == 0) {
      useRowMajor = false;
    } else if (forcePacking && std::strcmp(forcePacking, "row_major") == 0) {
      useRowMajor = true;
    } else {
      useRowMajor = problem->m * problem->n * problem->k <=
                    kRowMajorOperationLimit;
    }
    appendString(plan, builder, aarch64::gemm::kRhsPackingAttrName,
                 useRowMajor ? "row_major" : "packed");
    appendString(plan, builder, aarch64::gemm::kRhsPackSourceAttrName,
                 useRowMajor ? "none" : "generated");
  }
  appendI64(plan, builder, "thread_count", candidate->threadCount);
  appendString(plan, builder, "thread_partition", "static-2d");
  appendString(plan, builder, "first_kc_mode", "overwrite");
  appendString(plan, builder, "next_kc_mode", "accumulate");
  if (auto epilogue =
          op->getAttrOfType<ArrayAttr>(aarch64::gemm::kEpilogueAttrName))
    plan.append("epilogue", epilogue);

  op->setDiscardableAttr(aarch64::gemm::kPlanAttrName,
                         plan.getDictionary(op->getContext()));
  op->removeDiscardableAttr(aarch64::gemm::kProblemAttrName);
  op->removeDiscardableAttr(aarch64::gemm::kCandidateAttrName);
  return success();
}

class AArch64FinalizeGemmPlan
    : public AArch64FinalizeGemmPlanBase<AArch64FinalizeGemmPlan> {
 public:
  using Base::Base;

  void runOnOperation() override {
    WalkResult result = getOperation().walk([](Operation *op) -> WalkResult {
      if (!aarch64::gemm::isGemmAnchor(op)) return WalkResult::advance();
      if (!op->hasAttr(aarch64::gemm::kProblemAttrName))
        return WalkResult::advance();
      return failed(finalizeGemmPlan(op)) ? WalkResult::interrupt()
                                          : WalkResult::advance();
    });
    if (result.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAArch64FinalizeGemmPlan() {
  return std::make_unique<AArch64FinalizeGemmPlan>();
}

}  // namespace annc
