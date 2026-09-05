#include "GemmPlan.h"
#include "Target/aarch64/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace annc {
namespace {

void appendI64(NamedAttrList &attrs, Builder &builder, llvm::StringRef name,
               int64_t value) {
  attrs.append(name, builder.getI64IntegerAttr(value));
}

void appendString(NamedAttrList &attrs, Builder &builder, llvm::StringRef name,
                  llvm::StringRef value) {
  attrs.append(name, builder.getStringAttr(value));
}

// Validate the prepacked RHS contract against the freshly selected candidate
// and copy its data binding into the finalized plan.
LogicalResult appendPrepackedPlanFields(
    Operation *op, Builder &builder, NamedAttrList &plan,
    const aarch64::gemm::GemmProblem &problem,
    const aarch64::gemm::GemmCandidate &candidate) {
  auto contract = op->getAttrOfType<DictionaryAttr>(
      aarch64::gemm::kPrepackedRhsAttrName);
  auto symbol =
      contract ? contract.getAs<StringAttr>("data_symbol") : StringAttr();
  auto elements =
      contract ? contract.getAs<IntegerAttr>("data_elements") : IntegerAttr();
  if (!contract || !symbol || symbol.getValue().empty() || !elements ||
      elements.getInt() <= 0)
    return op->emitOpError(
        "prepacked RHS requires a valid data_symbol/data_elements contract");

  const auto &abi = aarch64::gemm::getGemmKernelABI(
      candidate.target, candidate.isa, candidate.dataType,
      candidate.executionKind);
  // Recompute the packed-B size from the candidate geometry; the SVE panel
  // width rounds each N block up to the vector length.
  const int64_t panel =
      candidate.isa == aarch64::gemm::GemmIsa::kSve ? abi.vectorLengthBytes : 4;
  int64_t expectedElements = 0;
  for (const auto &block : aarch64::gemm::partitionCache2D(
           problem.k, problem.n, candidate.cacheTile.kc,
           candidate.cacheTile.nc))
    expectedElements +=
        block.kSize * ((block.nSize + panel - 1) / panel) * panel;

  // Every contract field that influences the packed-B layout must match the
  // candidate exactly; a drift means the prepacked data is unusable.  The
  // ISA/vector-length drift is caught by the recomputed data_elements.
  const std::pair<llvm::StringRef, int64_t> i64Fields[] = {
      {"version", aarch64::gemm::kPrepackedRhsVersion},
      {"k", problem.k},
      {"n", problem.n},
      {"nc", candidate.cacheTile.nc},
      {"kc", candidate.cacheTile.kc},
      {"data_elements", expectedElements}};
  for (const auto &field : i64Fields) {
    auto value = contract.getAs<IntegerAttr>(field.first);
    if (!value || value.getInt() != field.second) {
      op->emitOpError() << "prepacked RHS contract field " << field.first
                        << " must equal " << field.second;
      return failure();
    }
  }
  if (candidate.executionKind != aarch64::gemm::GemmExecutionKind::kGemm)
    return op->emitOpError("prepacked RHS requires the generic GEMM path");

  appendString(plan, builder, aarch64::gemm::kRhsPackingAttrName, "prepacked");
  appendString(plan, builder, aarch64::gemm::kRhsPackSourceAttrName,
               "prepacked");
  appendString(plan, builder, "pack_b_schema",
               aarch64::gemm::getGemmPackedBSchema(candidate.isa));
  appendString(plan, builder, "pack_b_block_order", "pc-jc");
  appendString(plan, builder, "rhs_data_symbol", symbol.getValue());
  appendI64(plan, builder, "rhs_data_elements", elements.getInt());
  return success();
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
  switch (candidate->rhsPacking) {
    case aarch64::gemm::RhsPacking::kPrepacked:
      if (failed(appendPrepackedPlanFields(op, builder, plan, *problem,
                                           *candidate)))
        return failure();
      break;
    case aarch64::gemm::RhsPacking::kDirect:
      appendString(plan, builder, aarch64::gemm::kRhsPackingAttrName, "direct");
      appendString(plan, builder, aarch64::gemm::kRhsPackSourceAttrName,
                   "none");
      break;
    case aarch64::gemm::RhsPacking::kPacked:
      appendString(plan, builder, "pack_b_schema",
                   aarch64::gemm::getGemmPackedBSchema(candidate->isa));
      appendString(plan, builder, "pack_b_block_order", "pc-jc");
      appendString(plan, builder, "pack_b_execution", "full-then-compute");
      appendString(plan, builder, aarch64::gemm::kRhsPackingAttrName,
                   "packed");
      appendString(plan, builder, aarch64::gemm::kRhsPackSourceAttrName,
                   "generated");
      break;
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
