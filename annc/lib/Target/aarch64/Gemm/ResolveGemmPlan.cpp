#include "GemmPlan.h"
#include "Target/aarch64/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace annc {
namespace {

bool matchesProjectedMap(AffineMap map,
                         llvm::ArrayRef<unsigned> dimensions) {
  if (map.getNumDims() != 3 || map.getNumSymbols() != 0 ||
      map.getNumResults() != dimensions.size())
    return false;
  for (auto [expression, dimension] :
       llvm::zip_equal(map.getResults(), dimensions)) {
    auto dim = llvm::dyn_cast<AffineDimExpr>(expression);
    if (!dim || dim.getPosition() != dimension)
      return false;
  }
  return true;
}

LogicalResult validateGemmGeneric(linalg::GenericOp generic) {
  if (generic->getNumResults() != 0)
    return generic.emitOpError(
        "AArch64 GEMM requires buffer-semantics linalg.generic");
  if (generic.getInputs().size() < 2 || generic.getOutputs().size() != 1)
    return generic.emitOpError(
        "annc.gemm requires A/B inputs and exactly one C output");

  auto iteratorTypes = generic.getIteratorTypesArray();
  if (iteratorTypes.size() != 3 ||
      iteratorTypes[0] != utils::IteratorType::parallel ||
      iteratorTypes[1] != utils::IteratorType::parallel ||
      iteratorTypes[2] != utils::IteratorType::reduction)
    return generic.emitOpError(
        "annc.gemm requires M/N/K iterator types "
        "parallel/parallel/reduction");

  auto maps = generic.getIndexingMapsArray();
  if (maps.size() != generic.getInputs().size() + generic.getOutputs().size() ||
      !matchesProjectedMap(maps[0], {0, 2}) ||
      !matchesProjectedMap(maps[1], {2, 1}) ||
      !matchesProjectedMap(maps.back(), {0, 1}))
    return generic.emitOpError(
        "annc.gemm requires A=(M,K), B=(K,N), and C=(M,N) indexing maps");
  return success();
}

LogicalResult validateMemRef(Operation *op, MemRefType type,
                             llvm::StringRef operandName) {
  if (type.getRank() != 2 || !type.hasStaticShape())
    return op->emitOpError()
           << "AArch64 GEMM requires static rank-2 " << operandName;
  if (!type.getElementType().isF32())
    return op->emitOpError() << "AArch64 GEMM requires f32 " << operandName;
  if (!type.getLayout().isIdentity())
    return op->emitOpError()
           << "AArch64 GEMM requires row-major identity-layout "
           << operandName;
  return success();
}

FailureOr<aarch64::gemm::GemmProblem> getGemmProblem(Operation *op) {
  auto lhsType =
      llvm::dyn_cast<MemRefType>(aarch64::gemm::getGemmInput(op, 0).getType());
  auto rhsType =
      llvm::dyn_cast<MemRefType>(aarch64::gemm::getGemmInput(op, 1).getType());
  auto outType =
      llvm::dyn_cast<MemRefType>(aarch64::gemm::getGemmOutput(op, 0).getType());
  if (!lhsType || !rhsType || !outType) {
    op->emitOpError(
        "aarch64-resolve-gemm-plan requires bufferized memref operands");
    return failure();
  }
  if (failed(validateMemRef(op, lhsType, "lhs")) ||
      failed(validateMemRef(op, rhsType, "rhs")) ||
      failed(validateMemRef(op, outType, "output"))) {
    return failure();
  }

  int64_t m = lhsType.getShape()[0];
  int64_t k = lhsType.getShape()[1];
  int64_t n = rhsType.getShape()[1];
  if (m <= 0 || n <= 0 || k <= 0) {
    op->emitOpError("AArch64 GEMM requires positive M/N/K");
    return failure();
  }
  if (rhsType.getShape()[0] != k || outType.getShape()[0] != m ||
      outType.getShape()[1] != n) {
    op->emitOpError("inconsistent M/N/K dimensions");
    return failure();
  }

  return aarch64::gemm::GemmProblem{
      aarch64::gemm::kPlanVersion, m, n, k, k, n, n};
}

void appendI64(NamedAttrList &attrs, Builder &builder, llvm::StringRef name,
               int64_t value) {
  attrs.append(name, builder.getI64IntegerAttr(value));
}

LogicalResult resolveGemmProblem(Operation *op) {
  if (op->hasAttr(aarch64::gemm::kProblemAttrName) ||
      op->hasAttr(aarch64::gemm::kCandidateAttrName) ||
      op->hasAttr(aarch64::gemm::kPlanAttrName)) {
    return op->emitOpError("already has AArch64 GEMM planning state");
  }

  if (auto generic = llvm::dyn_cast<linalg::GenericOp>(op);
      generic && failed(validateGemmGeneric(generic)))
    return failure();

  FailureOr<aarch64::gemm::GemmProblem> problem = getGemmProblem(op);
  if (failed(problem)) return failure();

  if (auto attr = op->getAttrOfType<BoolAttr>("left_transpose");
      attr && attr.getValue())
    return op->emitOpError("AArch64 GEMM does not support transposed lhs");
  if (auto attr = op->getAttrOfType<BoolAttr>("right_transpose");
      attr && attr.getValue())
    return op->emitOpError("AArch64 GEMM does not support transposed rhs");
  if (auto attr = op->getAttrOfType<BoolAttr>("output_transpose");
      attr && attr.getValue())
    return op->emitOpError(
        "AArch64 GEMM does not support transposed output");

  Builder builder(op->getContext());
  NamedAttrList problemAttrs;
  appendI64(problemAttrs, builder, "version", problem->version);
  appendI64(problemAttrs, builder, "m", problem->m);
  appendI64(problemAttrs, builder, "n", problem->n);
  appendI64(problemAttrs, builder, "k", problem->k);
  appendI64(problemAttrs, builder, "lda", problem->lda);
  appendI64(problemAttrs, builder, "ldb", problem->ldb);
  appendI64(problemAttrs, builder, "ldc", problem->ldc);
  op->setDiscardableAttr(aarch64::gemm::kProblemAttrName,
                         problemAttrs.getDictionary(op->getContext()));
  return success();
}

class AArch64ResolveGemmPlan
    : public AArch64ResolveGemmPlanBase<AArch64ResolveGemmPlan> {
 public:
  using Base::Base;

  void runOnOperation() override {
    WalkResult result = getOperation().walk([](Operation *op) -> WalkResult {
      if (!aarch64::gemm::isGemmAnchor(op)) return WalkResult::advance();
      return failed(resolveGemmProblem(op)) ? WalkResult::interrupt()
                                            : WalkResult::advance();
    });
    if (result.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAArch64ResolveGemmPlan() {
  return std::make_unique<AArch64ResolveGemmPlan>();
}

}  // namespace annc
