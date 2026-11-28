#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/BuiltinKernelSchema.h"
#include "Dialect/Atir/CustomOpSchema.h"
#include "Dialect/Atir/Passes/FusionV2/CarrierMatchers.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Dialect/Atir/Passes/Patterns/CustomFusionPatternBase.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistry.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistryMacros.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace atir;

namespace {

bool classifyTbiArgs(func::FuncOp func) {
  if (func.getNumArguments() != 4) return false;
  static const unsigned kRanks[4] = {3, 2, 1, 3};
  for (unsigned i = 0; i < 4; ++i) {
    auto ty = dyn_cast<RankedTensorType>(func.getArgument(i).getType());
    if (!ty || !ty.getElementType().isF32() || ty.getRank() != kRanks[i])
      return false;
  }
  return true;
}

LogicalResult matchTbiV2(mlir::Operation *anchorOp, func::FuncOp func,
                         SmallVectorImpl<Operation *> &fusedOps) {
  fusionv2::ConcatLike concat = fusionv2::ConcatLike::match(anchorOp);
  if (!concat) return failure();
  auto dim = concat.dimension();
  if (!dim || *dim != 2) return failure();
  Block &body = func.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(body.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1) return failure();
  if (returnOp.getOperand(0) != concat.result()) return failure();
  if (!classifyTbiArgs(func)) return failure();

  for (Operation &op : body.getOperations()) {
    if (isa<func::ReturnOp>(op)) continue;
    fusedOps.push_back(&op);
  }
  return success();
}

void orderedBoundaryInputs(func::FuncOp func, SmallVectorImpl<Value> &inputs) {
  for (Value arg : func.getArguments()) inputs.push_back(arg);
}

template <typename AnchorOp>
struct KpTargetBehaviorInteractionV2RewriteImpl
    : public CustomFusionPatternBase<AnchorOp> {
  using Base = CustomFusionPatternBase<AnchorOp>;
  using Base::Base;

  mlir::LogicalResult matchFusion(
      AnchorOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    if (!atir::fusionv2::ConcatLike::match(anchor.getOperation()))
      return mlir::failure();
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!func || !func->hasAttr("annc.kernel")) return mlir::failure();
    return matchTbiV2(anchor.getOperation(), func, fusedOps);
  }

  std::string getCustomOpName(
      AnchorOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    return "KPFusedTargetBehaviorInteraction";
  }

  CustomOpSchema getCustomOpSchema(
      AnchorOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    auto schema = getContextKernelSchema(getCustomOpName(anchor, fusedOps));
    assert(schema && "unregistered kernel name; match should have rejected");
    return *schema;
  }

  void getOrderedBoundaryInputs(
      AnchorOp anchor, llvm::ArrayRef<mlir::Operation *> fusedOps,
      llvm::SmallVectorImpl<mlir::Value> &inputs) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!func) return;
    orderedBoundaryInputs(func, inputs);
  }

  void augmentFusedSet(
      AnchorOp anchor, llvm::ArrayRef<mlir::Operation *> fusedOps,
      llvm::SmallDenseSet<mlir::Operation *> &fusedSet) const override {
    for (Operation *op : fusedOps)
      if (auto assuming = dyn_cast<shape::AssumingOp>(op))
        assuming->walk([&](Operation *inner) { fusedSet.insert(inner); });
  }
};

using KpTargetBehaviorInteractionV2ConcatenateRewrite =
    KpTargetBehaviorInteractionV2RewriteImpl<stablehlo::ConcatenateOp>;
using KpTargetBehaviorInteractionV2ResidueRewrite =
    KpTargetBehaviorInteractionV2RewriteImpl<atir::UnsupportedOp>;

REGISTER_CUSTOM_PATTERN(KpTargetBehaviorInteractionV2ConcatenateRewrite);
REGISTER_CUSTOM_PATTERN(KpTargetBehaviorInteractionV2ResidueRewrite);

}  // namespace
