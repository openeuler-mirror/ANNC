#include <string>

#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/BuiltinKernelSchema.h"
#include "Dialect/Atir/CustomOpSchema.h"
#include "Dialect/Atir/Passes/FusionV2/CarrierMatchers.h"
#include "Dialect/Atir/Passes/Patterns/CustomFusionPatternBase.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistry.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistryMacros.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace atir;

namespace {

// Validates the ABI argument slots by position (params rank2 f32, indices
// rank 1..2 integer, pack scalars i32); returns the R{r1}x{r2} suffix.
std::string classifyActionIdGatherArgs(func::FuncOp func) {
  if (func.getNumArguments() != 5) return "";
  auto paramsTy = dyn_cast<RankedTensorType>(func.getArgument(0).getType());
  if (!paramsTy || !paramsTy.getElementType().isF32() ||
      paramsTy.getRank() != 2)
    return "";
  auto packDimTy = dyn_cast<RankedTensorType>(func.getArgument(3).getType());
  auto packTy = dyn_cast<RankedTensorType>(func.getArgument(4).getType());
  if (!packDimTy || packDimTy.getRank() != 0 ||
      !packDimTy.getElementType().isInteger(32))
    return "";
  if (!packTy || packTy.getRank() != 0 ||
      !packTy.getElementType().isInteger(32))
    return "";
  unsigned ranks[2] = {0, 0};
  for (unsigned i = 0; i < 2; ++i) {
    auto ty = dyn_cast<RankedTensorType>(func.getArgument(1 + i).getType());
    if (!ty || !ty.getElementType().isInteger() || ty.getRank() < 1 ||
        ty.getRank() > 2)
      return "";
    ranks[i] = ty.getRank();
  }
  return "R" + std::to_string(ranks[0]) + "x" + std::to_string(ranks[1]);
}

// Matches the op-fusion-v2 outlined kernel: value-returned func whose
// return feeds directly from the tail concat (both carriers normalized
// by ConcatLike); the closure layout is guaranteed by the outline side.
LogicalResult matchActionIdGatherV2(mlir::Operation *anchorOp,
                                    func::FuncOp func,
                                    SmallVectorImpl<Operation *> &fusedOps) {
  fusionv2::ConcatLike concat = fusionv2::ConcatLike::match(anchorOp);
  if (!concat) return failure();
  auto dim = concat.dimension();
  if (!dim || *dim != 1) return failure();
  Block &body = func.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(body.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1) return failure();
  if (returnOp.getOperand(0) != concat.result()) return failure();

  if (classifyActionIdGatherArgs(func).empty()) return failure();

  for (Operation &op : body.getOperations()) {
    if (isa<func::ReturnOp>(op)) continue;
    fusedOps.push_back(&op);
  }
  return success();
}

// ABI order = func argument order (fixed to the schema order upstream).
void orderedBoundaryInputs(func::FuncOp func, SmallVectorImpl<Value> &inputs) {
  for (Value arg : func.getArguments()) inputs.push_back(arg);
}

template <typename AnchorOp>
struct KpEmbeddingActionIdGatherV2RewriteImpl
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
    return matchActionIdGatherV2(anchor.getOperation(), func, fusedOps);
  }

  std::string getCustomOpName(
      AnchorOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    std::string suffix =
        func ? classifyActionIdGatherArgs(func) : std::string();
    return "KPFusedEmbeddingActionIdGather" +
           (suffix.empty() ? std::string("R2x2") : suffix);
  }

  // Same registry schema as V1 (params(2) + indices(1..2) + pack scalars
  // -> output(2), T/T1/T2 bound by dtype).
  CustomOpSchema getCustomOpSchema(
      AnchorOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    return *getContextKernelSchema(getCustomOpName(anchor, fusedOps));
  }

  void getOrderedBoundaryInputs(
      AnchorOp anchor, llvm::ArrayRef<mlir::Operation *> fusedOps,
      llvm::SmallVectorImpl<mlir::Value> &inputs) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!func) return;
    orderedBoundaryInputs(func, inputs);
  }
};

using KpEmbeddingActionIdGatherV2ConcatenateRewrite =
    KpEmbeddingActionIdGatherV2RewriteImpl<stablehlo::ConcatenateOp>;
using KpEmbeddingActionIdGatherV2ResidueRewrite =
    KpEmbeddingActionIdGatherV2RewriteImpl<atir::UnsupportedOp>;

REGISTER_CUSTOM_PATTERN(KpEmbeddingActionIdGatherV2ConcatenateRewrite);
REGISTER_CUSTOM_PATTERN(KpEmbeddingActionIdGatherV2ResidueRewrite);

}  // namespace
