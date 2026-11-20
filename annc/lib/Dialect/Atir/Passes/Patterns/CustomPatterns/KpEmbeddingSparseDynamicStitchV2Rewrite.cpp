#include <string>

#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/BuiltinKernelSchema.h"
#include "Dialect/Atir/CustomOpSchema.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Dialect/Atir/Passes/Patterns/CustomFusionPatternBase.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistry.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistryMacros.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace atir;

namespace {

// The pattern name the outline side writes into metadata (keeps the
// anchor off same-shape remops inside other patterns' sites).
constexpr llvm::StringRef kPatternName = "kp_sparse_dynamic_stitch";

bool isStitchKernelFunc(func::FuncOp func) {
  if (!func || !func->hasAttr("annc.kernel")) return false;
  auto metadata = func->getAttrOfType<DictionaryAttr>("fusion.metadata");
  if (!metadata) return false;
  auto pattern = dyn_cast_or_null<StringAttr>(metadata.get("fusion.pattern"));
  return pattern && pattern.getValue() == kPatternName;
}

LogicalResult matchStitchV2(func::FuncOp func, int64_t &n,
                            SmallVectorImpl<Operation *> &fusedOps) {
  n = func.getNumArguments() - 1;
  if (n < 2 || n > 16) return failure();
  for (Value arg : func.getArguments()) {
    auto ty = dyn_cast<RankedTensorType>(arg.getType());
    if (!ty) return failure();
  }
  Block &body = func.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(body.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1) return failure();
  auto outTy = dyn_cast<RankedTensorType>(returnOp.getOperand(0).getType());
  if (!outTy || !outTy.getElementType().isF32() || outTy.getRank() != 2)
    return failure();
  // variables f32 2-D, x integral 1-D (outline ABI order = func args).
  auto xTy = dyn_cast<RankedTensorType>(func.getArgument(0).getType());
  if (!xTy || xTy.getRank() != 1 || !isa<IntegerType>(xTy.getElementType()))
    return failure();
  for (int64_t k = 1; k <= n; ++k) {
    auto vTy =
        dyn_cast<RankedTensorType>(func.getArgument(k).getType());
    if (!vTy || !vTy.getElementType().isF32() || vTy.getRank() != 2)
      return failure();
  }
  for (Operation &op : body.getOperations()) {
    if (isa<func::ReturnOp>(op)) continue;
    fusedOps.push_back(&op);
  }
  return success();
}

struct KpEmbeddingSparseDynamicStitchV2Rewrite
    : public CustomFusionPatternBase<stablehlo::RemOp> {
  using Base = CustomFusionPatternBase<stablehlo::RemOp>;
  using Base::Base;

  mlir::LogicalResult matchFusion(
      stablehlo::RemOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!isStitchKernelFunc(func)) return mlir::failure();
    int64_t n = 0;
    return matchStitchV2(func, n, fusedOps);
  }

  std::string getCustomOpName(
      stablehlo::RemOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    int64_t n = func ? (func.getNumArguments() - 1) : 0;
    return "KPFusedSparseDynamicStitch";
  }

  CustomOpSchema getCustomOpSchema(
      stablehlo::RemOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    // Shared registry schema: x(1-D) + variable_i(2-D,T) -> output(2-D,T).
    auto func = anchor->template getParentOfType<func::FuncOp>();
    int64_t n = func ? (func.getNumArguments() - 1) : 0;
    return *getContextKernelSchema(getCustomOpName(anchor, fusedOps), {n});
  }

  void getOrderedBoundaryInputs(
      stablehlo::RemOp anchor, llvm::ArrayRef<mlir::Operation *> fusedOps,
      llvm::SmallVectorImpl<mlir::Value> &inputs) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!func) return;
    // Outline ABI order = (x, v_0..v_N-1) = the func argument order.
    for (Value arg : func.getArguments()) inputs.push_back(arg);
  }
};

REGISTER_CUSTOM_PATTERN(KpEmbeddingSparseDynamicStitchV2Rewrite);

}  // namespace
