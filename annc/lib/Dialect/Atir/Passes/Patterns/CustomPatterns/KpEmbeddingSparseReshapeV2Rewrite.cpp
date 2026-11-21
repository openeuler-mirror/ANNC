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

using namespace mlir;
using namespace atir;

namespace {

// new_shape）。
struct SparseReshapeArgs {
  Value sliceInput, begin, packConst, newShape;
};

bool classifySparseReshapeArgs(func::FuncOp func, SparseReshapeArgs &args) {
  if (func.getNumArguments() != 4) return false;
  for (Value arg : func.getArguments()) {
    auto ty = dyn_cast<RankedTensorType>(arg.getType());
    if (!ty) return false;
    if (ty.getRank() == 2 && ty.getElementType().isInteger(64)) {
      if (args.sliceInput) return false;
      args.sliceInput = arg;
    } else if (ty.getRank() == 1 && ty.getElementType().isInteger(32) &&
               ty.getShape()[0] == 2) {
      if (args.begin) return false;
      args.begin = arg;
    } else if (ty.getRank() == 0 &&
               (ty.getElementType().isInteger(32) ||
                ty.getElementType().isInteger(64))) {
      if (args.packConst) return false;
      args.packConst = arg;
    } else if (ty.getRank() == 1 && ty.getElementType().isInteger(64) &&
               ty.getShape()[0] == 2) {
      if (args.newShape) return false;
      args.newShape = arg;
    } else {
      return false;
    }
  }
  return args.sliceInput && args.begin && args.packConst && args.newShape;
}

LogicalResult matchSparseReshapeV2(atir::SparseReshapeV2Op anchor,
                                   func::FuncOp func,
                                   SmallVectorImpl<Operation *> &fusedOps) {
  // The registry schema only defines the I64 builtin; an i32 pack_const
  // would name an unregistered kernel, so reject it up front.
  SparseReshapeArgs pre;
  if (!classifySparseReshapeArgs(func, pre)) return failure();
  auto packTy = dyn_cast<RankedTensorType>(pre.packConst.getType());
  if (!packTy || !packTy.getElementType().isInteger(64)) return failure();
  Block &body = func.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(body.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 2) return failure();
  if (returnOp.getOperand(0) != anchor.getOutputIndices() ||
      returnOp.getOperand(1) != anchor.getOutputShape())
    return failure();

  SparseReshapeArgs args;
  if (!classifySparseReshapeArgs(func, args)) return failure();

  for (Operation &op : body.getOperations()) {
    if (isa<func::ReturnOp>(op)) continue;
    fusedOps.push_back(&op);
  }
  return success();
}

std::string sparseReshapeV2OpName(Type packConstType) {
  auto tensorType = dyn_cast<RankedTensorType>(packConstType);
  std::string name = "KPFusedSparseReshape";
  name +=
      (tensorType && tensorType.getElementType().isInteger(64)) ? "I64" : "I32";
  return name;
}

void orderedBoundaryInputs(func::FuncOp func, SmallVectorImpl<Value> &inputs) {
  SparseReshapeArgs args;
  if (!classifySparseReshapeArgs(func, args)) return;
  for (Value v :
       {args.sliceInput, args.begin, args.packConst, args.newShape})
    inputs.push_back(v);
}

struct KpEmbeddingSparseReshapeV2Rewrite
    : public CustomFusionPatternBase<atir::SparseReshapeV2Op> {
  using Base = CustomFusionPatternBase<atir::SparseReshapeV2Op>;
  using Base::Base;

  mlir::LogicalResult matchFusion(
      atir::SparseReshapeV2Op anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!func || !func->hasAttr("annc.kernel")) return mlir::failure();
    return matchSparseReshapeV2(anchor, func, fusedOps);
  }

  std::string getCustomOpName(
      atir::SparseReshapeV2Op anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    SparseReshapeArgs args;
    Type packConstType;
    if (func && classifySparseReshapeArgs(func, args))
      packConstType = args.packConst.getType();
    return sparseReshapeV2OpName(packConstType);
  }

  CustomOpSchema getCustomOpSchema(
      atir::SparseReshapeV2Op anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    auto schema = getContextKernelSchema(getCustomOpName(anchor, fusedOps));
    assert(schema && "unregistered kernel name; match should have rejected");
    return *schema;
  }

  void getOrderedBoundaryInputs(
      atir::SparseReshapeV2Op anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps,
      llvm::SmallVectorImpl<mlir::Value> &inputs) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!func) return;
    orderedBoundaryInputs(func, inputs);
  }
};

REGISTER_CUSTOM_PATTERN(KpEmbeddingSparseReshapeV2Rewrite);

}  // namespace
