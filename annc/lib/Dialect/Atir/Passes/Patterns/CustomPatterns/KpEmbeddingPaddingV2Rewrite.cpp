#include "Dialect/Atir/BuiltinKernelSchema.h"
#include "Dialect/Atir/CustomOpSchema.h"
#include "Dialect/Atir/Passes/Patterns/CustomFusionPatternBase.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistry.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistryMacros.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace atir;

namespace {

constexpr llvm::StringRef kPatternName = "kp_embedding_padding";
constexpr llvm::StringRef kFastPatternName = "kp_embedding_padding_fast";

bool isPaddingKernelFunc(func::FuncOp func, llvm::StringRef patternName) {
  if (!func || !func->hasAttr("annc.kernel")) return false;
  auto metadata = func->getAttrOfType<DictionaryAttr>("fusion.metadata");
  if (!metadata) return false;
  auto pattern = dyn_cast_or_null<StringAttr>(metadata.get("fusion.pattern"));
  return pattern && pattern.getValue() == patternName;
}

// ABI arg slots by semantics; input_rows and pack share the rank0 i32
// signature and split by position - the outline side fixes rows before
// pack (takeValue in EmbeddingPaddingV2.cpp / the pick loop in FastV2).
struct EmbeddingPaddingArgs {
  Value originShape, inputRows, pack, data, reshapeSizes;
};

bool classifyEmbeddingPaddingArgs(func::FuncOp func,
                                  EmbeddingPaddingArgs &args) {
  if (func.getNumArguments() != 5) return false;
  for (Value arg : func.getArguments()) {
    auto ty = dyn_cast<RankedTensorType>(arg.getType());
    if (!ty) return false;
    if (ty.getRank() == 1 && ty.getElementType().isInteger(64)) {
      if (args.originShape) return false;
      args.originShape = arg;
    } else if (ty.getRank() == 0 && ty.getElementType().isInteger(32)) {
      if (!args.inputRows) {
        args.inputRows = arg;
      } else if (!args.pack) {
        args.pack = arg;
      } else {
        return false;
      }
    } else if (ty.getRank() == 2 && ty.getElementType().isF32()) {
      if (args.data) return false;
      args.data = arg;
    } else if (ty.getRank() == 1 && ty.getElementType().isInteger(32)) {
      if (args.reshapeSizes) return false;
      args.reshapeSizes = arg;
    } else {
      return false;
    }
  }
  return args.originShape && args.inputRows && args.pack && args.data &&
         args.reshapeSizes;
}

void orderedPaddingInputs(func::FuncOp func,
                          llvm::SmallVectorImpl<mlir::Value> &inputs) {
  EmbeddingPaddingArgs args;
  if (!classifyEmbeddingPaddingArgs(func, args)) return;
  for (Value v : {args.originShape, args.inputRows, args.pack, args.data,
                  args.reshapeSizes})
    inputs.push_back(v);
}

struct KpEmbeddingPaddingV2Rewrite
    : public CustomFusionPatternBase<stablehlo::ConcatenateOp> {
  using Base = CustomFusionPatternBase<stablehlo::ConcatenateOp>;
  using Base::Base;

  // Values-returned contract: return (padding scalar i32, padded matrix
  // f32 rank2), the matrix wired straight to this anchor.
  mlir::LogicalResult matchFusion(
      stablehlo::ConcatenateOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    auto func = anchor->getParentOfType<func::FuncOp>();
    if (!isPaddingKernelFunc(func, kPatternName)) return mlir::failure();
    auto returnOp =
        dyn_cast<func::ReturnOp>(func.getBody().front().getTerminator());
    if (!returnOp || returnOp.getNumOperands() != 2) return mlir::failure();
    if (returnOp.getOperand(1) != anchor.getResult()) return mlir::failure();
    auto padTy = dyn_cast<RankedTensorType>(returnOp.getOperand(0).getType());
    if (!padTy || padTy.getRank() != 0 || !padTy.getElementType().isInteger(32))
      return mlir::failure();
    EmbeddingPaddingArgs args;
    if (!classifyEmbeddingPaddingArgs(func, args)) return mlir::failure();

    for (Operation &op : func.getBody().front().getOperations()) {
      if (isa<func::ReturnOp>(op)) continue;
      fusedOps.push_back(&op);
    }
    return fusedOps.empty() ? mlir::failure() : mlir::success();
  }

  std::string getCustomOpName(
      stablehlo::ConcatenateOp,
      llvm::ArrayRef<mlir::Operation *>) const override {
    return "KPFusedEmbeddingPadding";
  }

  CustomOpSchema getCustomOpSchema(
      stablehlo::ConcatenateOp,
      llvm::ArrayRef<mlir::Operation *>) const override {
    return *getContextKernelSchema("KPFusedEmbeddingPadding");
  }

  void getOrderedBoundaryInputs(
      stablehlo::ConcatenateOp anchorOp, llvm::ArrayRef<mlir::Operation *>,
      llvm::SmallVectorImpl<mlir::Value> &inputs) const override {
    auto func = anchorOp->getParentOfType<func::FuncOp>();
    if (func) orderedPaddingInputs(func, inputs);
  }
};

REGISTER_CUSTOM_PATTERN(KpEmbeddingPaddingV2Rewrite);

// Fast-variant kernel check: return (padding_rows, reshape_rows) with the
// reshape rows wired straight to the anchor result.
LogicalResult matchPaddingFastKernel(func::FuncOp func, Value rowsValue,
                                     SmallVectorImpl<Operation *> &fusedOps) {
  if (!isPaddingKernelFunc(func, kFastPatternName)) return failure();
  Block &body = func.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(body.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 2) return failure();
  if (returnOp.getOperand(1) != rowsValue) return failure();
  auto padTy = dyn_cast<RankedTensorType>(returnOp.getOperand(0).getType());
  if (!padTy || padTy.getRank() != 0 || !padTy.getElementType().isInteger(32))
    return failure();
  EmbeddingPaddingArgs args;
  if (!classifyEmbeddingPaddingArgs(func, args)) return failure();

  for (Operation &op : body.getOperations()) {
    if (isa<func::ReturnOp>(op)) continue;
    fusedOps.push_back(&op);
  }
  return fusedOps.empty() ? failure() : success();
}

// Scalar-reshape anchor (the import-folded form).
struct KpEmbeddingPaddingFastScalarV2Rewrite
    : public CustomFusionPatternBase<stablehlo::ReshapeOp> {
  using Base = CustomFusionPatternBase<stablehlo::ReshapeOp>;
  using Base::Base;

  mlir::LogicalResult matchFusion(
      stablehlo::ReshapeOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    auto ty = dyn_cast<RankedTensorType>(anchor.getType());
    if (!ty || ty.getRank() != 0 || !ty.getElementType().isInteger(32))
      return mlir::failure();
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!func) return mlir::failure();
    return matchPaddingFastKernel(func, anchor.getResult(), fusedOps);
  }

  std::string getCustomOpName(
      stablehlo::ReshapeOp,
      llvm::ArrayRef<mlir::Operation *>) const override {
    return "KPFusedEmbeddingPaddingFast";
  }

  CustomOpSchema getCustomOpSchema(
      stablehlo::ReshapeOp, llvm::ArrayRef<mlir::Operation *>) const override {
    return *getContextKernelSchema("KPFusedEmbeddingPaddingFast");
  }

  void getOrderedBoundaryInputs(
      stablehlo::ReshapeOp anchor, llvm::ArrayRef<mlir::Operation *>,
      llvm::SmallVectorImpl<mlir::Value> &inputs) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (func) orderedPaddingInputs(func, inputs);
  }
};

// Unfolded StridedSlice residue anchor.
struct KpEmbeddingPaddingFastSliceV2Rewrite
    : public CustomFusionPatternBase<atir::UnsupportedOp> {
  using Base = CustomFusionPatternBase<atir::UnsupportedOp>;
  using Base::Base;

  mlir::LogicalResult matchFusion(
      atir::UnsupportedOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    if (anchor.getOpType() != "StridedSlice") return mlir::failure();
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!func) return mlir::failure();
    return matchPaddingFastKernel(func, anchor.getResult(0), fusedOps);
  }

  std::string getCustomOpName(
      atir::UnsupportedOp,
      llvm::ArrayRef<mlir::Operation *>) const override {
    return "KPFusedEmbeddingPaddingFast";
  }

  CustomOpSchema getCustomOpSchema(
      atir::UnsupportedOp,
      llvm::ArrayRef<mlir::Operation *>) const override {
    return *getContextKernelSchema("KPFusedEmbeddingPaddingFast");
  }

  void getOrderedBoundaryInputs(
      atir::UnsupportedOp anchor, llvm::ArrayRef<mlir::Operation *>,
      llvm::SmallVectorImpl<mlir::Value> &inputs) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (func) orderedPaddingInputs(func, inputs);
  }
};

REGISTER_CUSTOM_PATTERN(KpEmbeddingPaddingFastScalarV2Rewrite);
REGISTER_CUSTOM_PATTERN(KpEmbeddingPaddingFastSliceV2Rewrite);

}  // namespace
