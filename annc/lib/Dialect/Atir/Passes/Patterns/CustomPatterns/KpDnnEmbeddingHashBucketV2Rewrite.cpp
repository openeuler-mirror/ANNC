#include <string>

#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/BuiltinKernelSchema.h"
#include "Dialect/Atir/CustomOpSchema.h"
#include "Dialect/Atir/Passes/Patterns/CustomFusionPatternBase.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistry.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistryMacros.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace atir;

namespace {

constexpr llvm::StringRef kPatternName = "dnn_embedding_hash_bucket";

bool isHashBucketKernelFunc(func::FuncOp func) {
  if (!func || !func->hasAttr("annc.kernel")) return false;
  auto metadata = func->getAttrOfType<DictionaryAttr>("fusion.metadata");
  if (!metadata) return false;
  auto pattern = dyn_cast_or_null<StringAttr>(metadata.get("fusion.pattern"));
  return pattern && pattern.getValue() == kPatternName;
}

// Validates the ABI argument slots by position: input rank1 with string
// elements, embedding_weight rank2 f32.
bool classifyHashBucketArgs(func::FuncOp func) {
  if (func.getNumArguments() != 2) return false;
  auto inTy = dyn_cast<RankedTensorType>(func.getArgument(0).getType());
  if (!inTy || inTy.getRank() != 1 ||
      !isa<atir::StringType>(inTy.getElementType()))
    return false;
  auto wTy = dyn_cast<RankedTensorType>(func.getArgument(1).getType());
  return wTy && wTy.getRank() == 2 && wTy.getElementType().isF32();
}

// Matches the op-fusion-v2 outlined kernel: value-returned func tagged with
// the pattern name; num_buckets rides the anchor attr into the schema.
LogicalResult matchHashBucketV2(atir::StringToHashBucketFastV2Op anchor,
                                func::FuncOp func,
                                SmallVectorImpl<Operation *> &fusedOps) {
  if (!isHashBucketKernelFunc(func)) return failure();
  Block &body = func.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(body.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1) return failure();
  auto outTy = dyn_cast<RankedTensorType>(returnOp.getOperand(0).getType());
  if (!outTy || outTy.getRank() != 2 || !outTy.getElementType().isF32())
    return failure();
  if (!classifyHashBucketArgs(func)) return failure();
  if (anchor.getNumBuckets() == 0) return failure();

  for (Operation &op : body.getOperations()) {
    if (isa<func::ReturnOp>(op)) continue;
    fusedOps.push_back(&op);
  }
  return fusedOps.empty() ? failure() : success();
}

// ABI order = func argument order (fixed to the schema order upstream).
void orderedBoundaryInputs(func::FuncOp func, SmallVectorImpl<Value> &inputs) {
  for (Value arg : func.getArguments()) inputs.push_back(arg);
}

struct KpDnnEmbeddingHashBucketV2Rewrite
    : public CustomFusionPatternBase<atir::StringToHashBucketFastV2Op> {
  using Base = CustomFusionPatternBase<atir::StringToHashBucketFastV2Op>;
  using Base::Base;

  mlir::LogicalResult matchFusion(
      atir::StringToHashBucketFastV2Op anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!func) return mlir::failure();
    return matchHashBucketV2(anchor, func, fusedOps);
  }

  std::string getCustomOpName(
      atir::StringToHashBucketFastV2Op,
      llvm::ArrayRef<mlir::Operation *>) const override {
    return "KPFusedDnnEmbeddingWithHashBucket";
  }

  // Same registry schema as V1; num_buckets comes from the anchor attr.
  CustomOpSchema getCustomOpSchema(
      atir::StringToHashBucketFastV2Op anchor,
      llvm::ArrayRef<mlir::Operation *>) const override {
    return *getContextKernelSchema(
        "KPFusedDnnEmbeddingWithHashBucket",
        static_cast<int64_t>(anchor.getNumBuckets()));
  }

  void getOrderedBoundaryInputs(
      atir::StringToHashBucketFastV2Op anchor,
      llvm::ArrayRef<mlir::Operation *>,
      llvm::SmallVectorImpl<mlir::Value> &inputs) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!func) return;
    orderedBoundaryInputs(func, inputs);
  }
};

REGISTER_CUSTOM_PATTERN(KpDnnEmbeddingHashBucketV2Rewrite);

}  // namespace
