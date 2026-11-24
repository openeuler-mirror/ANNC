#include "Dialect/Atir/AtirOps.h"
#include "Support/Log.h"
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

// func 参数按 ABI 语义识别（与 outline 侧同判据；不依赖参数序——ABI 序由
// getOrderedBoundaryInputs 固化为 schema 序 keys/begin/data/indices/begin_1）。
struct SegmentReduceArgs {
  Value keys, begin, begin_1, data, indices;
};

// The outline ABI fixes the func argument order to the schema sequence
// [keys, begin, data, indices, begin_1], so slots are taken by position
// and types are asserted afterwards. begin may be a static [2] constant or
// the Form-A' runtime-broadcast dynamic width; type-only classification
// would mis-take a dynamic begin for i32 indices.
bool classifySegmentReduceArgs(func::FuncOp func, SegmentReduceArgs &args) {
  if (func.getNumArguments() != 5) return false;
  args.keys = func.getArgument(0);
  args.begin = func.getArgument(1);
  args.data = func.getArgument(2);
  args.indices = func.getArgument(3);
  args.begin_1 = func.getArgument(4);
  auto tyOf = [](Value v) {
    return dyn_cast<RankedTensorType>(v.getType());
  };
  auto keysTy = tyOf(args.keys);
  auto beginTy = tyOf(args.begin);
  auto dataTy = tyOf(args.data);
  auto indicesTy = tyOf(args.indices);
  auto begin1Ty = tyOf(args.begin_1);
  return keysTy && keysTy.getRank() == 2 &&
         keysTy.getElementType().isInteger(64) && beginTy &&
         beginTy.getRank() == 1 && beginTy.getElementType().isInteger(32) &&
         dataTy && dataTy.getRank() == 2 && dataTy.getElementType().isF32() &&
         indicesTy && indicesTy.getRank() == 1 &&
         (indicesTy.getElementType().isInteger(32) ||
          indicesTy.getElementType().isInteger(64)) &&
         begin1Ty && begin1Ty.getRank() == 1 &&
         begin1Ty.getElementType().isInteger(32);
}

// Shared match: the anchor is sparse_segment_{mean,sum}_v2 (combiner
// passed by the caller).
LogicalResult matchSegmentReduceV2(Operation *ssOp, int64_t combiner,
                                   func::FuncOp func, Type &indicesType,
                                   SmallVectorImpl<Operation *> &fusedOps) {
  auto why = [&](const char *r) {
    ANNC_LOG_DEBUG("fusion-pattern")
        << "seg-reduce-v2 reject(" << r << ") "
        << func.getName().str() << "\n";
    return failure();
  };
  // Value-return contract: return carries dual values (ss result and the
  // i32 slice_output scalar); outputs are host-allocated.
  Block &body = func.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(body.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 2) return why("return-arity");
  if (returnOp.getOperand(0) != ssOp->getResult(0)) return why("return-port0");
  auto sliceTy = dyn_cast<RankedTensorType>(returnOp.getOperand(1).getType());
  if (!sliceTy || !sliceTy.getElementType().isInteger(32) ||
      sliceTy.getRank() != 0)
    return why("return-port1");

  SegmentReduceArgs args;
  if (!classifySegmentReduceArgs(func, args)) return why("arg-classify");
  indicesType = args.indices.getType();

  // num_segments constant 0 (the kernel infers batch from segment_ids).
  Value numSegments = combiner == 1
                          ? cast<atir::SparseSegmentMeanV2Op>(ssOp).getNumSegments()
                          : cast<atir::SparseSegmentSumV2Op>(ssOp).getNumSegments();
  Operation *numDef = numSegments.getDefiningOp();
  auto numSplat =
      numDef ? numDef->getAttrOfType<DenseElementsAttr>("value") : nullptr;
  if (!numSplat || !numSplat.isSplat() ||
      !numSplat.getSplatValue<APInt>().isZero())
    return why("num-segments");

  // fusedOps order drives the escaping-result order, which must match the
  // kernel return (output, slice_output): the odml body lists the count /
  // slice chain before the reduce, so anchor the reduce first explicitly.
  fusedOps.push_back(ssOp);
  for (Operation &op : body.getOperations()) {
    if (isa<func::ReturnOp>(op) || &op == ssOp) continue;
    fusedOps.push_back(&op);
  }
  return success();
}

std::string segmentReduceOpName(int64_t combiner, Type indicesType) {
  auto tensorType = dyn_cast<RankedTensorType>(indicesType);
  std::string name = "KPFusedSparseSegmentReduce";
  name +=
      (tensorType && tensorType.getElementType().isInteger(64)) ? "I64" : "I32";
  name += (combiner == 1) ? "Mean" : "Sum";
  return name;
}

// ABI order pinned to the schema sequence [keys, begin, data, indices,
// begin_1] (picked by semantics; the base's fusedOps-order collection is
// site-internal values stay out of the ABI).
void orderedBoundaryInputs(func::FuncOp func,
                           SmallVectorImpl<Value> &inputs) {
  SegmentReduceArgs args;
  if (!classifySegmentReduceArgs(func, args)) return;
  for (Value v : {args.keys, args.begin, args.data, args.indices, args.begin_1})
    inputs.push_back(v);
}

template <typename SsOp, int64_t Combiner>
struct KpEmbeddingSparseSegmentReduceV2Rewrite
    : public CustomFusionPatternBase<SsOp> {
  using Base = CustomFusionPatternBase<SsOp>;
  using Base::Base;

  mlir::LogicalResult matchFusion(
      SsOp anchor, llvm::SmallVectorImpl<mlir::Operation *> &fusedOps)
      const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!func || !func->hasAttr("annc.kernel")) return mlir::failure();
    Type indicesType;
    return matchSegmentReduceV2(anchor.getOperation(), Combiner, func,
                                indicesType, fusedOps);
  }

  std::string getCustomOpName(
      SsOp anchor, llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    SegmentReduceArgs args;
    Type indicesType;
    if (func && classifySegmentReduceArgs(func, args))
      indicesType = args.indices.getType();
    return segmentReduceOpName(Combiner, indicesType);
  }

  CustomOpSchema getCustomOpSchema(
      SsOp anchor, llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    // Shared registry schema (V1 parity): keys(2-D) + begin(1-D) +
    // data(2-D,T) + indices(1-D,Tidx) + begin_1(1-D) → output(2-D,T) +
    // slice_output(0-D); combiner/indices dtype live in the op name.
    return *getContextKernelSchema(getCustomOpName(anchor, fusedOps));
  }

  void getOrderedBoundaryInputs(
      SsOp anchor, llvm::ArrayRef<mlir::Operation *> fusedOps,
      llvm::SmallVectorImpl<mlir::Value> &inputs) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!func) return;
    orderedBoundaryInputs(func, inputs);
  }
};

using KpEmbeddingSparseSegmentMeanV2Rewrite =
    KpEmbeddingSparseSegmentReduceV2Rewrite<atir::SparseSegmentMeanV2Op, 1>;
using KpEmbeddingSparseSegmentSumV2Rewrite =
    KpEmbeddingSparseSegmentReduceV2Rewrite<atir::SparseSegmentSumV2Op, 0>;

REGISTER_CUSTOM_PATTERN(KpEmbeddingSparseSegmentMeanV2Rewrite);
REGISTER_CUSTOM_PATTERN(KpEmbeddingSparseSegmentSumV2Rewrite);

}  // namespace
