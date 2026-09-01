#include "Dialect/Atir/Passes/Patterns/CustomFusionPatternBase.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistryMacros.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Block.h"

using namespace mlir;
using namespace atir;

namespace {

// Fusion pattern for the KP-style sparse segment reduce (second level,
// Execution V2).  The first level (OpFusion's
// FuseKpSparseSegmentReduceAsFuncCallPattern) extracts the subgraph into a
// private kernel func with signature
//   (!llvm.ptr, keys, begin, data, indices, begin_1)
//       -> (output f32[batch,E], slice_output i32)
// and fusion.metadata abi = "annc_execution_v2".  This pattern rewrites the
// func body into a single atir.CustomizeOp calling the hand-written
// KPFusedSparseSegmentReduce<Idx><Sum|Mean> kernel; the two schema .Result()
// entries map to execution slots 0/1 allocated inside the kernel.
//
// Matched subgraph (ATIR ops only):
//
//   data, indices ─► SparseSegmentMean|Sum(segment_ids=StridedSlice(keys))
//                       │
//                       └─► Shape ─► StridedSlice(shrink=1, anchor)
//   keys ─► StridedSlice(shrink=2) ─► segment_ids
//
// dtype contract: data float32 2-D, keys int64 2-D, indices int32/int64 1-D.
// combiner (Sum/Mean) and Tidx are kernel specializations resolved through
// the multi-name scheme (no attrs channel in the B KernelRegistry).
struct KpEmbeddingSparseSegmentReduceRewrite
    : public CustomFusionPatternBase<StridedSliceOp> {
  KpEmbeddingSparseSegmentReduceRewrite(
      MLIRContext *context, const CustomOpTypeFilter &customOpFilter,
      PatternBenefit benefit = 8)
      : CustomFusionPatternBase<StridedSliceOp>(context, customOpFilter,
                                                benefit) {}

  mlir::LogicalResult matchFusion(
      StridedSliceOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    // Only rewrite inside the V2 kernel func extracted by the first-level
    // pattern: private func with annc.kernel, argument 0 = !llvm.ptr,
    // 5 memref inputs, 2 results.
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!func || !func->hasAttr("annc.kernel")) {
      return failure();
    }
    if (func.getNumArguments() != 6 ||
        !isa<LLVM::LLVMPointerType>(func.getArgument(0).getType())) {
      return failure();
    }
    if (func.getFunctionType().getResults().size() != 2) {
      return failure();
    }
    Value keysArg = func.getArgument(1);
    Value beginArg = func.getArgument(2);
    Value dataArg = func.getArgument(3);
    Value indicesArg = func.getArgument(4);
    Value begin1Arg = func.getArgument(5);

    // anchor: shrink=1, begin constant 1-D with 1 element.
    if (anchor.getShrinkAxisMask() != 1) {
      return failure();
    }
    if (anchor.getEllipsisMask() != 0 || anchor.getNewAxisMask() != 0) {
      return failure();
    }
    auto begin1Type = dyn_cast<atir::TensorType>(anchor.getBegin().getType());
    if (!begin1Type || begin1Type.getShape().size() != 1 ||
        begin1Type.getShape()[0] != 1) {
      return failure();
    }

    auto shape = anchor.getInput().getDefiningOp<ShapeOp>();
    if (!shape) {
      return failure();
    }

    // ss_reduce: SparseSegmentMean (combiner=1) or SparseSegmentSum (0).
    Value dataVal, indicesVal, segmentIdsVal, numSegmentsVal;
    Operation *ssOp = nullptr;
    int64_t combiner = -1;
    if (auto m = shape.getInput().getDefiningOp<SparseSegmentMeanOp>()) {
      ssOp = m.getOperation();
      combiner = 1;
      dataVal = m.getData();
      indicesVal = m.getIndices();
      segmentIdsVal = m.getSegmentIds();
      numSegmentsVal = m.getNumSegments();
    } else if (auto s2 = shape.getInput().getDefiningOp<SparseSegmentSumOp>()) {
      ssOp = s2.getOperation();
      combiner = 0;
      dataVal = s2.getInput();
      indicesVal = s2.getIndices();
      segmentIdsVal = s2.getSegmentIds();
      numSegmentsVal = s2.getNumSegments();
    }
    if (combiner < 0 || !ssOp) {
      return failure();
    }
    if (!checkConstantInt(numSegmentsVal, 0)) {
      return failure();
    }

    // 一级同放宽: 接受 Cast(i32) ← StridedSlice(i64) 包装。
    auto keysSs = segmentIdsVal.getDefiningOp<StridedSliceOp>();
    if (!keysSs) {
      auto segCast = segmentIdsVal.getDefiningOp<CastOp>();
      if (!segCast) return failure();
      keysSs = segCast.getInput().getDefiningOp<StridedSliceOp>();
      if (!keysSs) return failure();
    }
    if (keysSs.getShrinkAxisMask() != 2) return failure();
    if (keysSs.getBeginMask() != 1 || keysSs.getEndMask() != 1) {
      return failure();
    }
    if (keysSs.getEllipsisMask() != 0 || keysSs.getNewAxisMask() != 0) {
      return failure();
    }
    if (!checkConstantInts(keysSs.getStrides(), {1, 1})) {
      return failure();
    }
    auto beginType = dyn_cast<atir::TensorType>(keysSs.getBegin().getType());
    if (!beginType || beginType.getShape().size() != 1 ||
        beginType.getShape()[0] != 2) {
      return failure();
    }

    // Boundary contract: the subgraph must be rooted at the func args; the
    // output-buffer operands become local BufferOps inside the kernel func.
    if (dataVal != dataArg || indicesVal != indicesArg ||
        keysSs.getInput() != keysArg || keysSs.getBegin() != beginArg ||
        anchor.getBegin() != begin1Arg) {
      return failure();
    }
    if (!isa<BufferOp>(ssOp->getOperand(0).getDefiningOp()) ||
        !isa<BufferOp>(anchor.getOutput().getDefiningOp())) {
      return failure();
    }

    // dtype contract: data f32 2-D, keys i64 2-D, indices i32/i64 1-D.
    auto dType = dyn_cast<atir::TensorType>(dataArg.getType());
    auto kType = dyn_cast<atir::TensorType>(keysArg.getType());
    auto iType = dyn_cast<atir::TensorType>(indicesArg.getType());
    if (!dType || !dType.getElementType().isF32() ||
        dType.getShape().size() != 2)
      return failure();
    if (!kType || !kType.getElementType().isInteger(64) ||
        kType.getShape().size() != 2)
      return failure();
    if (!iType || !(iType.getElementType().isInteger(32) ||
                    iType.getElementType().isInteger(64)) ||
        iType.getShape().size() != 1)
      return failure();

    // Escape contract: intermediates may not escape; the two fused outputs
    // are consumed only by the enclosing func return (V2 result contract).
    // keysSs result may feed ssOp directly or through a Cast wrapper.
    for (Operation *user : keysSs.getResult().getUsers()) {
      if (user == ssOp) continue;
      if (isa<CastOp>(user)) {
        for (Operation *castUser : user->getResult(0).getUsers()) {
          if (castUser != ssOp) return failure();
        }
        continue;
      }
      return failure();
    }
    for (Operation *user : shape.getResult().getUsers()) {
      if (user != anchor.getOperation()) return failure();
    }

    // Collect the subgraph in post-order, stopping at the func arguments.
    SmallVector<Value, 5> boundaryValues = {keysArg, beginArg, dataArg,
                                            indicesArg, begin1Arg};
    SmallPtrSet<Operation *, 32> visited;
    collectDefiningOpsPostOrder(anchor.getOperation(), boundaryValues, visited,
                                fusedOps);
    if (fusedOps.empty()) {
      return failure();
    }
    if (!llvm::is_contained(fusedOps, ssOp) ||
        !llvm::is_contained(fusedOps, shape.getOperation()) ||
        !llvm::is_contained(fusedOps, keysSs.getOperation()) ||
        !llvm::is_contained(fusedOps, anchor.getOperation())) {
      return failure();
    }

    // V2 result contract: fused outputs may be used by fused ops and the
    // enclosing func return only; external non-return users are illegal.
    Value ssResult = ssOp->getResult(0);
    Value sliceResult = anchor.getResult();
    auto checkResultUsers = [&](Value value,
                                llvm::ArrayRef<Operation *> fusedUsers) {
      for (Operation *user : value.getUsers()) {
        if (isa<func::ReturnOp>(user)) continue;
        if (!llvm::is_contained(fusedUsers, user)) return false;
      }
      return !value.use_empty();
    };
    if (!checkResultUsers(ssResult, {shape.getOperation()}) ||
        !checkResultUsers(sliceResult, {}))
      return failure();

    return success();
  }

  std::string getCustomOpName(
      StridedSliceOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    auto shape = anchor.getInput().getDefiningOp<ShapeOp>();
    bool isMean =
        shape && shape.getInput().getDefiningOp<SparseSegmentMeanOp>();
    auto iType = dyn_cast<atir::TensorType>(func.getArgument(4).getType());
    std::string name = "KPFusedSparseSegmentReduce";
    name += (iType && iType.getElementType().isInteger(64)) ? "I64" : "I32";
    name += isMean ? "Mean" : "Sum";
    return name;
  }

  CustomOpSchema getCustomOpSchema(
      StridedSliceOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    Value keysArg = func.getArgument(1);
    Value beginArg = func.getArgument(2);
    Value dataArg = func.getArgument(3);
    Value indicesArg = func.getArgument(4);
    Value begin1Arg = func.getArgument(5);
    auto getRank = [](Type type) -> int64_t {
      if (auto tensorType = dyn_cast<atir::TensorType>(type))
        return static_cast<int64_t>(tensorType.getShape().size());
      return 0;
    };
    // MemRef arg order matches getOrderedBoundaryInputs and the kernel
    // wrapper: keys, begin, data, indices, begin_1.  Tidx binds the indices
    // element type and matches the kernel's TypeConstraint.  Results map to
    // execution slots 0 (output) and 1 (slice_output).
    CustomOpSchema schema =
        CustomOpSchema::get(getCustomOpName(anchor, fusedOps))
            .TypeVar("T")
            .TypeVar("Tidx");
    schema.MemRefArg("keys", getRank(keysArg.getType()), "");
    schema.MemRefArg("begin", getRank(beginArg.getType()), "");
    schema.MemRefArg("data", getRank(dataArg.getType()), "T");
    schema.MemRefArg("indices", getRank(indicesArg.getType()), "Tidx");
    schema.MemRefArg("begin_1", getRank(begin1Arg.getType()), "");
    schema.Result("output", 2, "T");
    schema.Result("slice_output", 0, "");
    return schema;
  }

  void getOrderedBoundaryInputs(
      StridedSliceOp anchor, llvm::ArrayRef<mlir::Operation *> fusedOps,
      llvm::SmallVectorImpl<mlir::Value> &inputs) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    inputs.push_back(func.getArgument(1));  // keys
    inputs.push_back(func.getArgument(2));  // begin
    inputs.push_back(func.getArgument(3));  // data
    inputs.push_back(func.getArgument(4));  // indices
    inputs.push_back(func.getArgument(5));  // begin_1
  }

 private:
  static FailureOr<int64_t> getConstantInt(Value v) {
    auto constOp = v.getDefiningOp<ConstantOp>();
    if (!constOp) return failure();
    auto tensorType = dyn_cast<atir::TensorType>(v.getType());
    if (!tensorType) return failure();
    DenseElementsAttr dataAttr = tensorType.getCacheData();
    if (!dataAttr) return failure();
    if (!dataAttr.getElementType().isIntOrIndex()) return failure();
    if (dataAttr.isSplat()) {
      return dataAttr.getSplatValue<APInt>().getSExtValue();
    }
    if (dataAttr.getNumElements() != 1) return failure();
    return (*dataAttr.getValues<APInt>().begin()).getSExtValue();
  }

  static bool checkConstantInt(Value v, int64_t expected) {
    FailureOr<int64_t> valOr = getConstantInt(v);
    return succeeded(valOr) && *valOr == expected;
  }

  static FailureOr<SmallVector<int64_t>> getConstantInts(Value v) {
    auto constOp = v.getDefiningOp<ConstantOp>();
    if (!constOp) return failure();
    auto tensorType = dyn_cast<atir::TensorType>(v.getType());
    if (!tensorType) return failure();
    DenseElementsAttr dataAttr = tensorType.getCacheData();
    if (!dataAttr) return failure();
    if (!dataAttr.getElementType().isIntOrIndex()) return failure();
    ArrayRef<int64_t> shape = tensorType.getShape();
    if (shape.size() != 1) return failure();
    SmallVector<int64_t> values;
    values.reserve(dataAttr.getNumElements());
    for (const APInt &val : dataAttr.getValues<APInt>()) {
      values.push_back(val.getSExtValue());
    }
    return values;
  }

  static bool checkConstantInts(Value v, ArrayRef<int64_t> expected) {
    FailureOr<SmallVector<int64_t>> valsOr = getConstantInts(v);
    if (failed(valsOr)) return false;
    if (valsOr->size() != expected.size()) return false;
    for (size_t i = 0; i < expected.size(); ++i) {
      if ((*valsOr)[i] != expected[i]) return false;
    }
    return true;
  }

  static void collectDefiningOpsPostOrder(Operation *op,
                                          ArrayRef<Value> boundaryValues,
                                          SmallPtrSetImpl<Operation *> &visited,
                                          SmallVectorImpl<Operation *> &ops) {
    if (!op || visited.count(op)) return;
    if (isa<VariableOp>(op)) return;
    if (definesBoundary(op, boundaryValues)) return;
    visited.insert(op);

    for (Value operand : op->getOperands()) {
      if (llvm::is_contained(boundaryValues, operand)) continue;
      collectDefiningOpsPostOrder(operand.getDefiningOp(), boundaryValues,
                                  visited, ops);
    }
    ops.push_back(op);
  }

  static bool definesBoundary(Operation *op, ArrayRef<Value> boundaryValues) {
    for (Value result : op->getResults()) {
      if (llvm::is_contained(boundaryValues, result)) return true;
    }
    return false;
  }
};

REGISTER_CUSTOM_PATTERN(KpEmbeddingSparseSegmentReduceRewrite);

}  // namespace
