#include "Dialect/Atir/BuiltinKernelSchema.h"
#include "Dialect/Atir/Passes/Patterns/CustomFusionPatternBase.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistryMacros.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Block.h"

using namespace mlir;
using namespace atir;

namespace {

// Shared chain match for the 812 KPFusedSparseReshape subgraph (second level,
// anchor = SparseReshape, a functional op).  Context V3: the kernel func
// signature is (keys, begin, pack_const, new_shape) -> 2 results;
// the .Result() schema entries replace A's functional output buffer handling
// (mapFunctionalOutputBuffers is not needed).
struct SparseReshapeChain {
  ConcatV2Op concat;
  CastOp cast1;
  ReshapeOp reshape1;
  RangeOp range;
  ReshapeOp reshape;
  StridedSliceOp ss;
  CastOp cast;
  PackOp pack;
  StridedSliceOp ss1;
  ShapeOp shape;
  Value keys;
  Value begin;
  Value newShape;
  Value packConst;
};

// 常量 helper (与其余 812 二级 pattern 相同实现)。
FailureOr<int64_t> getConstInt(Value v) {
  auto constOp = v.getDefiningOp<ConstantOp>();
  if (!constOp) return failure();
  auto tensorType = dyn_cast<atir::TensorType>(v.getType());
  if (!tensorType) return failure();
  DenseElementsAttr dataAttr = tensorType.getCacheData();
  if (!dataAttr) return failure();
  if (!dataAttr.getElementType().isIntOrIndex()) return failure();
  if (dataAttr.isSplat()) return dataAttr.getSplatValue<APInt>().getSExtValue();
  if (dataAttr.getNumElements() != 1) return failure();
  return (*dataAttr.getValues<APInt>().begin()).getSExtValue();
}

bool checkConstantIntForChain(Value v, int64_t expected) {
  FailureOr<int64_t> valOr = getConstInt(v);
  return succeeded(valOr) && *valOr == expected;
}

bool checkConstantIntsForChain(Value v, ArrayRef<int64_t> expected) {
  auto constOp = v.getDefiningOp<ConstantOp>();
  if (!constOp) return false;
  auto tensorType = dyn_cast<atir::TensorType>(v.getType());
  if (!tensorType) return false;
  DenseElementsAttr dataAttr = tensorType.getCacheData();
  if (!dataAttr) return false;
  if (!dataAttr.getElementType().isIntOrIndex()) return false;
  ArrayRef<int64_t> shape = tensorType.getShape();
  if (shape.size() != 1) return false;
  SmallVector<int64_t> values;
  for (const APInt &val : dataAttr.getValues<APInt>()) {
    values.push_back(val.getSExtValue());
  }
  if (values.size() != expected.size()) return false;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (values[i] != expected[i]) return false;
  }
  return true;
}

bool checkConstantForChain(Value v) {
  return v.getDefiningOp<ConstantOp>() != nullptr;
}

LogicalResult matchSparseReshapeChain(SparseReshapeOp anchor,
                                      SparseReshapeChain &chain) {
  // ── indices 分支: ConcatV2([cast_1, reshape], axis={-1}) ──
  auto concat = anchor.getIndices().getDefiningOp<ConcatV2Op>();
  if (!concat) return failure();
  chain.concat = concat;
  if (concat.getValues().size() != 2) return failure();
  // Keep the original PR contract: concatenate the two index columns along
  // the last dimension using the negative axis spelling.
  if (!checkConstantIntForChain(concat.getAxis(), -1)) return failure();

  // 分支 A: Cast(Reshape(Range(0, limit, 1), {-1,1})).
  auto cast1 = concat.getValues()[0].getDefiningOp<CastOp>();
  if (!cast1) return failure();
  chain.cast1 = cast1;
  auto reshape1 = cast1.getInput().getDefiningOp<ReshapeOp>();
  if (!reshape1) return failure();
  chain.reshape1 = reshape1;
  if (!checkConstantIntsForChain(reshape1.getTargetShape(), {-1, 1}))
    return failure();
  auto range = reshape1.getInput().getDefiningOp<RangeOp>();
  if (!range) return failure();
  chain.range = range;
  // The downloaded kernel regenerates the row range from the input row count;
  // only a constant limit or the in-chain Shape(keys)[0] value is valid.
  if (!checkConstantIntForChain(range.getStart(), 0)) return failure();
  if (!checkConstantIntForChain(range.getDelta(), 1)) return failure();

  // 分支 B: Reshape(StridedSlice(keys, begin), {-1,1}).
  auto reshape = concat.getValues()[1].getDefiningOp<ReshapeOp>();
  if (!reshape) return failure();
  chain.reshape = reshape;
  if (!checkConstantIntsForChain(reshape.getTargetShape(), {-1, 1}))
    return failure();
  auto ss = reshape.getInput().getDefiningOp<StridedSliceOp>();
  if (!ss) return failure();
  chain.ss = ss;
  if (ss.getShrinkAxisMask() != 2) return failure();
  if (ss.getBeginMask() != 1 || ss.getEndMask() != 1) return failure();
  if (ss.getEllipsisMask() != 0 || ss.getNewAxisMask() != 0) return failure();
  {
    auto t = dyn_cast<atir::TensorType>(ss.getBegin().getType());
    if (!t || t.getShape().size() != 1 || t.getShape()[0] != 2)
      return failure();
  }
  if (!checkConstantForChain(ss.getEnd()) ||
      !checkConstantForChain(ss.getStrides()))
    return failure();

  // ── input_shape 分支: Cast(Pack([StridedSlice(shrink=1)(Shape(keys)),
  // pack_const])) ──
  auto cast = anchor.getInputShape().getDefiningOp<CastOp>();
  if (!cast) return failure();
  chain.cast = cast;
  auto pack = cast.getInput().getDefiningOp<PackOp>();
  if (!pack || pack.getInputs().size() != 2) return failure();
  chain.pack = pack;
  if (pack.getAxis() != 0) return failure();
  auto ss1 = pack.getInputs()[0].getDefiningOp<StridedSliceOp>();
  if (!ss1 || ss1.getShrinkAxisMask() != 1) return failure();
  chain.ss1 = ss1;
  if (!checkConstantIntsForChain(ss1.getBegin(), {0}) ||
      !checkConstantIntsForChain(ss1.getEnd(), {1}) ||
      !checkConstantIntsForChain(ss1.getStrides(), {1}))
    return failure();
  auto shape = ss1.getInput().getDefiningOp<ShapeOp>();
  if (!shape) return failure();
  if (!checkConstantForChain(range.getLimit()) &&
      range.getLimit() != ss1.getResult())
    return failure();
  chain.shape = shape;

  chain.keys = shape.getInput();
  chain.begin = ss.getBegin();
  chain.newShape = anchor.getNewShape();
  chain.packConst = pack.getInputs()[1];
  return success();
}

bool checkSparseReshapeDtype(SparseReshapeChain &chain) {
  // new_shape and pack_const follow the downloaded kernel's i64 contract.
  auto kType = dyn_cast<atir::TensorType>(chain.keys.getType());
  auto bType = dyn_cast<atir::TensorType>(chain.begin.getType());
  auto nType = dyn_cast<atir::TensorType>(chain.newShape.getType());
  auto pType = dyn_cast<atir::TensorType>(chain.packConst.getType());
  if (!kType || !kType.getElementType().isInteger(64) ||
      kType.getShape().size() != 2)
    return false;
  if (!bType || !bType.getElementType().isInteger(32) ||
      bType.getShape().size() != 1 || bType.getShape()[0] != 2)
    return false;
  if (!nType || !nType.getElementType().isInteger(64) ||
      nType.getShape().size() != 1 || nType.getShape()[0] != 2)
    return false;
  if (!pType || !pType.getElementType().isInteger(64) ||
      pType.getShape().size() != 0)
    return false;
  auto t1 = dyn_cast<atir::TensorType>(chain.cast1.getResult().getType());
  auto t2 = dyn_cast<atir::TensorType>(chain.cast.getResult().getType());
  return t1 && t1.getElementType().isInteger(64) && t2 &&
         t2.getElementType().isInteger(64);
}

bool checkSparseReshapeEscape(SparseReshapeChain &chain, Operation *anchorOp) {
  for (Operation *user : chain.range.getResult().getUsers()) {
    if (user != chain.reshape1.getOperation()) return false;
  }
  for (Operation *user : chain.reshape1.getResult().getUsers()) {
    if (user != chain.cast1.getOperation()) return false;
  }
  for (Operation *user : chain.cast1.getResult().getUsers()) {
    if (user != chain.concat.getOperation()) return false;
  }
  for (Operation *user : chain.ss.getResult().getUsers()) {
    if (user != chain.reshape.getOperation()) return false;
  }
  for (Operation *user : chain.reshape.getResult().getUsers()) {
    if (user != chain.concat.getOperation()) return false;
  }
  for (Operation *user : chain.concat.getResult().getUsers()) {
    if (user != anchorOp) return false;
  }
  for (Operation *user : chain.shape.getResult().getUsers()) {
    if (user != chain.ss1.getOperation()) return false;
  }
  // D-2 放宽 2 (一级同): ss1 = Shape(keys)[0] = num_rows 同时喂 input_shape
  // 分支的 Pack 与 indices 分支 Range 的 limit。两端都在融合区内, 是链内
  // 交叉引用而非逃逸。
  bool limitFromSs1 = (chain.range.getLimit() == chain.ss1.getResult());
  for (Operation *user : chain.ss1.getResult().getUsers()) {
    if (user == chain.pack.getOperation()) continue;
    if (limitFromSs1 && user == chain.range.getOperation()) continue;
    return false;
  }
  for (Operation *user : chain.pack.getResult().getUsers()) {
    if (user != chain.cast.getOperation()) return false;
  }
  for (Operation *user : chain.cast.getResult().getUsers()) {
    if (user != anchorOp) return false;
  }
  return true;
}

void collectSparseReshapeOps(Operation *op, ArrayRef<Value> boundaryValues,
                             SmallPtrSetImpl<Operation *> &visited,
                             SmallVectorImpl<Operation *> &ops) {
  if (!op || visited.count(op)) return;
  if (isa<VariableOp>(op)) return;
  if (isa<BufferOp>(op)) return;
  for (Value result : op->getResults()) {
    if (llvm::is_contained(boundaryValues, result)) return;
  }
  visited.insert(op);
  for (Value operand : op->getOperands()) {
    if (llvm::is_contained(boundaryValues, operand)) continue;
    collectSparseReshapeOps(operand.getDefiningOp(), boundaryValues, visited,
                            ops);
  }
  ops.push_back(op);
}

// Second level: rewrite the context kernel func (extracted by OpFusion's
// FuseKpSparseReshapeAsFuncCallPattern) into a single atir.Customize call to
// the hand-written KPFusedSparseReshapeI64 kernel.  The two .Result()
// entries map to execution slots 0 (out_indices) and 1 (out_shape).
struct KpEmbeddingSparseReshapeRewrite
    : public CustomFusionPatternBase<SparseReshapeOp> {
  KpEmbeddingSparseReshapeRewrite(MLIRContext *context,
                                  const CustomOpTypeFilter &customOpFilter,
                                  PatternBenefit benefit = 8)
      : CustomFusionPatternBase<SparseReshapeOp>(context, customOpFilter,
                                                 benefit) {}

  mlir::LogicalResult matchFusion(
      SparseReshapeOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!func || !func->hasAttr("annc.kernel")) {
      return failure();
    }
    if (func.getNumArguments() != 4) return failure();
    if (func.getFunctionType().getResults().size() != 2) return failure();
    Value keysArg = func.getArgument(0);
    Value beginArg = func.getArgument(1);
    Value packConstArg = func.getArgument(2);
    Value newShapeArg = func.getArgument(3);

    SparseReshapeChain chain;
    if (failed(matchSparseReshapeChain(anchor, chain))) {
      return failure();
    }
    if (chain.keys != keysArg || chain.begin != beginArg ||
        chain.newShape != newShapeArg || chain.packConst != packConstArg) {
      return failure();
    }
    if (!checkSparseReshapeDtype(chain)) {
      return failure();
    }
    if (!checkSparseReshapeEscape(chain, anchor.getOperation())) {
      return failure();
    }

    SmallVector<Value, 4> boundaryValues = {keysArg, beginArg, packConstArg,
                                            newShapeArg};
    SmallPtrSet<Operation *, 32> visited;
    collectSparseReshapeOps(anchor.getOperation(), boundaryValues, visited,
                            fusedOps);
    if (fusedOps.empty()) return failure();
    if (!llvm::is_contained(fusedOps, chain.concat.getOperation()) ||
        !llvm::is_contained(fusedOps, chain.range.getOperation()) ||
        !llvm::is_contained(fusedOps, chain.pack.getOperation()) ||
        !llvm::is_contained(fusedOps, anchor.getOperation()))
      return failure();

    // context result contract: the two functional anchor results are consumed
    // by the enclosing func return only.
    for (Value result : anchor->getResults()) {
      bool usedByReturnOnly = !result.use_empty();
      for (Operation *user : result.getUsers()) {
        if (!isa<func::ReturnOp>(user)) usedByReturnOnly = false;
      }
      if (!usedByReturnOnly) return failure();
    }

    return success();
  }

  std::string getCustomOpName(
      SparseReshapeOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    return "KPFusedSparseReshapeI64";
  }

  CustomOpSchema getCustomOpSchema(
      SparseReshapeOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    return *getContextKernelSchema(getCustomOpName(anchor, fusedOps));
  }

  void getOrderedBoundaryInputs(
      SparseReshapeOp anchor, llvm::ArrayRef<mlir::Operation *> fusedOps,
      llvm::SmallVectorImpl<mlir::Value> &inputs) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    inputs.push_back(func.getArgument(0));  // keys (slice_input)
    inputs.push_back(func.getArgument(1));  // begin
    inputs.push_back(func.getArgument(2));  // pack_const
    inputs.push_back(func.getArgument(3));  // new_shape
  }
};

}  // namespace

REGISTER_CUSTOM_PATTERN(KpEmbeddingSparseReshapeRewrite);
