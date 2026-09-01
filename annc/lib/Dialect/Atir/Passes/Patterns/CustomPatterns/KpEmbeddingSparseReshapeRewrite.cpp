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
// anchor = SparseReshape, a functional op).  Execution V2: the kernel func
// signature is (!llvm.ptr, keys, begin, pack_const, new_shape) -> 2 results;
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
  Value limit;  // Range.limit (常量或 0-D i32 动态边界)
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
  // 812 只查 delta=1; 移植补查 start=0。limit 允许常量或 0-D i32 动态值
  // (kernel func 中是额外 block arg; kernel 不用该值)。
  if (!checkConstantIntForChain(range.getStart(), 0)) return failure();
  if (!checkConstantIntForChain(range.getDelta(), 1)) return failure();
  if (!checkConstantForChain(range.getLimit())) {
    auto lType = dyn_cast<atir::TensorType>(range.getLimit().getType());
    if (!lType || !lType.getElementType().isInteger(32) ||
        lType.getShape().size() != 0)
      return failure();
  }

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
  chain.shape = shape;

  chain.keys = shape.getInput();
  chain.begin = ss.getBegin();
  chain.newShape = anchor.getNewShape();
  chain.packConst = pack.getInputs()[1];
  chain.limit = range.getLimit();
  return success();
}

bool checkSparseReshapeDtype(SparseReshapeChain &chain) {
  // new_shape 固定 i64 (SparseReshape/ReshapeKp 语义), pack_const 为 T
  // (i32/i64, kernel 特化) — 两者不必一致。
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
  if (!pType || !(pType.getElementType().isInteger(32) ||
                  pType.getElementType().isInteger(64)) ||
      pType.getShape().size() != 0)
    return false;
  auto t1 = dyn_cast<atir::TensorType>(chain.cast1.getResult().getType());
  auto t2 = dyn_cast<atir::TensorType>(chain.cast.getResult().getType());
  return t1 && t1.getElementType().isInteger(64) && t2 &&
         t2.getElementType().isInteger(64);
}

bool checkSparseReshapeEscape(SparseReshapeChain &chain,
                              Operation *anchorOp) {
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

// Second level: rewrite the V2 kernel func (extracted by OpFusion's
// FuseKpSparseReshapeAsFuncCallPattern) into a single atir.Customize call to
// the hand-written KPFusedSparseReshape<Idx> kernel.  The two .Result()
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
    if ((func.getNumArguments() != 5 && func.getNumArguments() != 6) ||
        !isa<LLVM::LLVMPointerType>(func.getArgument(0).getType())) {
      return failure();
    }
    if (func.getFunctionType().getResults().size() != 2) return failure();
    Value keysArg = func.getArgument(1);
    Value beginArg = func.getArgument(2);
    Value packConstArg = func.getArgument(3);
    Value newShapeArg = func.getArgument(4);
    Value limitArg =
        func.getNumArguments() == 6 ? func.getArgument(5) : Value();

    SparseReshapeChain chain;
    if (failed(matchSparseReshapeChain(anchor, chain))) {
      return failure();
    }
    if (chain.keys != keysArg || chain.begin != beginArg ||
        chain.newShape != newShapeArg || chain.packConst != packConstArg) {
      return failure();
    }
    // 动态 limit ⇔ 6-arg 签名 (一级同约定); 5-arg 时 limit 必须是常量, 或是
    // 链内交叉引用 ss1 (= num_rows, 一级 limitFromSs1 走非 DynLimit 特化)。
    if (static_cast<bool>(limitArg)) {
      if (chain.limit != limitArg) return failure();
    } else if (!checkConstantForChain(chain.limit) &&
               chain.limit != chain.ss1.getResult()) {
      return failure();
    }
    if (!checkSparseReshapeDtype(chain)) {
      return failure();
    }
    if (!checkSparseReshapeEscape(chain, anchor.getOperation())) {
      return failure();
    }

    SmallVector<Value, 5> boundaryValues = {keysArg, beginArg, packConstArg,
                                            newShapeArg};
    if (static_cast<bool>(limitArg)) boundaryValues.push_back(limitArg);
    SmallPtrSet<Operation *, 32> visited;
    collectSparseReshapeOps(anchor.getOperation(), boundaryValues, visited,
                            fusedOps);
    if (fusedOps.empty()) return failure();
    if (!llvm::is_contained(fusedOps, chain.concat.getOperation()) ||
        !llvm::is_contained(fusedOps, chain.range.getOperation()) ||
        !llvm::is_contained(fusedOps, chain.pack.getOperation()) ||
        !llvm::is_contained(fusedOps, anchor.getOperation()))
      return failure();

    // V2 result contract: the two functional anchor results are consumed by
    // the enclosing func return only.
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
    auto packType =
        dyn_cast<atir::TensorType>(func.getArgument(3).getType());
    std::string name = "KPFusedSparseReshape";
    name +=
        (packType && packType.getElementType().isInteger(64)) ? "I64" : "I32";
    if (func.getNumArguments() == 6) name += "DynLimit";
    return name;
  }

  CustomOpSchema getCustomOpSchema(
      SparseReshapeOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    // MemRef arg order = getOrderedBoundaryInputs = kernel wrapper inputs.
    // T binds pack_const only; new_shape is fixed i64.  DynLimit variants
    // carry an extra unused 0-D i32 range_limit.
    auto func = anchor->template getParentOfType<func::FuncOp>();
    auto schema = CustomOpSchema::get(getCustomOpName(anchor, fusedOps))
                      .TypeVar("T")
                      .MemRefArg("slice_input", 2, "")
                      .MemRefArg("begin", 1, "")
                      .MemRefArg("pack_const", 0, "T")
                      .MemRefArg("new_shape", 1, "");
    if (func.getNumArguments() == 6) schema.MemRefArg("range_limit", 0, "");
    return schema.Result("out_indices", 2, "").Result("out_shape", 1, "");
  }

  void getOrderedBoundaryInputs(
      SparseReshapeOp anchor, llvm::ArrayRef<mlir::Operation *> fusedOps,
      llvm::SmallVectorImpl<mlir::Value> &inputs) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    inputs.push_back(func.getArgument(1));  // keys (slice_input)
    inputs.push_back(func.getArgument(2));  // begin
    inputs.push_back(func.getArgument(3));  // pack_const
    inputs.push_back(func.getArgument(4));  // new_shape
    if (func.getNumArguments() == 6)
      inputs.push_back(func.getArgument(5));  // range_limit (dynamic limit)
  }
};

}  // namespace

REGISTER_CUSTOM_PATTERN(KpEmbeddingSparseReshapeRewrite);
