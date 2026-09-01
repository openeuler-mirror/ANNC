#include "Dialect/Atir/Passes/Patterns/CustomFusionPatternBase.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistryMacros.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Block.h"

using namespace mlir;
using namespace atir;

namespace {

// Shared chain match for the 812 KPFusedSparseSelect subgraph (second level,
// anchor = ConcatV2), Execution V2: kernel func signature is
// (!llvm.ptr, a, b, c, gt, eq1, eq2, eq3) -> (x, y, w); the three .Result()
// entries replace A's output-buffer handling.
struct SparseSelectChain {
  ConcatV2Op anchor;
  WhereOp select0;
  WhereOp select1;
  WhereOp select2;
  CastOp cast;
  CompareOp greater;
  CompareOp equal;
  CompareOp equal1;
  CompareOp equal2;
  ReshapeOp reshape4;
  ReshapeOp reshape1;
  ReshapeOp reshape2;
  FillOp fill;
  FillOp fill0;
  FillOp fill1;
  FillOp fill2;
  Value a;
  Value b;
  Value c;
  Value gt;
  Value eq1;
  Value eq2;
  Value eq3;
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

bool checkConstantFloatForChain(Value v, float expected) {
  auto constOp = v.getDefiningOp<ConstantOp>();
  if (!constOp) return false;
  auto tensorType = dyn_cast<atir::TensorType>(v.getType());
  if (!tensorType) return false;
  DenseElementsAttr dataAttr = tensorType.getCacheData();
  if (!dataAttr) return false;
  if (!dataAttr.getElementType().isF32()) return false;
  if (dataAttr.isSplat())
    return dataAttr.getSplatValue<float>() == expected;
  if (dataAttr.getNumElements() != 1) return false;
  return (*dataAttr.getValues<float>().begin()) == expected;
}

// 匹配 Compare(reshape(x), const) (EQ/GT, reshape {-1,1}, 常量 0-D i32)。
ReshapeOp matchCompareAgainstConstForChain(CompareOp cmp, StringRef dir,
                                           Value *constOut) {
  if (!cmp || cmp.getComparisonDirection() != dir) return nullptr;
  auto reshape = cmp.getLhs().getDefiningOp<ReshapeOp>();
  Value constV = cmp.getRhs();
  if (!reshape) {
    reshape = cmp.getRhs().getDefiningOp<ReshapeOp>();
    constV = cmp.getLhs();
  }
  if (!reshape) return nullptr;
  if (!checkConstantIntsForChain(reshape.getTargetShape(), {-1, 1}))
    return nullptr;
  // 比较常量是 0-D i32: 主图中是 ConstantOp; kernel func 中是 block arg
  // (一级把 4 个比较常量作为边界输入传入)。
  if (auto c = constV.getDefiningOp<ConstantOp>()) {
    // 主图常量 — 值在 matchFusion 的边界检查里与对应 arg 比对。
  } else if (constV.getDefiningOp() != nullptr) {
    return nullptr;
  }
  auto t = dyn_cast<atir::TensorType>(constV.getType());
  if (!t || !t.getElementType().isInteger(32) || t.getShape().size() != 0)
    return nullptr;
  *constOut = constV;
  return reshape;
}

bool matchWhereThenFillForChain(WhereOp where, FillOp *fillOut) {
  if (!where || where.getInputs().size() != 3) return false;
  auto fill = where.getInputs()[1].getDefiningOp<FillOp>();
  if (!fill || !checkConstantFloatForChain(fill.getValueInput(), 1.0f))
    return false;
  if (fillOut) *fillOut = fill;
  return true;
}

LogicalResult matchSparseSelectChain(ConcatV2Op anchor,
                                     SparseSelectChain &chain) {
  chain.anchor = anchor;
  if (anchor.getValues().size() != 3) return failure();
  if (!checkConstantIntForChain(anchor.getAxis(), 1)) return failure();
  auto emptyType =
      dyn_cast<atir::TensorType>(anchor.getValues()[2].getType());
  if (!emptyType || !emptyType.getElementType().isF32() ||
      emptyType.getShape().size() != 2 || emptyType.getShape()[1] != 0)
    return failure();

  auto select0 = anchor.getValues()[0].getDefiningOp<WhereOp>();
  if (!select0 || select0.getInputs().size() != 3) return failure();
  chain.select0 = select0;
  auto equal1 = select0.getInputs()[0].getDefiningOp<CompareOp>();
  if (!equal1) return failure();
  chain.equal1 = equal1;
  if (!matchWhereThenFillForChain(select0, &chain.fill0)) return failure();
  auto select1 = select0.getInputs()[2].getDefiningOp<WhereOp>();
  if (!select1 || select1.getInputs().size() != 3) return failure();
  chain.select1 = select1;
  auto equal = select1.getInputs()[0].getDefiningOp<CompareOp>();
  if (!equal) return failure();
  chain.equal = equal;
  if (!matchWhereThenFillForChain(select1, &chain.fill)) return failure();
  auto cast = select1.getInputs()[2].getDefiningOp<CastOp>();
  if (!cast) return failure();
  chain.cast = cast;
  {
    auto t = dyn_cast<atir::TensorType>(cast.getResult().getType());
    if (!t || !t.getElementType().isF32()) return failure();
  }
  auto greater = cast.getInput().getDefiningOp<CompareOp>();
  if (!greater) return failure();
  chain.greater = greater;

  chain.reshape4 = matchCompareAgainstConstForChain(greater, "GT", &chain.gt);
  auto reshape1a = matchCompareAgainstConstForChain(equal, "EQ", &chain.eq1);
  auto reshape1b = matchCompareAgainstConstForChain(equal1, "EQ", &chain.eq2);
  if (!chain.reshape4 || !reshape1a || !reshape1b || reshape1a != reshape1b)
    return failure();
  chain.reshape1 = reshape1a;

  auto select2 = anchor.getValues()[1].getDefiningOp<WhereOp>();
  if (!select2 || select2.getInputs().size() != 3) return failure();
  chain.select2 = select2;
  auto equal2 = select2.getInputs()[0].getDefiningOp<CompareOp>();
  if (!equal2) return failure();
  chain.equal2 = equal2;
  if (!matchWhereThenFillForChain(select2, &chain.fill2)) return failure();
  auto fill1Op = select2.getInputs()[2].getDefiningOp<FillOp>();
  if (!fill1Op || !checkConstantFloatForChain(fill1Op.getValueInput(), 1.0f))
    return failure();
  chain.fill1 = fill1Op;

  chain.reshape2 = matchCompareAgainstConstForChain(equal2, "EQ", &chain.eq3);
  if (!chain.reshape2) return failure();

  chain.a = chain.reshape4.getInput();
  chain.b = chain.reshape1.getInput();
  chain.c = chain.reshape2.getInput();
  return success();
}

bool checkSparseSelectDtype(SparseSelectChain &chain) {
  for (Value v : {chain.a, chain.b, chain.c}) {
    auto t = dyn_cast<atir::TensorType>(v.getType());
    if (!t || !t.getElementType().isInteger(32)) return false;
  }
  // 比较常量在 kernel func 中是 block args (一级边界输入), 只查类型。
  for (Value v : {chain.gt, chain.eq1, chain.eq2, chain.eq3}) {
    auto t = dyn_cast<atir::TensorType>(v.getType());
    if (!t || !t.getElementType().isInteger(32) || t.getShape().size() != 0)
      return false;
  }
  auto t0 = dyn_cast<atir::TensorType>(chain.select0.getResult().getType());
  auto tw = dyn_cast<atir::TensorType>(chain.anchor.getResult().getType());
  return t0 && t0.getElementType().isF32() && t0.getShape().size() == 2 &&
         tw && tw.getElementType().isF32() && tw.getShape().size() == 2;
}

bool checkSparseSelectEscape(SparseSelectChain &chain) {
  // V2: fused outputs (reshape4/select0/anchor results) are consumed by the
  // enclosing func return; all other intermediate results stay internal.
  auto checkOnlyOrReturn = [](Value v, ArrayRef<Operation *> allowed) {
    for (Operation *user : v.getUsers()) {
      if (isa<func::ReturnOp>(user)) continue;
      if (!llvm::is_contained(allowed, user)) return false;
    }
    return !v.use_empty();
  };
  auto checkOnly = [](Value v, ArrayRef<Operation *> allowed) {
    for (Operation *user : v.getUsers()) {
      if (!llvm::is_contained(allowed, user)) return false;
    }
    return true;
  };
  if (!checkOnlyOrReturn(chain.reshape4.getResult(),
                         {chain.greater.getOperation()}))
    return false;
  if (!checkOnly(chain.greater.getResult(), {chain.cast.getOperation()}))
    return false;
  if (!checkOnly(chain.cast.getResult(), {chain.select1.getOperation()}))
    return false;
  // reshape_1 结果除 equal/equal_1 外还被 fill/fill_0 的 Shape 输入使用。
  {
    SmallVector<Operation *> r1Users = {chain.equal.getOperation(),
                                        chain.equal1.getOperation()};
    if (auto s = chain.fill.getShapeInput().getDefiningOp<ShapeOp>())
      r1Users.push_back(s.getOperation());
    if (auto s = chain.fill0.getShapeInput().getDefiningOp<ShapeOp>())
      r1Users.push_back(s.getOperation());
    if (!checkOnly(chain.reshape1.getResult(), r1Users)) return false;
  }
  if (!checkOnly(chain.equal.getResult(), {chain.select1.getOperation()}))
    return false;
  if (!checkOnly(chain.equal1.getResult(), {chain.select0.getOperation()}))
    return false;
  if (!checkOnly(chain.fill.getResult(), {chain.select1.getOperation()}))
    return false;
  if (!checkOnly(chain.fill0.getResult(), {chain.select0.getOperation()}))
    return false;
  if (!checkOnly(chain.select1.getResult(), {chain.select0.getOperation()}))
    return false;
  if (!checkOnlyOrReturn(chain.select0.getResult(),
                         {chain.anchor.getOperation()}))
    return false;
  // reshape_2 结果除 equal_2 外还被 fill_1/fill_2 的 Shape 输入使用。
  {
    SmallVector<Operation *> r2Users = {chain.equal2.getOperation()};
    if (auto s = chain.fill1.getShapeInput().getDefiningOp<ShapeOp>())
      r2Users.push_back(s.getOperation());
    if (auto s = chain.fill2.getShapeInput().getDefiningOp<ShapeOp>())
      r2Users.push_back(s.getOperation());
    if (!checkOnly(chain.reshape2.getResult(), r2Users)) return false;
  }
  if (!checkOnly(chain.equal2.getResult(), {chain.select2.getOperation()}))
    return false;
  if (!checkOnly(chain.fill1.getResult(), {chain.select2.getOperation()}))
    return false;
  if (!checkOnly(chain.fill2.getResult(), {chain.select2.getOperation()}))
    return false;
  if (!checkOnly(chain.select2.getResult(), {chain.anchor.getOperation()}))
    return false;
  // w (anchor result): return only.
  for (Operation *user : chain.anchor.getResult().getUsers()) {
    if (!isa<func::ReturnOp>(user)) return false;
  }
  return !chain.anchor.getResult().use_empty();
}

void collectSparseSelectOps(Operation *op, ArrayRef<Value> boundaryValues,
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
    collectSparseSelectOps(operand.getDefiningOp(), boundaryValues, visited,
                           ops);
  }
  ops.push_back(op);
}

// Second level: rewrite the V2 kernel func (extracted by OpFusion's
// FuseKpSparseSelectAsFuncCallPattern) into a single atir.Customize call to
// the hand-written KPFusedSparseSelect kernel.  The three .Result() entries
// map to execution slots 0 (x), 1 (y), 2 (w); A's eraseFusedOutputOp hack is
// not needed (Result-based outputs erase cleanly).
struct KpEmbeddingSparseSelectRewrite
    : public CustomFusionPatternBase<ConcatV2Op> {
  KpEmbeddingSparseSelectRewrite(MLIRContext *context,
                                 const CustomOpTypeFilter &customOpFilter,
                                 PatternBenefit benefit = 8)
      : CustomFusionPatternBase<ConcatV2Op>(context, customOpFilter,
                                            benefit) {}

  mlir::LogicalResult matchFusion(
      ConcatV2Op anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!func || !func->hasAttr("annc.kernel")) {
      return failure();
    }
    if (func.getNumArguments() != 8 ||
        !isa<LLVM::LLVMPointerType>(func.getArgument(0).getType())) {
      return failure();
    }
    if (func.getFunctionType().getResults().size() != 3) return failure();
    Value aArg = func.getArgument(1);
    Value bArg = func.getArgument(2);
    Value cArg = func.getArgument(3);
    Value gtArg = func.getArgument(4);
    Value eq1Arg = func.getArgument(5);
    Value eq2Arg = func.getArgument(6);
    Value eq3Arg = func.getArgument(7);

    SparseSelectChain chain;
    if (failed(matchSparseSelectChain(anchor, chain))) {
      return failure();
    }
    if (chain.a != aArg || chain.b != bArg || chain.c != cArg ||
        chain.gt != gtArg || chain.eq1 != eq1Arg || chain.eq2 != eq2Arg ||
        chain.eq3 != eq3Arg) {
      return failure();
    }
    if (!checkSparseSelectDtype(chain)) {
      return failure();
    }
    if (!checkSparseSelectEscape(chain)) {
      return failure();
    }

    SmallVector<Value, 7> boundaryValues = {aArg, bArg, cArg,
                                            gtArg, eq1Arg, eq2Arg, eq3Arg};
    SmallPtrSet<Operation *, 32> visited;
    SmallVector<Operation *> collected;
    collectSparseSelectOps(anchor.getOperation(), boundaryValues, visited,
                           collected);
    if (collected.empty()) return failure();
    if (!llvm::is_contained(collected, chain.reshape4.getOperation()) ||
        !llvm::is_contained(collected, chain.select0.getOperation()) ||
        !llvm::is_contained(collected, chain.select2.getOperation()) ||
        !llvm::is_contained(collected, anchor.getOperation()))
      return failure();

    // 显式主序: 保证 escaping-result 收集序 = [x, y, w] (= 一级
    // boundaryOutputs = execution slot 0/1/2)。链上 op 按序排前, 其余
    // 收集到的 op (empty 的 shape 链、常量等) 追加在后。
    SmallVector<Operation *, 16> mainOrder = {
        chain.reshape4.getOperation(), chain.reshape1.getOperation(),
        chain.reshape2.getOperation(), chain.greater.getOperation(),
        chain.cast.getOperation(),     chain.equal.getOperation(),
        chain.equal1.getOperation(),   chain.equal2.getOperation(),
        chain.fill.getOperation(),     chain.fill0.getOperation(),
        chain.fill1.getOperation(),    chain.fill2.getOperation(),
        chain.select1.getOperation(),  chain.select0.getOperation(),
        chain.select2.getOperation(),  chain.anchor.getOperation()};
    for (Operation *op : mainOrder) {
      if (llvm::is_contained(collected, op)) fusedOps.push_back(op);
    }
    for (Operation *op : collected) {
      if (!llvm::is_contained(fusedOps, op)) fusedOps.push_back(op);
    }
    return success();
  }

  std::string getCustomOpName(
      ConcatV2Op anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    return "KPFusedSparseSelect";
  }

  CustomOpSchema getCustomOpSchema(
      ConcatV2Op anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    // MemRef arg order = getOrderedBoundaryInputs = kernel wrapper inputs
    // (a, b, c, gt, eq1, eq2, eq3); T binds a/b/c and the 4 scalars.
    return CustomOpSchema::get("KPFusedSparseSelect")
        .TypeVar("T")
        .MemRefArg("a", 1, "T")
        .MemRefArg("b", 1, "T")
        .MemRefArg("c", 1, "T")
        .MemRefArg("gt", 0, "T")
        .MemRefArg("eq1", 0, "T")
        .MemRefArg("eq2", 0, "T")
        .MemRefArg("eq3", 0, "T")
        .Result("out_x", 2, "")
        .Result("out_y", 2, "")
        .Result("out_w", 2, "");
  }

  void getOrderedBoundaryInputs(
      ConcatV2Op anchor, llvm::ArrayRef<mlir::Operation *> fusedOps,
      llvm::SmallVectorImpl<mlir::Value> &inputs) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    inputs.push_back(func.getArgument(1));  // a
    inputs.push_back(func.getArgument(2));  // b
    inputs.push_back(func.getArgument(3));  // c
    inputs.push_back(func.getArgument(4));  // gt
    inputs.push_back(func.getArgument(5));  // eq1
    inputs.push_back(func.getArgument(6));  // eq2
    inputs.push_back(func.getArgument(7));  // eq3
  }
};

}  // namespace

REGISTER_CUSTOM_PATTERN(KpEmbeddingSparseSelectRewrite);
