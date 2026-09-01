#include "Dialect/Atir/Passes/Patterns/CustomFusionPatternBase.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistryMacros.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Block.h"

using namespace mlir;
using namespace atir;

namespace {

// Shared chain match (from the ConcatV2 back): <Cast/StridedSlice 外壳> ->
// Sub -> Pack -> Fill, filling boundary values and escape checks.
struct PaddingChain {
  ConcatV2Op concat;
  FillOp fill;
  PackOp pack;
  SubOp sub;
  // Sub.x 与 originShape 之间被剥掉的 Cast/StridedSlice 层, 由外向内排列
  // (peeled[0] 直接喂 Sub)。规范序是 {ss, Cast}, presort 镜像序是
  // {Cast, ss, ss} —— 见一级 peelKpEmbeddingPaddingOriginShape。
  SmallVector<Operation *, 3> peeled;
  Value originShape;
  Value data;
  Value inputRows;
  Value packVal;
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

bool checkConstantZeroForChain(Value v) {
  auto constOp = v.getDefiningOp<ConstantOp>();
  if (!constOp) return false;
  auto tensorType = dyn_cast<atir::TensorType>(v.getType());
  if (!tensorType) return false;
  DenseElementsAttr dataAttr = tensorType.getCacheData();
  if (!dataAttr) return false;
  if (dataAttr.getElementType().isF32()) {
    return dataAttr.isSplat()
               ? dataAttr.getSplatValue<float>() == 0.0f
               : (*dataAttr.getValues<float>().begin()) == 0.0f;
  }
  if (dataAttr.getElementType().isIntOrIndex()) {
    return dataAttr.isSplat()
               ? dataAttr.getSplatValue<APInt>().isZero()
               : (*dataAttr.getValues<APInt>().begin()).isZero();
  }
  return false;
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

void collectPaddingOps(Operation *op, ArrayRef<Value> boundaryValues,
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
    collectPaddingOps(operand.getDefiningOp(), boundaryValues, visited, ops);
  }
  ops.push_back(op);
}

LogicalResult matchPaddingChain(ConcatV2Op concat, PaddingChain &chain) {
  chain.concat = concat;
  if (concat.getValues().size() != 2) return failure();
  if (!checkConstantIntForChain(concat.getAxis(), 0)) return failure();

  auto fill = concat.getValues()[1].getDefiningOp<FillOp>();
  if (!fill) return failure();
  chain.fill = fill;
  if (!checkConstantZeroForChain(fill.getValueInput())) return failure();

  auto pack = fill.getShapeInput().getDefiningOp<PackOp>();
  if (!pack || pack.getInputs().size() != 2) return failure();
  chain.pack = pack;
  if (pack.getAxis() != 0) return failure();

  Value subVal, packVal;
  if (pack.getInputs()[0].getDefiningOp<SubOp>()) {
    subVal = pack.getInputs()[0];
    packVal = pack.getInputs()[1];
  } else if (pack.getInputs()[1].getDefiningOp<SubOp>()) {
    subVal = pack.getInputs()[1];
    packVal = pack.getInputs()[0];
  } else {
    return failure();
  }
  auto sub = subVal.getDefiningOp<SubOp>();
  if (!sub) return failure();
  chain.sub = sub;

  // 一级同放宽: 从 Sub.x 剥 0..1 层 Cast + 1..2 层 StridedSlice, 兼容规范序
  // ss(Cast(origin)) 与 presort 镜像序 Cast(ss(ss(origin))), 两者都等价于
  // int32(originShape[0])。
  Value cur = sub.getX();
  int numCast = 0;
  int numSlice = 0;
  while (true) {
    Operation *def = cur.getDefiningOp();
    if (auto castOp = dyn_cast_or_null<CastOp>(def)) {
      if (++numCast > 1) return failure();
      chain.peeled.push_back(castOp.getOperation());
      cur = castOp.getInput();
      continue;
    }
    auto ssOp = dyn_cast_or_null<StridedSliceOp>(def);
    if (!ssOp) break;
    if (++numSlice > 2) return failure();
    if (ssOp.getShrinkAxisMask() != 1) return failure();
    bool oneElem = checkConstantIntsForChain(ssOp.getBegin(), {0}) &&
                   checkConstantIntsForChain(ssOp.getEnd(), {1}) &&
                   checkConstantIntsForChain(ssOp.getStrides(), {1});
    // 2 元素形态只在 new_axis_mask=2 下等价 (见一级注释)。
    bool twoElem = ssOp.getNewAxisMask() == 2 && ssOp.getEllipsisMask() == 0 &&
                   checkConstantIntsForChain(ssOp.getBegin(), {0, 0}) &&
                   checkConstantIntsForChain(ssOp.getEnd(), {1, 0}) &&
                   checkConstantIntsForChain(ssOp.getStrides(), {1, 1});
    if (!oneElem && !twoElem) return failure();
    chain.peeled.push_back(ssOp.getOperation());
    cur = ssOp.getInput();
  }
  if (numSlice < 1) return failure();

  chain.originShape = cur;
  chain.data = concat.getValues()[0];
  chain.inputRows = sub.getY();
  chain.packVal = packVal;
  return success();
}

bool checkPaddingEscape(PaddingChain &chain, Operation *concatUser) {
  // 剥掉的每一层只能喂链上的下一层 (最外层喂 Sub), 逐级不逃逸 —— 与放宽前
  // 的 cast→ss→sub 两级检查等强, 只是层数可变。
  for (size_t i = 0; i < chain.peeled.size(); ++i) {
    Operation *next =
        (i == 0) ? chain.sub.getOperation() : chain.peeled[i - 1];
    for (Operation *user : chain.peeled[i]->getResult(0).getUsers()) {
      if (user != next) return false;
    }
  }
  // V2: the sub result is a fused output — it feeds pack internally and the
  // enclosing func return externally.
  for (Operation *user : chain.sub.getResult().getUsers()) {
    if (user != chain.pack.getOperation() && !isa<func::ReturnOp>(user))
      return false;
  }
  for (Operation *user : chain.pack.getResult().getUsers()) {
    if (user != chain.fill.getOperation()) return false;
  }
  for (Operation *user : chain.fill.getResult().getUsers()) {
    if (user != chain.concat.getOperation()) return false;
  }
  for (Operation *user : chain.concat.getResult().getUsers()) {
    if (user != concatUser) return false;
  }
  return true;
}

bool checkPaddingDtype(Value originShape, Value data) {
  auto oType = dyn_cast<atir::TensorType>(originShape.getType());
  auto dType = dyn_cast<atir::TensorType>(data.getType());
  return oType && oType.getElementType().isInteger(64) &&
         oType.getShape().size() == 1 && dType &&
         dType.getElementType().isF32() && dType.getShape().size() == 2;
}

// V2 kernel func signature check: private func with annc.kernel, argument 0
// = !llvm.ptr, 5 memref inputs, 2 results.
LogicalResult checkV2KernelFunc(func::FuncOp func) {
  if (!func || !func->hasAttr("annc.kernel")) return failure();
  if (func.getNumArguments() != 6 ||
      !isa<LLVM::LLVMPointerType>(func.getArgument(0).getType()))
    return failure();
  if (func.getFunctionType().getResults().size() != 2) return failure();
  return success();
}

bool checkResultContract(Value value,
                         llvm::ArrayRef<Operation *> fusedUsers) {
  for (Operation *user : value.getUsers()) {
    if (isa<func::ReturnOp>(user)) continue;
    if (!llvm::is_contained(fusedUsers, user)) return false;
  }
  return !value.use_empty();
}

// ── Padding (anchor = Reshape) ─────────────────────────────────────────────
struct KpEmbeddingPaddingRewrite
    : public CustomFusionPatternBase<ReshapeOp> {
  KpEmbeddingPaddingRewrite(MLIRContext *context,
                            const CustomOpTypeFilter &customOpFilter,
                            PatternBenefit benefit = 8)
      : CustomFusionPatternBase<ReshapeOp>(context, customOpFilter, benefit) {}

  mlir::LogicalResult matchFusion(
      ReshapeOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (failed(checkV2KernelFunc(func))) return failure();
    Value originShapeArg = func.getArgument(1);
    Value inputRowsArg = func.getArgument(2);
    Value packArg = func.getArgument(3);
    Value dataArg = func.getArgument(4);
    Value reshapeSizesArg = func.getArgument(5);
    if (!isa<BufferOp>(anchor.getOutput().getDefiningOp())) return failure();

    // 注: 812 的 "anchor 有 ConcatV2 用户" 检查在一级 (OpFusion, 主图层面)
    // 已保证; kernel func 只克隆定义链, 不含该用户, 二级不再重复检查。
    PaddingChain chain;
    auto concat = anchor.getInput().getDefiningOp<ConcatV2Op>();
    if (!concat) {
      return failure();
    }
    if (failed(matchPaddingChain(concat, chain))) {
      return failure();
    }
    if (chain.originShape != originShapeArg || chain.data != dataArg ||
        chain.inputRows != inputRowsArg ||
        anchor.getTargetShape() != reshapeSizesArg ||
        chain.packVal != packArg) {
      return failure();
    }
    if (!isa<BufferOp>(chain.sub.getOutput().getDefiningOp())) {
      return failure();
    }
    if (!checkPaddingDtype(chain.originShape, chain.data)) {
      return failure();
    }
    if (!checkPaddingEscape(chain, anchor.getOperation())) {
      return failure();
    }

    SmallVector<Value, 5> boundaryValues = {originShapeArg, dataArg,
                                            inputRowsArg, reshapeSizesArg,
                                            packArg};
    SmallPtrSet<Operation *, 32> visited;
    collectPaddingOps(anchor.getOperation(), boundaryValues, visited,
                      fusedOps);
    if (fusedOps.empty()) return failure();
    if (!llvm::is_contained(fusedOps, chain.concat.getOperation()) ||
        !llvm::is_contained(fusedOps, chain.sub.getOperation()) ||
        !llvm::is_contained(fusedOps, anchor.getOperation()))
      return failure();

    // V2 result contract: sub result (padding_rows) feeds pack internally and
    // the enclosing return externally; anchor result is the return value.
    if (!checkResultContract(chain.sub.getResult(),
                             {chain.pack.getOperation()}) ||
        !checkResultContract(anchor.getResult(), {}))
      return failure();

    return success();
  }

  std::string getCustomOpName(
      ReshapeOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    return "KPFusedEmbeddingPadding";
  }

  CustomOpSchema getCustomOpSchema(
      ReshapeOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    // MemRef arg order = getOrderedBoundaryInputs = kernel wrapper inputs.
    // Results map to execution slots: 0 = padding_rows (0D i32),
    // 1 = out_data (2D f32).
    return CustomOpSchema::get("KPFusedEmbeddingPadding")
        .TypeVar("T")
        .MemRefArg("origin_shape", 1, "")
        .MemRefArg("input_rows", 0, "")
        .MemRefArg("pack", 0, "")
        .MemRefArg("data", 2, "T")
        .MemRefArg("reshape_sizes", 1, "")
        .Result("out_padding_rows", 0, "")
        .Result("out_data", 2, "T");
  }

  void getOrderedBoundaryInputs(
      ReshapeOp anchor, llvm::ArrayRef<mlir::Operation *> fusedOps,
      llvm::SmallVectorImpl<mlir::Value> &inputs) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    inputs.push_back(func.getArgument(1));  // origin_shape
    inputs.push_back(func.getArgument(2));  // input_rows
    inputs.push_back(func.getArgument(3));  // pack
    inputs.push_back(func.getArgument(4));  // data
    inputs.push_back(func.getArgument(5));  // reshape_sizes
  }
};

// ── PaddingFast (anchor = StridedSlice) ─────────────────────────────────────
struct KpEmbeddingPaddingFastRewrite
    : public CustomFusionPatternBase<StridedSliceOp> {
  KpEmbeddingPaddingFastRewrite(MLIRContext *context,
                                const CustomOpTypeFilter &customOpFilter,
                                PatternBenefit benefit = 8)
      : CustomFusionPatternBase<StridedSliceOp>(context, customOpFilter,
                                                benefit) {}

  mlir::LogicalResult matchFusion(
      StridedSliceOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (failed(checkV2KernelFunc(func))) return failure();
    Value originShapeArg = func.getArgument(1);
    Value inputRowsArg = func.getArgument(2);
    Value packArg = func.getArgument(3);
    Value dataArg = func.getArgument(4);
    Value reshapeSizesArg = func.getArgument(5);
    if (!isa<BufferOp>(anchor.getOutput().getDefiningOp())) return failure();

    if (anchor.getShrinkAxisMask() != 1) return failure();
    // anchor 必须取 shape 的第 0 元素 (kernel 输出 reshape_rows = shape[0])。
    // 812 只查 begin/end 形状不查值 — 移植补查。
    if (!checkConstantIntsForChain(anchor.getBegin(), {0}) ||
        !checkConstantIntsForChain(anchor.getEnd(), {1}) ||
        !checkConstantIntsForChain(anchor.getStrides(), {1}))
      return failure();
    auto beginTy = dyn_cast<atir::TensorType>(anchor.getBegin().getType());
    auto endTy = dyn_cast<atir::TensorType>(anchor.getEnd().getType());
    if (!beginTy || !endTy || beginTy.getShape().size() != 1 ||
        beginTy.getShape()[0] != 1 || endTy.getShape().size() != 1 ||
        endTy.getShape()[0] != 1)
      return failure();

    auto shape = anchor.getInput().getDefiningOp<ShapeOp>();
    if (!shape) return failure();
    auto reshape = shape.getInput().getDefiningOp<ReshapeOp>();
    if (!reshape) return failure();
    PaddingChain chain;
    auto concat = reshape.getInput().getDefiningOp<ConcatV2Op>();
    if (!concat || failed(matchPaddingChain(concat, chain))) return failure();
    if (chain.originShape != originShapeArg || chain.data != dataArg ||
        chain.inputRows != inputRowsArg ||
        reshape.getTargetShape() != reshapeSizesArg ||
        chain.packVal != packArg)
      return failure();
    if (!isa<BufferOp>(chain.sub.getOutput().getDefiningOp())) {
      return failure();
    }
    if (!checkPaddingDtype(chain.originShape, chain.data)) return failure();
    if (!checkPaddingEscape(chain, reshape.getOperation())) return failure();
    for (Operation *user : reshape.getResult().getUsers()) {
      if (user != shape.getOperation()) return failure();
    }
    for (Operation *user : shape.getResult().getUsers()) {
      if (user != anchor.getOperation()) return failure();
    }

    SmallVector<Value, 5> boundaryValues = {originShapeArg, dataArg,
                                            inputRowsArg, reshapeSizesArg,
                                            packArg};
    SmallPtrSet<Operation *, 32> visited;
    collectPaddingOps(anchor.getOperation(), boundaryValues, visited,
                      fusedOps);
    if (fusedOps.empty()) return failure();
    if (!llvm::is_contained(fusedOps, chain.concat.getOperation()) ||
        !llvm::is_contained(fusedOps, chain.sub.getOperation()) ||
        !llvm::is_contained(fusedOps, reshape.getOperation()) ||
        !llvm::is_contained(fusedOps, anchor.getOperation()))
      return failure();

    // V2 result contract: sub result (padding_rows) feeds pack internally and
    // the enclosing return externally; anchor result (reshape_rows scalar) is
    // the return value.
    if (!checkResultContract(chain.sub.getResult(),
                             {chain.pack.getOperation()}) ||
        !checkResultContract(anchor.getResult(), {}))
      return failure();

    return success();
  }

  std::string getCustomOpName(
      StridedSliceOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    return "KPFusedEmbeddingPaddingFast";
  }

  CustomOpSchema getCustomOpSchema(
      StridedSliceOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    return CustomOpSchema::get("KPFusedEmbeddingPaddingFast")
        .TypeVar("T")
        .MemRefArg("origin_shape", 1, "")
        .MemRefArg("input_rows", 0, "")
        .MemRefArg("pack", 0, "")
        .MemRefArg("data", 2, "T")
        .MemRefArg("reshape_sizes", 1, "")
        .Result("out_padding_rows", 0, "")
        .Result("out_reshape_rows", 0, "");
  }

  void getOrderedBoundaryInputs(
      StridedSliceOp anchor, llvm::ArrayRef<mlir::Operation *> fusedOps,
      llvm::SmallVectorImpl<mlir::Value> &inputs) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    inputs.push_back(func.getArgument(1));  // origin_shape
    inputs.push_back(func.getArgument(2));  // input_rows
    inputs.push_back(func.getArgument(3));  // pack
    inputs.push_back(func.getArgument(4));  // data
    inputs.push_back(func.getArgument(5));  // reshape_sizes
  }
};

}  // namespace

REGISTER_CUSTOM_PATTERN(KpEmbeddingPaddingRewrite);
REGISTER_CUSTOM_PATTERN(KpEmbeddingPaddingFastRewrite);
