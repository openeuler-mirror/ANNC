#include "Dialect/Atir/Passes/Patterns/CustomFusionPatternBase.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistryMacros.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"

using namespace mlir;
using namespace atir;

// Fusion pattern for the KP-style embedding action-id gather (second level).
//
// The first level (OpFusion's FuseKpEmbeddingActionIdGatherAsFuncCallPattern)
// extracts the subgraph into a private func with annc.kernel +
// fusion.metadata, taking (indices1, params, indices2, pack_dim, pack,
// output).  This pattern rewrites that func's body into a single
// atir.Customize call to the hand-written KPFusedEmbeddingActionIdGather
// kernel.
//
// Matched subgraph (ATIR ops only):
//
//   indices1 ─► Gather(inner, axis=0) ─► Gather(outer, axis=0) ─► Reshape
//   params   ─► inner params                                shape=Pack(-1)
//   indices2 ─► outer indices                                        │
//   pack_dim, pack ─► Pack ─► Fill(value=0) ────────────────────────►┴
//                   ConcatV2(axis=-1, anchor) ◄───────────────────────┘
//
// Fused kernel: output[i, j, (k), :] = params[indices1[indices2[i, j], (k)]]
// reshaped to [pack_size, a_reshaped_cols] and zero-padded per row with
// pack_const columns.
//
// v1 dtype contract: indices1 int64, indices2 int32, params float32 (the op
// registration defaults).  R1/R2 (indices ranks, 1 or 2) become kernel
// specialization attrs.
struct KpEmbeddingActionIdGatherRewrite
    : public CustomFusionPatternBase<ConcatV2Op> {
  KpEmbeddingActionIdGatherRewrite(MLIRContext *context,
                                   const CustomOpTypeFilter &customOpFilter,
                                   PatternBenefit benefit = 8)
      : CustomFusionPatternBase<ConcatV2Op>(context, customOpFilter, benefit) {}

  mlir::LogicalResult matchFusion(
      ConcatV2Op anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    // Only rewrite inside the kernel func extracted by the first-level
    // pattern: private func with annc.kernel, 6 arguments, no results.
    auto func = anchor->template getParentOfType<func::FuncOp>();

    if (!func || !func->hasAttr("annc.kernel")) {
      return failure();
    }

    if (func.getNumArguments() != 6) {
      return failure();
    }

    if (!func.getFunctionType().getResults().empty()) {
      return failure();
    }
    Value indices1Arg = func.getArgument(0);
    Value paramsArg = func.getArgument(1);
    Value indices2Arg = func.getArgument(2);
    Value packDimArg = func.getArgument(3);
    Value packArg = func.getArgument(4);
    Value outputArg = func.getArgument(5);

    if (anchor.getOutput() != outputArg) {
      return failure();
    }

    // Structural chain from the anchor.

    if (anchor.getValues().size() != 2) {
      return failure();
    }

    if (!checkConstantInt(anchor.getAxis(), -1)) {
      return failure();
    }
    auto reshape = anchor.getValues()[0].getDefiningOp<ReshapeOp>();

    if (!reshape) {
      return failure();
    }
    auto packShape = reshape.getTargetShape().getDefiningOp<PackOp>();

    if (!packShape || packShape.getInputs().size() != 2) {
      return failure();
    }

    if (packShape.getAxis() != 0) {
      return failure();
    }

    if (!checkConstantInt(packShape.getInputs()[1], -1)) {
      return failure();
    }
    auto outer = reshape.getInput().getDefiningOp<GatherOp>();

    if (!outer || !checkConstantInt(outer.getAxis(), 0)) {
      return failure();
    }
    auto inner = outer.getParams().getDefiningOp<GatherOp>();

    if (!inner || !checkConstantInt(inner.getAxis(), 0)) {
      return failure();
    }

    auto fill = anchor.getValues()[1].getDefiningOp<FillOp>();

    if (!fill) {
      return failure();
    }
    // The fused kernel hard-codes zero padding; the Fill value must be 0
    // (f32 zero in the TF graph).

    if (!checkConstantZero(fill.getValueInput())) {
      return failure();
    }
    auto packDims = fill.getShapeInput().getDefiningOp<PackOp>();

    if (!packDims || packDims.getInputs().size() != 2) {
      return failure();
    }

    if (packDims.getAxis() != 0) {
      return failure();
    }
    // The Reshape shape Pack's first input must be the same pack_size value
    // the kernel receives (packDims input 0).  812's rewriter does not check
    // this; a mismatch would make the reshape shape and the kernel's
    // pack_size disagree, silently corrupting semantics.

    if (packShape.getInputs()[0] != packDims.getInputs()[0]) {
      return failure();
    }

    // Boundary contract: the subgraph must be rooted at the func args.
    if (inner.getIndices() != indices1Arg || inner.getParams() != paramsArg ||
        outer.getIndices() != indices2Arg ||
        packDims.getInputs()[0] != packDimArg ||
        packDims.getInputs()[1] != packArg) {
      return failure();
    }

    // dtype contract: params float32; indices1/indices2 为 int32/int64 的
    // 4 种组合之一 (与一级一致, registry 按 T1/T2 约束选 kernel); indices
    // ranks 1 or 2.
    auto i1Type = dyn_cast<atir::TensorType>(indices1Arg.getType());
    auto i2Type = dyn_cast<atir::TensorType>(indices2Arg.getType());
    auto pType = dyn_cast<atir::TensorType>(paramsArg.getType());
    if (!i1Type || !(i1Type.getElementType().isInteger(32) ||
                     i1Type.getElementType().isInteger(64)))
      return failure();
    if (!i2Type || !(i2Type.getElementType().isInteger(32) ||
                     i2Type.getElementType().isInteger(64)))
      return failure();
    if (!pType || !pType.getElementType().isF32()) return failure();
    if (i1Type.getShape().size() < 1 || i1Type.getShape().size() > 2)
      return failure();
    if (i2Type.getShape().size() < 1 || i2Type.getShape().size() > 2)
      return failure();
    if (pType.getShape().size() != 2) return failure();
    auto pdType = dyn_cast<atir::TensorType>(packDimArg.getType());
    auto pkType = dyn_cast<atir::TensorType>(packArg.getType());
    if (!pdType || !pdType.getElementType().isInteger(32) ||
        pdType.getShape().size() != 0)
      return failure();
    if (!pkType || !pkType.getElementType().isInteger(32) ||
        pkType.getShape().size() != 0)
      return failure();

    // Single-output contract: intermediate results may not escape.
    for (Operation *user : inner.getResult().getUsers()) {
      if (user != outer.getOperation()) return failure();
    }
    for (Operation *user : outer.getResult().getUsers()) {
      if (user != reshape.getOperation()) return failure();
    }
    for (Operation *user : reshape.getResult().getUsers()) {
      if (user != anchor.getOperation()) return failure();
    }
    // Deliberately asymmetric with the first level, which relaxes the
    // packShape/fill checks so sister anchors can share them in the main
    // graph.  Here the checks stay: the first level clones its kernel ops
    // into a fresh func, so every clone is single-user and the checks are
    // vacuous for anything this pass sees in practice.  Should a malformed
    // kernel func reach us anyway, CustomFusionPatternBase would turn the
    // escaping value into an extra CustomizeOp result that the schema does
    // not declare, so keeping them is a cheap guard, not a match restriction.
    for (Operation *user : packShape.getResult().getUsers()) {
      if (user != reshape.getOperation()) return failure();
    }
    for (Operation *user : fill.getResult().getUsers()) {
      if (user != anchor.getOperation()) return failure();
    }
    for (Operation *user : packDims.getResult().getUsers()) {
      if (user != fill.getOperation()) return failure();
    }

    // Collect the subgraph in post-order, stopping at the func arguments.
    SmallVector<Value, 6> boundaryValues = {indices1Arg, paramsArg,
                                            indices2Arg, packDimArg,
                                            packArg,    outputArg};
    SmallPtrSet<Operation *, 32> visited;
    collectDefiningOpsPostOrder(anchor.getOperation(), boundaryValues, visited,
                                fusedOps);
    if (fusedOps.empty()) {
      return failure();
    }
    if (!llvm::is_contained(fusedOps, reshape.getOperation()) ||
        !llvm::is_contained(fusedOps, packShape.getOperation()) ||
        !llvm::is_contained(fusedOps, outer.getOperation()) ||
        !llvm::is_contained(fusedOps, inner.getOperation()) ||
        !llvm::is_contained(fusedOps, fill.getOperation()) ||
        !llvm::is_contained(fusedOps, packDims.getOperation()) ||
        !llvm::is_contained(fusedOps, anchor.getOperation())) {
      return failure();
    }

    return success();
  }

  std::string getCustomOpName(
      ConcatV2Op anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    // 多名字方案: 每 (R1,R2) 特化一个独立 op 名, 由 B 侧 registry 按
    // op_type 精确匹配(不依赖 attrs 通道——B 的 registry 无 attr 维度)。
    auto func = anchor->template getParentOfType<func::FuncOp>();
    auto getRank = [](Type type) -> int64_t {
      if (auto tensorType = dyn_cast<atir::TensorType>(type))
        return static_cast<int64_t>(tensorType.getShape().size());
      return 0;
    };
    int64_t rank1 = getRank(func.getArgument(0).getType());
    int64_t rank2 = getRank(func.getArgument(2).getType());
    return "KPFusedEmbeddingActionIdGatherR" + std::to_string(rank1) + "x" +
           std::to_string(rank2);
  }

  CustomOpSchema getCustomOpSchema(
      ConcatV2Op anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    Value indices1Arg = func.getArgument(0);
    Value paramsArg = func.getArgument(1);
    Value indices2Arg = func.getArgument(2);

    auto getRank = [](Type type) -> int64_t {
      if (auto tensorType = dyn_cast<atir::TensorType>(type))
        return static_cast<int64_t>(tensorType.getShape().size());
      return 0;
    };
    int64_t rank1 = getRank(indices1Arg.getType());
    int64_t rank2 = getRank(indices2Arg.getType());

    // Arg order matches the base class's collected inputValues:
    //   [output buffer, real inputs in fusedOps order] =
    //   [output, params, indices1, indices2, pack_dim, pack]
    // (params precedes indices1: the inner Gather's operand order is
    // (buffer, params, indices, axis), scanned before indices2/pack dims).
    // R1/R2 specialize the kernel on the indices ranks (matched exactly via
    // the registry attr mechanism).
    // Schema name must match getCustomOpName: the CustomizeOp metadata
    // records custom.op_name from here, and lowering reads opType from
    // getCustomOpName — the two must agree for the multi-name scheme.
    return CustomOpSchema::get(getCustomOpName(anchor, fusedOps))
        .TypeVar("T")
        .TypeVar("T1")
        .TypeVar("T2")
        .MemRefArg("output", 2, "T")
        .MemRefArg("params", 2, "T")
        .MemRefArg("indices1", rank1, "T1")
        .MemRefArg("indices2", rank2, "T2")
        .MemRefArg("pack_dim", 0, "")
        .MemRefArg("pack", 0, "")
        .I64AttrArg("R1", rank1)
        .I64AttrArg("R2", rank2);
  }

 private:
  // === Constant helpers (same helpers as the other 812 patterns) ===

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

  static bool checkConstantZero(Value v) {
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

  static void collectDefiningOpsPostOrder(Operation *op,
                                          ArrayRef<Value> boundaryValues,
                                          SmallPtrSetImpl<Operation *> &visited,
                                          SmallVectorImpl<Operation *> &ops) {
    if (!op || visited.count(op)) return;
    if (isa<VariableOp>(op)) return;
    if (isa<BufferOp>(op)) return;
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

REGISTER_CUSTOM_PATTERN(KpEmbeddingActionIdGatherRewrite);
