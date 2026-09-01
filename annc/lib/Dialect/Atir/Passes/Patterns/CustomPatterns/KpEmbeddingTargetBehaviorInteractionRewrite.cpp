#include "Dialect/Atir/Passes/Patterns/CustomFusionPatternBase.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistryMacros.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"

using namespace mlir;
using namespace atir;

// Fusion pattern for the KP-style target-behavior interaction (second level).
//
// The first level (OpFusion's FuseKpTargetBehaviorInteractionAsFuncCallPattern)
// extracts the subgraph into a private func with annc.kernel +
// fusion.metadata, taking (batch_input, weight, bias, tile_input, output).
// This pattern rewrites that func's body into a single atir.Customize call
// to the hand-written KPFusedTargetBehaviorInteraction kernel.
//
// Matched subgraph (ATIR ops only):
//
//   batch_input ─► BatchMatMul ─► AddV2(add_v2_1) ─► Abs ─►┐
//   weight      ─► B                                bias  ├─► AddV2(add_v2_2)
//   (bias) ────────────────────────────────────────────────┘      │
//                                                          Mul(0.5)│
//   tile_input ─► Tile ─┬─────────────────────────────► ConcatV2(anchor)◄─┘
//                       ├─► Sub(realdiv, tile) ─────────► (values[2])
//                       └─► Mul(realdiv, tile) ─────────► (values[3])
//
// Fused kernel: relu = ReLU(BatchMatMul(batch_input, weight) + bias);
// output = Concat([relu, tile, relu-tile, relu*tile], axis=-1) — [B,M,4*N].
//
// dtype contract: all float32; shapes [B,M,K]/[K,N]/[N]/[B,M,N] (3D/2D/1D/3D).
struct KpEmbeddingTargetBehaviorInteractionRewrite
    : public CustomFusionPatternBase<ConcatV2Op> {
  KpEmbeddingTargetBehaviorInteractionRewrite(
      MLIRContext *context, const CustomOpTypeFilter &customOpFilter,
      PatternBenefit benefit = 8)
      : CustomFusionPatternBase<ConcatV2Op>(context, customOpFilter, benefit) {}

  mlir::LogicalResult matchFusion(
      ConcatV2Op anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    // Only rewrite inside the kernel func extracted by the first-level
    // pattern: private func with annc.kernel, 5 arguments, no results.
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!func || !func->hasAttr("annc.kernel")) {
      return failure();
    }
    if (func.getNumArguments() != 5) {
      return failure();
    }
    if (!func.getFunctionType().getResults().empty()) {
      return failure();
    }
    Value batchInputArg = func.getArgument(0);
    Value weightArg = func.getArgument(1);
    Value biasArg = func.getArgument(2);
    Value tileInputArg = func.getArgument(3);
    Value outputArg = func.getArgument(4);
    if (anchor.getOutput() != outputArg) {
      return failure();
    }

    // Structural chain from the anchor.
    if (anchor.getValues().size() != 4) {
      return failure();
    }
    if (!checkConstantInt(anchor.getAxis(), -1)) {
      return failure();
    }

    auto realdiv = anchor.getValues()[0].getDefiningOp<MulOp>();
    if (!realdiv || !checkConstantFloat(realdiv.getY(), 0.5f)) {
      return failure();
    }
    auto add22 = realdiv.getX().getDefiningOp<AddOp>();
    if (!add22 || add22.getInputs().size() != 2) {
      return failure();
    }
    if (add22.getDoRelu()) {
      return failure();
    }

    AbsOp abs = nullptr;
    AddOp add21 = nullptr;
    if (auto a = add22.getInputs()[0].getDefiningOp<AbsOp>()) {
      abs = a;
      add21 = add22.getInputs()[1].getDefiningOp<AddOp>();
    } else if (auto a = add22.getInputs()[1].getDefiningOp<AbsOp>()) {
      abs = a;
      add21 = add22.getInputs()[0].getDefiningOp<AddOp>();
    }
    if (!abs || !add21 || abs.getInput() != add21.getResult()) {
      return failure();
    }
    if (add21.getInputs().size() != 2 || add21.getDoRelu()) {
      return failure();
    }

    BatchMatMulOp bmm = nullptr;
    Value biasVal;
    if (auto b0 = add21.getInputs()[0].getDefiningOp<BatchMatMulOp>()) {
      bmm = b0;
      biasVal = add21.getInputs()[1];
    } else if (auto b1 = add21.getInputs()[1].getDefiningOp<BatchMatMulOp>()) {
      bmm = b1;
      biasVal = add21.getInputs()[0];
    }
    if (!bmm || bmm.getTransposeA() || bmm.getTransposeB()) {
      return failure();
    }

    auto tileOp = anchor.getValues()[1].getDefiningOp<TileOp>();
    // Tile must be identity ([1,1,1] multiples) — the kernel uses
    // tile_input [B,M,N] directly (812's rewriter does not check this).
    if (!tileOp || !checkConstantInts(tileOp.getMultiples(), {1, 1, 1})) {
      return failure();
    }

    auto sub = anchor.getValues()[2].getDefiningOp<SubOp>();
    if (!sub || sub.getX() != realdiv.getResult()) {
      return failure();
    }
    auto mul = anchor.getValues()[3].getDefiningOp<MulOp>();
    if (!mul || mul.getX() != realdiv.getResult()) {
      return failure();
    }
    if (sub.getY() != tileOp.getResult() || mul.getY() != tileOp.getResult()) {
      return failure();
    }

    // Boundary contract: the subgraph must be rooted at the func args.
    if (bmm.getA() != batchInputArg || bmm.getB() != weightArg ||
        biasVal != biasArg || tileOp.getInput() != tileInputArg) {
      return failure();
    }

    // dtype contract: all float32; shapes 3D/2D/1D/3D.
    auto biType = dyn_cast<atir::TensorType>(batchInputArg.getType());
    auto wType = dyn_cast<atir::TensorType>(weightArg.getType());
    auto bType = dyn_cast<atir::TensorType>(biasArg.getType());
    auto tType = dyn_cast<atir::TensorType>(tileInputArg.getType());
    if (!biType || !biType.getElementType().isF32() ||
        biType.getShape().size() != 3)
      return failure();
    if (!wType || !wType.getElementType().isF32() ||
        wType.getShape().size() != 2)
      return failure();
    if (!bType || !bType.getElementType().isF32() ||
        bType.getShape().size() != 1)
      return failure();
    if (!tType || !tType.getElementType().isF32() ||
        tType.getShape().size() != 3)
      return failure();

    // Single-output contract: intermediate results may not escape.
    for (Operation *user : bmm.getResult().getUsers()) {
      if (user != add21.getOperation()) return failure();
    }
    for (Operation *user : add21.getResult().getUsers()) {
      if (user != abs.getOperation() && user != add22.getOperation())
        return failure();
    }
    for (Operation *user : abs.getResult().getUsers()) {
      if (user != add22.getOperation()) return failure();
    }
    for (Operation *user : add22.getResult().getUsers()) {
      if (user != realdiv.getOperation()) return failure();
    }
    for (Operation *user : realdiv.getResult().getUsers()) {
      if (user != anchor.getOperation() && user != sub.getOperation() &&
          user != mul.getOperation())
        return failure();
    }
    for (Operation *user : tileOp.getResult().getUsers()) {
      if (user != anchor.getOperation() && user != sub.getOperation() &&
          user != mul.getOperation())
        return failure();
    }
    for (Operation *user : sub.getResult().getUsers()) {
      if (user != anchor.getOperation()) return failure();
    }
    for (Operation *user : mul.getResult().getUsers()) {
      if (user != anchor.getOperation()) return failure();
    }

    // Collect the subgraph in post-order, stopping at the func arguments.
    SmallVector<Value, 5> boundaryValues = {batchInputArg, weightArg, biasArg,
                                            tileInputArg, outputArg};
    SmallPtrSet<Operation *, 32> visited;
    collectDefiningOpsPostOrder(anchor.getOperation(), boundaryValues, visited,
                                fusedOps);
    if (fusedOps.empty()) {
      return failure();
    }
    if (!llvm::is_contained(fusedOps, realdiv.getOperation()) ||
        !llvm::is_contained(fusedOps, add22.getOperation()) ||
        !llvm::is_contained(fusedOps, abs.getOperation()) ||
        !llvm::is_contained(fusedOps, add21.getOperation()) ||
        !llvm::is_contained(fusedOps, bmm.getOperation()) ||
        !llvm::is_contained(fusedOps, tileOp.getOperation()) ||
        !llvm::is_contained(fusedOps, sub.getOperation()) ||
        !llvm::is_contained(fusedOps, mul.getOperation()) ||
        !llvm::is_contained(fusedOps, anchor.getOperation())) {
      return failure();
    }

    return success();
  }

  std::string getCustomOpName(
      ConcatV2Op anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    return "KPFusedTargetBehaviorInteraction";
  }

  CustomOpSchema getCustomOpSchema(
      ConcatV2Op anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    auto func = anchor->template getParentOfType<func::FuncOp>();
    Value batchInputArg = func.getArgument(0);
    Value weightArg = func.getArgument(1);
    Value biasArg = func.getArgument(2);
    Value tileInputArg = func.getArgument(3);

    auto getRank = [](Type type) -> int64_t {
      if (auto tensorType = dyn_cast<atir::TensorType>(type))
        return static_cast<int64_t>(tensorType.getShape().size());
      return 0;
    };

    // Arg order matches the base class's collected inputValues:
    //   [output buffer, real inputs in fusedOps order] =
    //   [output, batch_input, weight, bias, tile_input]
    return CustomOpSchema::get("KPFusedTargetBehaviorInteraction")
        .TypeVar("T")
        .MemRefArg("output", 3, "T")
        .MemRefArg("batch_input", getRank(batchInputArg.getType()), "T")
        .MemRefArg("weight", getRank(weightArg.getType()), "T")
        .MemRefArg("bias", getRank(biasArg.getType()), "T")
        .MemRefArg("tile_input", getRank(tileInputArg.getType()), "T");
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

  static bool checkConstantFloat(Value v, float expected) {
    auto constOp = v.getDefiningOp<ConstantOp>();
    if (!constOp) return false;
    auto tensorType = dyn_cast<atir::TensorType>(v.getType());
    if (!tensorType) return false;
    DenseElementsAttr dataAttr = tensorType.getCacheData();
    if (!dataAttr) return false;
    if (!dataAttr.getElementType().isF32()) return false;
    if (dataAttr.isSplat()) return dataAttr.getSplatValue<float>() == expected;
    if (dataAttr.getNumElements() != 1) return false;
    return (*dataAttr.getValues<float>().begin()) == expected;
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

REGISTER_CUSTOM_PATTERN(KpEmbeddingTargetBehaviorInteractionRewrite);
