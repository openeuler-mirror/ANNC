#include "Dialect/Atir/Passes/Patterns/CustomFusionPatternBase.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistryMacros.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"

using namespace mlir;
using namespace atir;

// Fusion pattern for the KP-style sparse dynamic stitch (second level).
//
// The first level (OpFusion's FuseKpSparseDynamicStitchAsFuncCallPattern)
// extracts the subgraph into a private func with annc.kernel +
// fusion.metadata, taking (x, v0..vN-1, output).  This pattern rewrites the
// func's body into a single atir.Customize call to the hand-written
// KPFusedSparseDynamicStitch kernel.
//
// Matched subgraph (ATIR ops only):
//
//   x ─► Size ─► Range(0, ·, 1) ─► DynamicPartition(data) ─► indices[0]
//   x ─► FloorMod(x, N) ─► Cast ─► DynamicPartition(partitions)
//   x ─► FloorDiv(x, N) ─► DynamicPartition ─ outputs[i] ─► Gather(axis=0,
//                       params=v_i) ─► data[i]
//   anchor = ParallelDynamicStitch(N indices + N data)
//
// Fused kernel: output[i] = variables[x[i] % N][x[i] / N].
//
// v1 dtype contract: x int64 1-D; variables float32 2-D (equal width).
// N (2..8) becomes a kernel specialization attr; the schema carries N
// variable memref args.
struct KpEmbeddingSparseDynamicStitchRewrite
    : public CustomFusionPatternBase<ParallelDynamicStitchOp> {
  KpEmbeddingSparseDynamicStitchRewrite(
      MLIRContext *context, const CustomOpTypeFilter &customOpFilter,
      PatternBenefit benefit = 8)
      : CustomFusionPatternBase<ParallelDynamicStitchOp>(
            context, customOpFilter, benefit) {}

  mlir::LogicalResult matchFusion(
      ParallelDynamicStitchOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    const int64_t N = anchor.getIndices().size();
    if (N < 2 || N > 16) {
      return failure();
    }
    if (anchor.getData().size() != N) {
      return failure();
    }

    // Only rewrite inside the kernel func extracted by the first-level
    // pattern: private func with annc.kernel, N+2 arguments (x, v0..vN-1,
    // output), no results.
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!func || !func->hasAttr("annc.kernel")) {
      return failure();
    }
    if (func.getNumArguments() != N + 2) {
      return failure();
    }
    if (!func.getFunctionType().getResults().empty()) {
      return failure();
    }
    Value xArg = func.getArgument(0);
    Value outputArg = func.getArgument(N + 1);
    if (anchor.getOutput() != outputArg) {
      return failure();
    }

    // Left branch: indices[0] and all other indices come from the same
    // DynamicPartition (the fused kernel only needs x and the variables, so
    // all index streams must agree with the partition semantics).
    auto partition =
        anchor.getIndices()[0].getDefiningOp<DynamicPartitionOp>();
    if (!partition || partition.getNumPartitions() != N) {
      return failure();
    }
    for (int64_t j = 1; j < N; ++j) {
      if (anchor.getIndices()[j] != partition.getOutputs()[j]) {
        return failure();
      }
    }

    auto range = partition.getData().getDefiningOp<RangeOp>();
    if (!range || !checkConstantInt(range.getDelta(), 1)) {
      return failure();
    }
    if (!checkConstantInt(range.getStart(), 0)) {
      return failure();
    }
    auto sizeOp = range.getLimit().getDefiningOp<SizeOp>();
    if (!sizeOp) {
      return failure();
    }
    Value x = sizeOp.getInput();

    auto cast = partition.getPartitions().getDefiningOp<CastOp>();
    if (!cast) {
      return failure();
    }
    auto floorMod = cast.getInput().getDefiningOp<FloorModOp>();
    if (!floorMod || floorMod.getX() != x) {
      return failure();
    }
    if (!checkConstantInt(floorMod.getY(), N)) {
      return failure();
    }

    // dtype contract: x int64.
    auto xType = dyn_cast<atir::TensorType>(x.getType());
    if (!xType || !xType.getElementType().isInteger(64)) {
      return failure();
    }

    // Right branch: one Gather per partition.
    SmallVector<Value, 8> variables;
    for (int64_t i = 0; i < N; ++i) {
      auto gather = anchor.getData()[i].getDefiningOp<GatherOp>();
      if (!gather || !checkConstantInt(gather.getAxis(), 0)) {
        return failure();
      }
      auto vType = dyn_cast<atir::TensorType>(gather.getParams().getType());
      if (!vType || !vType.getElementType().isF32() ||
          vType.getShape().size() != 2)
        return failure();
      variables.push_back(gather.getParams());
      auto partition1 = gather.getIndices().getDefiningOp<DynamicPartitionOp>();
      if (!partition1 || partition1.getNumPartitions() != N) {
        return failure();
      }
      if (gather.getIndices() != partition1.getOutputs()[i]) {
        return failure();
      }
      auto floorDiv = partition1.getData().getDefiningOp<FloorDivOp>();
      if (!floorDiv || floorDiv.getX() != x) {
        return failure();
      }
      if (!checkConstantInt(floorDiv.getY(), N)) {
        return failure();
      }
    }

    // Boundary contract: the subgraph must be rooted at the func args.
    if (x != xArg) {
      return failure();
    }
    for (int64_t i = 0; i < N; ++i) {
      if (variables[i] != func.getArgument(i + 1)) {
        return failure();
      }
    }

    // Single-output contract: intermediate results may not escape.
    for (Operation *user : range.getResult().getUsers()) {
      if (user != partition.getOperation()) return failure();
    }
    for (Operation *user : cast.getResult().getUsers()) {
      // The cast feeds the partitions of BOTH DynamicPartitions (left and
      // right branches share the same partitions stream).
      if (!isa<DynamicPartitionOp>(user)) return failure();
    }
    for (Operation *user : floorMod.getResult().getUsers()) {
      if (user != cast.getOperation()) return failure();
    }
    for (int64_t j = 0; j < N; ++j) {
      for (Operation *user : partition.getOutputs()[j].getUsers()) {
        if (user != anchor.getOperation()) return failure();
      }
      auto gather = anchor.getData()[j].getDefiningOp<GatherOp>();
      for (Operation *user : gather.getResult().getUsers()) {
        if (user != anchor.getOperation()) return failure();
      }
      auto partition1 =
          gather.getIndices().getDefiningOp<DynamicPartitionOp>();
      for (Operation *user : partition1.getOutputs()[j].getUsers()) {
        if (user != gather.getOperation()) return failure();
      }
      auto floorDiv = partition1.getData().getDefiningOp<FloorDivOp>();
      for (Operation *user : floorDiv.getResult().getUsers()) {
        if (user != partition1.getOperation()) return failure();
      }
    }

    // Collect the subgraph in post-order, stopping at the func arguments.
    SmallVector<Value, 10> boundaryValues;
    boundaryValues.push_back(xArg);
    for (int64_t i = 0; i < N; ++i) {
      boundaryValues.push_back(func.getArgument(i + 1));
    }
    boundaryValues.push_back(outputArg);
    SmallPtrSet<Operation *, 32> visited;
    collectDefiningOpsPostOrder(anchor.getOperation(), boundaryValues, visited,
                                fusedOps);
    if (fusedOps.empty()) {
      return failure();
    }
    if (!llvm::is_contained(fusedOps, partition.getOperation()) ||
        !llvm::is_contained(fusedOps, range.getOperation()) ||
        !llvm::is_contained(fusedOps, sizeOp.getOperation()) ||
        !llvm::is_contained(fusedOps, cast.getOperation()) ||
        !llvm::is_contained(fusedOps, floorMod.getOperation()) ||
        !llvm::is_contained(fusedOps, anchor.getOperation())) {
      return failure();
    }
    for (int64_t i = 0; i < N; ++i) {
      if (!llvm::is_contained(
              fusedOps, anchor.getData()[i].getDefiningOp())) {
        return failure();
      }
    }

    return success();
  }

  std::string getCustomOpName(
      ParallelDynamicStitchOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    // 多名字方案: 每 N 特化一个独立 op 名(同 kp-01 的说明)。
    const int64_t n = anchor.getIndices().size();
    return "KPFusedSparseDynamicStitchN" + std::to_string(n);
  }

  CustomOpSchema getCustomOpSchema(
      ParallelDynamicStitchOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    const int64_t N = anchor.getIndices().size();
    auto func = anchor->template getParentOfType<func::FuncOp>();
    Value xArg = func.getArgument(0);

    auto getRank = [](Type type) -> int64_t {
      if (auto tensorType = dyn_cast<atir::TensorType>(type))
        return static_cast<int64_t>(tensorType.getShape().size());
      return 0;
    };

    // Arg order matches the base class's collected inputValues:
    //   [output buffer, real inputs in fusedOps order] = [output, x, v0..vN-1]
    // N is a kernel specialization attr (exact match via the registry).
    // Schema name must match getCustomOpName (multi-name scheme).
    CustomOpSchema schema =
        CustomOpSchema::get(getCustomOpName(anchor, fusedOps)).TypeVar("T");
    schema.MemRefArg("output", 2, "T");
    schema.MemRefArg("x", getRank(xArg.getType()), "");
    for (int64_t i = 0; i < N; ++i) {
      schema.MemRefArg("variables", 2, "T");
    }
    schema.I64AttrArg("N", N);
    return schema;
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

REGISTER_CUSTOM_PATTERN(KpEmbeddingSparseDynamicStitchRewrite);
