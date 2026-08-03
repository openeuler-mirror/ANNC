#include "Dialect/Atir/Passes/Patterns/CustomFusionPatternBase.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistryMacros.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"

using namespace mlir;
using namespace atir;

// Fusion pattern for dnn_embedding_hash_bucket.
//
// Strict structural match from the anchor StringToHashBucketFastOp along
// def-use chains.  The matched subgraph is collected in post-order from the
// sink (the final Reshape that writes the output buffer), so the base class
// receives ops in def-use order and handles input/output collection.
//
// Match shape (ATIR ops only, no TF names):
//
//   arg0 ─► ExpandDims ─►┬─► GatherNd ─► StringToHashBucketFast (anchor)
//                       │                    │
//                       │                    ├─► ... ─► Gather ─►┐
//                       │                    │                    ├─► SparseFillEmptyRows
//                       └─► Where ◄─ Compare │                    │       ─► Unique ─► Gather ◄─ arg1
//                                           │                    │                      │
//                                           └─► SparseReshape ──►┘                      ─► SparseSegmentMean
//                                                                                         │
//                                                                                         ─► ... ─► Reshape(arg2)  (sink)
//
// Kernel ABI (base class collects in this order):
//   [output=arg2, input=arg0, embedding_weight=arg1]
struct DnnEmbeddingHashBucketRewrite
    : public CustomFusionPatternBase<StringToHashBucketFastOp> {
  DnnEmbeddingHashBucketRewrite(MLIRContext *context,
                                PatternBenefit benefit = 8)
      : CustomFusionPatternBase<StringToHashBucketFastOp>(context, benefit) {}

  mlir::LogicalResult matchFusion(
      StringToHashBucketFastOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    auto func = anchor->getParentOfType<func::FuncOp>();
    if (!func || !func->hasAttr("annc.kernel")) {
      return failure();
    }
    if (func.getNumArguments() != 3) {
      return failure();
    }
    if (!func.getFunctionType().getResults().empty()) {
      return failure();
    }

    Value inputArg = func.getArgument(0);    // dynamic string input
    Value weightArg = func.getArgument(1);   // embedding weights
    Value outputArg = func.getArgument(2);   // output buffer

    // Reverse chain: anchor ← GatherNd ← ExpandDims ← inputArg.
    auto gatherNd = anchor.getOperand(1).getDefiningOp<GatherNdOp>();
    if (!gatherNd) {
      return failure();
    }
    auto expandDims =
        gatherNd->getOperand(1).getDefiningOp<ExpandDimsOp>();
    if (!expandDims) {
      return failure();
    }
    if (expandDims->getOperand(1) != inputArg) {
      return failure();
    }

    // Forward: find the sink — the Reshape whose output buffer is outputArg.
    ReshapeOp sink = nullptr;
    for (Operation *user : outputArg.getUsers()) {
      auto reshape = dyn_cast<ReshapeOp>(user);
      if (reshape && reshape->getParentOfType<func::FuncOp>() == func) {
        if (sink) {
          return failure();  // ambiguous: more than one sink
        }
        sink = reshape;
      }
    }
    if (!sink) {
      return failure();
    }

    // Reachability check: the anchor must reach the sink through def-use
    // chains, and the subgraph must contain a SparseSegmentMean (the
    // distinguishing op of this fusion kind).
    if (!reaches(anchor.getOperation(), sink.getOperation())) {
      return failure();
    }
    bool hasSparseSegmentMean = false;
    bool hasUnique = false;
    walkDown(anchor.getOperation(), sink.getOperation(),
             [&](Operation *op) {
               if (isa<SparseSegmentMeanOp>(op)) hasSparseSegmentMean = true;
               if (isa<UniqueOp>(op)) hasUnique = true;
             });
    if (!hasSparseSegmentMean || !hasUnique) {
      return failure();
    }

    // Collect all defining ops from sink in post-order, stopping at the
    // three function arguments.  This yields ops in def-use order, which
    // the base class relies on for input/output collection.
    SmallVector<Value, 3> boundaryValues = {inputArg, weightArg, outputArg};
    SmallPtrSet<Operation *, 32> visited;
    collectDefiningOpsPostOrder(sink.getOperation(), boundaryValues, visited,
                                fusedOps);
    if (fusedOps.empty()) {
      return failure();
    }

    // Sanity: anchor and its reverse chain must all be in the collected set.
    if (!llvm::is_contained(fusedOps, anchor.getOperation())) {
      return failure();
    }
    if (!llvm::is_contained(fusedOps, gatherNd.getOperation())) {
      return failure();
    }
    if (!llvm::is_contained(fusedOps, expandDims.getOperation())) {
      return failure();
    }
    if (!llvm::is_contained(fusedOps, sink.getOperation())) {
      return failure();
    }

    // Verify all constants in the fused subgraph match the kernel's implicit
    // assumptions.  The kernel `KPFusedDnnEmbeddingWithHashBucket` ignores
    // all intermediate ops (Slice/Gather/SparseFillEmptyRows/Unique/
    // SparseSegmentMean/Where/etc) and assumes they simplify to identity
    // transformations.  If any constant deviates from the expected value,
    // the rewrite would be semantically incorrect, so we bail out and let
    // the subgraph fall back to normal lowering.
    if (failed(verifyConstants(fusedOps))) {
      return failure();
    }

    return success();
  }

  std::string getCustomOpName(
      StringToHashBucketFastOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    return "KPFusedDnnEmbeddingWithHashBucket";
  }

  CustomOpSchema getCustomOpSchema(
      StringToHashBucketFastOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    auto numBuckets = anchor.getNumBucketsAttr().getInt();
    auto func = anchor->getParentOfType<func::FuncOp>();
    Value output = func.getArgument(2);
    Value input = func.getArgument(0);
    Value weight = func.getArgument(1);

    auto getRank = [](Type type) -> int64_t {
      if (auto tensorType = dyn_cast<atir::TensorType>(type))
        return static_cast<int64_t>(tensorType.getShape().size());
      return 0;
    };

    // Arg order matches the base class's collected inputValues:
    //   [output buffer, real inputs in fusedOps order]
    // ExpandDims (uses arg0) precedes Gather (uses arg1) in def-use order,
    // so real inputs are [input, weight].
    return CustomOpSchema::get("KPFusedDnnEmbeddingWithHashBucket")
        .TypeVar("T")
        .MemRefArg("output", getRank(output.getType()), "T")
        .MemRefArg("input", getRank(input.getType()), "")
        .MemRefArg("embedding_weight", getRank(weight.getType()), "T")
        .I64AttrArg("num_buckets", numBuckets);
  }

private:
  // === Constant verification helpers ======================================
  //
  // The kernel `KPFusedDnnEmbeddingWithHashBucket` assumes specific constant
  // values in the matched subgraph.  These helpers read ATIR ConstantOp
  // values (stored in the result TensorType's `cacheData`) and compare them
  // against the kernel's expectations.  Any mismatch causes `matchFusion`
  // to fail, so the subgraph falls back to normal lowering.

  // Returns the single integer value of `v` if it is defined by a
  // ConstantOp whose cacheData is a single integer (scalar tensor, 1-element
  // 1-D tensor, or splat tensor of any shape).  Handles i32/i64.  Fails
  // otherwise (non-constant, no cacheData, non-integer, multi-element
  // non-splat).
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

  // Returns the integer values of a 1-D ConstantOp tensor.  Fails for
  // non-constants, missing cacheData, non-integer elements, or tensors
  // whose rank is not 1 (rank-0 tensors are also rejected; use
  // getConstantInt for those).
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

  // Returns true if `v` is a ConstantOp whose integer value equals
  // `expected`.  Handles scalar, 1-element, and splat tensors.
  static bool checkConstantInt(Value v, int64_t expected) {
    FailureOr<int64_t> valOr = getConstantInt(v);
    return succeeded(valOr) && *valOr == expected;
  }

  // Returns true if `v` is a 1-D ConstantOp whose integer values equal
  // `expected` (element-wise, length must match).
  static bool checkConstantInts(Value v, ArrayRef<int64_t> expected) {
    FailureOr<SmallVector<int64_t>> valsOr = getConstantInts(v);
    if (failed(valsOr)) return false;
    if (valsOr->size() != expected.size()) return false;
    for (size_t i = 0; i < expected.size(); ++i) {
      if ((*valsOr)[i] != expected[i]) return false;
    }
    return true;
  }

  // Verifies that every constant feeding a fused op matches the value the
  // kernel implicitly assumes.  Dispatches by op type.  Returns failure on
  // any mismatch or unexpected structure.
  static LogicalResult verifyConstants(ArrayRef<Operation *> fusedOps) {
    for (Operation *op : fusedOps) {
      if (auto cmp = dyn_cast<CompareOp>(op)) {
        StringRef dir = cmp.getComparisonDirection();
        if (dir == "NE") {
          // ExpandDims-output NE compare: rhs must be 0 (string-encoded i32).
          // Kernel hashes all input strings regardless of value.
          if (!checkConstantInt(cmp.getRhs(), 0)) return failure();
        } else if (dir == "GE") {
          // GreaterEqual/y: rhs must be 0 (i64).  hash>=0 is always true.
          if (!checkConstantInt(cmp.getRhs(), 0)) return failure();
        }
        // Other comparison directions: not part of the matched subgraph;
        // if present they would have failed the structural match.
      } else if (auto slice = dyn_cast<SliceOp>(op)) {
        // Slice on the Shape result: begin=[0], size=[1] (take dim 0).
        if (!checkConstantInts(slice.getBegin(), {0})) return failure();
        if (!checkConstantInts(slice.getSize(), {1})) return failure();
      } else if (auto prod = dyn_cast<ProdOp>(op)) {
        // Prod reduction axes: [0].
        if (!checkConstantInts(prod.getIndices(), {0})) return failure();
      } else if (auto gather = dyn_cast<GatherOp>(op)) {
        // All gathers in this subgraph have axis=0.
        if (!checkConstantInt(gather.getAxis(), 0)) return failure();
        // The first GatherV2 (params defined by ShapeOp) takes indices=1
        // to read shape[1].  Other gathers have dynamic indices.
        if (auto paramsOp = gather.getParams().getDefiningOp()) {
          if (isa<ShapeOp>(paramsOp)) {
            if (!checkConstantInt(gather.getIndices(), 1)) return failure();
          }
        }
      } else if (auto sfe = dyn_cast<SparseFillEmptyRowsOp>(op)) {
        // defaultValue (4th operand) must be 0 (i64 scalar).
        if (!checkConstantInt(sfe.getDefaultValue(), 0)) return failure();
      } else if (auto ssm = dyn_cast<SparseSegmentMeanOp>(op)) {
        // numSegments (5th operand) must be 0 (means inferred).
        if (!checkConstantInt(ssm.getNumSegments(), 0)) return failure();
      } else if (auto ss = dyn_cast<StridedSliceOp>(op)) {
        // Two strided slices in the subgraph, distinguished by begin rank.
        auto beginTy = dyn_cast<atir::TensorType>(ss.getBegin().getType());
        if (!beginTy) return failure();
        ArrayRef<int64_t> beginShape = beginTy.getShape();
        int64_t rank = beginShape.empty() ? 0 : beginShape[0];
        if (rank == 2) {
          // embedding_lookup_sparse/strided_slice:
          //   begin=[0,0], end=[0,1], strides=[1,1]
          if (!checkConstantInts(ss.getBegin(), {0, 0})) return failure();
          if (!checkConstantInts(ss.getEnd(), {0, 1})) return failure();
          if (!checkConstantInts(ss.getStrides(), {1, 1})) return failure();
        } else if (rank == 1) {
          // strided_slice on Shape_1:
          //   begin=[1], end=[2], strides=[1]
          if (!checkConstantInts(ss.getBegin(), {1})) return failure();
          if (!checkConstantInts(ss.getEnd(), {2})) return failure();
          if (!checkConstantInts(ss.getStrides(), {1})) return failure();
        } else {
          return failure();  // unexpected strided_slice begin rank
        }
      } else if (auto pack = dyn_cast<PackOp>(op)) {
        // Pack (stack) for the Tile multipliers: among its variadic inputs,
        // the one defined by a ConstantOp must equal 1 (stack/0 = dense<1>).
        // The subgraph also contains the SparseReshape new-shape pack
        // (Cast/x), whose inputs are both dynamic (Prod, Gather); it has no
        // constant to verify and is recognized by feeding a SparseReshapeOp.
        bool found = false;
        for (Value input : pack.getInputs()) {
          if (auto *defOp = input.getDefiningOp()) {
            if (isa<ConstantOp>(defOp)) {
              if (!checkConstantInt(input, 1)) return failure();
              found = true;
              break;
            }
          }
        }
        if (!found) {
          bool feedsSparseReshape =
              llvm::any_of(pack->getUsers(), [](Operation *user) {
                return isa<SparseReshapeOp>(user);
              });
          if (!feedsSparseReshape) return failure();
        }
      }
      // Other op types (ExpandDims, Shape, Where, GatherNd,
      // StringToHashBucketFast, SparseReshape, Unique, Cast, Tile,
      // ZerosLike, Reshape, Buffer) have no constant operands that the
      // kernel relies on, or their constants are covered via the user-op
      // dispatch above (e.g. Where's condition comes from a Compare whose
      // rhs we already verified).
    }
    return success();
  }

  // Post-order collection of defining ops, stopping at boundary values.
  // Constants ARE collected so they get erased together with the fused
  // subgraph (otherwise they'd linger in the func as dead ops).  The base
  // class excludes them from CustomizeOp operands — patterns surface the
  // relevant constant values as metadata attrs via getCustomOpSchema.
  static void collectDefiningOpsPostOrder(
      Operation *op, ArrayRef<Value> boundaryValues,
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

  // Forward reachability: does `from` reach `to` via result users?
  static bool reaches(Operation *from, Operation *to) {
    if (from == to) return true;
    SmallPtrSet<Operation *, 32> visited;
    SmallVector<Operation *, 16> worklist;
    for (Value res : from->getResults()) {
      for (Operation *user : res.getUsers()) {
        if (user == to) return true;
        worklist.push_back(user);
      }
    }
    while (!worklist.empty()) {
      Operation *cur = worklist.pop_back_val();
      if (!visited.insert(cur).second) continue;
      if (cur == to) return true;
      for (Value res : cur->getResults()) {
        for (Operation *user : res.getUsers()) {
          worklist.push_back(user);
        }
      }
    }
    return false;
  }

  // Walk all ops on forward def-use paths from `from` to `to` (inclusive),
  // invoking `callback` on each.
  template <typename Fn>
  static void walkDown(Operation *from, Operation *to, Fn &&callback) {
    SmallPtrSet<Operation *, 32> visited;
    SmallVector<Operation *, 16> worklist;
    callback(from);
    visited.insert(from);
    for (Value res : from->getResults()) {
      for (Operation *user : res.getUsers()) worklist.push_back(user);
    }
    while (!worklist.empty()) {
      Operation *cur = worklist.pop_back_val();
      if (!visited.insert(cur).second) continue;
      callback(cur);
      if (cur == to) continue;  // stop expanding past the sink
      for (Value res : cur->getResults()) {
        for (Operation *user : res.getUsers()) worklist.push_back(user);
      }
    }
  }
};

REGISTER_CUSTOM_PATTERN(DnnEmbeddingHashBucketRewrite);
