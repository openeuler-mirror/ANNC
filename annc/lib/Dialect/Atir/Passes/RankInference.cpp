// Rank inference for rank_known=false ATIR tensors.
//
// Serving-exported graphs often lose `_output_shapes`: either an output entry
// is unknown_rank, or a custom op carries no shape facts at all.  The builder
// faithfully turns those into `atir.tensor<*xT, rank_known = false>`, and
// every KP fusion pattern guards on exact ranks via getShape().size(), so the
// candidates die at the rank check.
//
// This pass restores ranks from two kinds of hard TF guarantees, in one
// proposal/commit fixed point:
//   1. op-semantics whitelist (Unique y/idx rank 1, SparseSegmentSum
//      data/result share one rank + indices=1/segmentIds=1/numSegments=0,
//      Shape/Size/Rank outputs, ...);
//   2. consumer-side constraints read backwards (the StridedSlice rank
//      equation result = input + popcount(newAxisMask) -
//      popcount(shrinkAxisMask) propagates both ways, MatMul pins lhs/rhs to
//      rank 2, ...).
// It also pins back `!atir.unknown` element types on custom-op outputs via
// the same mechanism (StridedSlice/Unique/Gather/ConcatV2/Pack dtype
// contracts).
//
// Safety rules:
//   - only values with rank_known=false are upgraded; serving facts are never
//     overwritten, dims are never guessed (committed dims are kDynamic unless
//     an equality class copies a guaranteed-identical shape);
//   - conflicting constraints (>=2 incompatible shapes) leave the value
//     unknown forever - conservative, no fusion rather than a wrong one;
//   - ops with SameOperandsAndResultShape (Cast, Sub, ...) get their whole
//     operand/result class unified in the same commit, because the trait
//     verifier rejects mixed known/unknown ranks;
//   - inferShape() is deliberately NOT called: UniqueOp::inferShape rebuilds
//     result types with the 2-arg builder and would drop name/encoding/
//     rankKnown descriptor fields.
//
// Run before atir-op-fusion (and before atir-identity-canonicalize, whose
// elimination checks input/result shape equality and thus benefits from
// unified ranks).

#include <optional>

#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace atir {
namespace {

// Defensive iteration bound; convergence is guaranteed (the known-rank set is
// monotone and finite) so this only guards against a rule bug.
constexpr int kMaxIterations = 1000;

// A proposed shape for a value.  Rank-only proposals carry kDynamic dims;
// equality classes copy the known member's actual dims (guaranteed identical
// by the op contract, not guessed).
using Shape = SmallVector<int64_t, 4>;

bool shapesCompatible(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i] && lhs[i] != ShapedType::kDynamic &&
        rhs[i] != ShapedType::kDynamic)
      return false;
  }
  return true;
}

TensorType rankedLike(TensorType type, ArrayRef<int64_t> shape) {
  Builder builder(type.getContext());
  return TensorType::get(
      type.getContext(), shape, type.getElementType(), type.getName(),
      type.getEncoding(), type.getStride(), type.getLayout(), type.getMemType(),
      type.getAddress(), type.getDeviceParallel(), type.getOnchipParallel(),
      type.getCacheData(), builder.getBoolAttr(true));
}

TensorType withElemType(TensorType type, Type elemType) {
  return TensorType::get(
      type.getContext(), type.getShape(), elemType, type.getName(),
      type.getEncoding(), type.getStride(), type.getLayout(), type.getMemType(),
      type.getAddress(), type.getDeviceParallel(), type.getOnchipParallel(),
      type.getCacheData(), type.getRankKnown());
}

int32_t popcountMask(int32_t value) {
  int32_t count = 0;
  while (value != 0) {
    value &= value - 1;
    ++count;
  }
  return count;
}

std::optional<int64_t> knownRankOf(Value value) {
  auto type = dyn_cast<TensorType>(value.getType());
  if (!type || !type.hasKnownRank()) return std::nullopt;
  return static_cast<int64_t>(type.getShape().size());
}

// Constant evidence: the length of a 1-D spec tensor (begin, targetShape,
// shapeInput).  Prefers the static type dim, falls back to the cached
// DenseElementsAttr element count when the dim itself is dynamic.
std::optional<int64_t> constVectorLengthOf(Value value) {
  auto type = dyn_cast<TensorType>(value.getType());
  if (!type || !type.hasKnownRank()) return std::nullopt;
  ArrayRef<int64_t> shape = type.getShape();
  if (shape.size() != 1) return std::nullopt;
  if (shape[0] != ShapedType::kDynamic) return shape[0];
  if (auto data = type.getCacheData())
    return static_cast<int64_t>(data.getNumElements());
  return std::nullopt;
}

struct RankSolver {
  llvm::DenseMap<Value, SmallVector<Shape, 2>> proposals;
  llvm::DenseSet<Value> conflicted;
  // Element-type inference: custom-op outputs carry `!atir.unknown` element
  // types.  Ops whose TF contract fixes the dtype for several operands
  // (StridedSlice T, Unique T, Gather/ConcatV2/Pack) pin them back.
  llvm::DenseMap<Value, SmallVector<Type, 2>> elemProposals;
  llvm::DenseSet<Value> conflictedElem;

  std::optional<Type> concreteElemOf(Value value) {
    auto type = dyn_cast<TensorType>(value.getType());
    if (!type) return std::nullopt;
    Type elem = type.getElementType();
    if (isa<UnknownType>(elem)) return std::nullopt;
    return elem;
  }

  void proposeElem(Value value, Type elem) {
    auto type = dyn_cast<TensorType>(value.getType());
    if (!type || !isa<UnknownType>(type.getElementType()) ||
        conflictedElem.count(value))
      return;
    elemProposals[value].push_back(elem);
  }

  void proposeElemEquality(ArrayRef<Value> members) {
    std::optional<Type> elem;
    for (Value member : members) {
      if (auto concrete = concreteElemOf(member)) {
        elem = concrete;
        break;
      }
    }
    if (!elem) return;
    for (Value member : members) proposeElem(member, *elem);
  }

  void propose(Value value, ArrayRef<int64_t> shape) {
    auto type = dyn_cast<TensorType>(value.getType());
    if (!type || type.hasKnownRank() || conflicted.count(value)) return;
    proposals[value].emplace_back(shape.begin(), shape.end());
  }

  void proposeRank(Value value, int64_t rank) {
    propose(value, Shape(static_cast<size_t>(rank), ShapedType::kDynamic));
  }

  // Every member shares one shape by op contract (same-shape trait family or
  // the builder's output-buffer invariant).  Any known member pins the rest.
  void proposeEquality(ArrayRef<Value> members) {
    std::optional<int64_t> rank;
    for (Value member : members) {
      if (auto known = knownRankOf(member)) {
        rank = known;
        break;
      }
    }
    if (!rank) return;
    for (Value member : members) proposeRank(member, *rank);
  }

  // The builder creates operand 0 of every compute op as an atir.buffer
  // result with the op's output type; downstream passes rely on buffer and
  // result keeping identical shapes.  Discriminate on the defining BufferOp,
  // not on type equality - after a commit the two types can diverge and a
  // type-equality check would miss the class in later iterations.
  void proposeBufferClass(Operation *op) {
    if (op->getNumOperands() < 1 || op->getNumResults() < 1) return;
    Value buffer = op->getOperand(0);
    if (!buffer.getDefiningOp<BufferOp>()) return;
    proposeEquality({buffer, op->getResult(0)});
  }

  void collect(Operation *op) {
    proposeBufferClass(op);

    // Same-shape family: all tensor operands + result share one shape.
    // Cast/Identity are the KP-relevant members; the rest carry the
    // SameOperandsAndResultShape trait, whose verifier rejects a class left
    // partially upgraded.
    if (isa<CastOp, IdentityOp, SubOp, FloorModOp, FloorDivOp, RsqrtOp, AbsOp,
            LogisticOp, ZerosLikeOp, StringToNumberOp, StaticRegexReplaceOp,
            StringToHashBucketFastOp>(op)) {
      SmallVector<Value, 4> members;
      for (Value operand : op->getOperands()) members.push_back(operand);
      for (Value result : op->getResults()) members.push_back(result);
      proposeEquality(members);
      return;
    }

    if (auto unique = dyn_cast<UniqueOp>(op)) {
      // TF's Unique shape fn asserts WithRank(input, 1), so x, y and idx are
      // all rank 1 unconditionally.
      proposeRank(unique.getX(), 1);
      proposeRank(unique.getY(), 1);
      proposeRank(unique.getIdx(), 1);
      // x and y share the T dtype.
      proposeElemEquality({unique.getX(), unique.getY()});
    } else if (auto matmul = dyn_cast<MatMulOp>(op)) {
      proposeRank(matmul.getLhs(), 2);
      proposeRank(matmul.getRhs(), 2);
      proposeRank(matmul.getResult(), 2);
    } else if (auto reduce = dyn_cast<SparseSegmentSumOp>(op)) {
      // TF SparseSegmentSum accepts N-D data (rank 1, 2 and 3 all verified
      // legal on TF 2.20) and the result keeps the data rank exactly, so
      // data and result only share one rank; only the spec operands are
      // fixed-rank.  SparseSegmentSum names its data operand `input`
      // (Mean uses `data`).
      proposeEquality({reduce.getInput(), reduce.getResult()});
      proposeRank(reduce.getIndices(), 1);
      proposeRank(reduce.getSegmentIds(), 1);
      proposeRank(reduce.getNumSegments(), 0);
      // Output element type matches data.
      proposeElemEquality({reduce.getInput(), reduce.getResult()});
    } else if (auto reduce = dyn_cast<SparseSegmentMeanOp>(op)) {
      proposeEquality({reduce.getData(), reduce.getResult()});
      proposeRank(reduce.getIndices(), 1);
      proposeRank(reduce.getSegmentIds(), 1);
      proposeRank(reduce.getNumSegments(), 0);
      proposeElemEquality({reduce.getData(), reduce.getResult()});
    } else if (auto concat = dyn_cast<ConcatV2Op>(op)) {
      std::optional<int64_t> rank;
      for (Value value : concat.getValues()) {
        if (auto known = knownRankOf(value)) {
          rank = known;
          break;
        }
      }
      if (!rank) rank = knownRankOf(concat.getResult());
      if (rank) {
        for (Value value : concat.getValues()) proposeRank(value, *rank);
        proposeRank(concat.getResult(), *rank);
      }
      proposeRank(concat.getAxis(), 0);
      SmallVector<Value, 4> members;
      members.push_back(concat.getResult());
      members.append(concat.getValues().begin(), concat.getValues().end());
      proposeElemEquality(members);
    } else if (auto pack = dyn_cast<PackOp>(op)) {
      // Inputs share one rank r and the result is r+1 (the new axis).  The
      // output operand is the builder's buffer for the RESULT (rank r+1), so
      // it must not join the input group - it is covered by the generic
      // output-buffer class instead.
      std::optional<int64_t> rank;
      for (Value input : pack.getInputs()) {
        if (auto known = knownRankOf(input)) {
          rank = known;
          break;
        }
      }
      if (!rank) {
        if (auto resultRank = knownRankOf(pack.getResult())) {
          if (*resultRank >= 1) rank = *resultRank - 1;
        }
      }
      if (rank) {
        for (Value input : pack.getInputs()) proposeRank(input, *rank);
        proposeRank(pack.getResult(), *rank + 1);
      }
      SmallVector<Value, 4> packMembers;
      packMembers.push_back(pack.getResult());
      packMembers.append(pack.getInputs().begin(), pack.getInputs().end());
      proposeElemEquality(packMembers);
    } else if (auto expand = dyn_cast<ExpandDimsOp>(op)) {
      if (auto rank = knownRankOf(expand.getInput()))
        proposeRank(expand.getResult(), *rank + 1);
      else if (auto rank = knownRankOf(expand.getResult())) {
        if (*rank >= 1) proposeRank(expand.getInput(), *rank - 1);
      }
      // TF ExpandDims accepts a 0-D axis or a 1-D single-element axis
      // (verified on TF 2.20), so the axis rank is deliberately not pinned.
    } else if (auto topk = dyn_cast<TopKOp>(op)) {
      std::optional<int64_t> rank = knownRankOf(topk.getInput());
      if (!rank) rank = knownRankOf(topk.getValues());
      if (!rank) rank = knownRankOf(topk.getIndices());
      if (rank) {
        proposeRank(topk.getInput(), *rank);
        proposeRank(topk.getValues(), *rank);
        proposeRank(topk.getIndices(), *rank);
      }
      proposeRank(topk.getK(), 0);
    } else if (auto gather = dyn_cast<GatherOp>(op)) {
      // TF GatherV2: out = params + indices - 1 - batch_dims (only the
      // batch_dims == 0 form is used by the KP patterns).
      if (gather.getBatchDims() == 0) {
        auto paramsRank = knownRankOf(gather.getParams());
        auto indicesRank = knownRankOf(gather.getIndices());
        auto resultRank = knownRankOf(gather.getResult());
        if (paramsRank && indicesRank)
          proposeRank(gather.getResult(), *paramsRank + *indicesRank - 1);
        if (paramsRank && resultRank)
          proposeRank(gather.getIndices(), *resultRank - *paramsRank + 1);
        if (indicesRank && resultRank)
          proposeRank(gather.getParams(), *resultRank - *indicesRank + 1);
      }
      proposeRank(gather.getAxis(), 0);
      // Gather output shares the params dtype.
      proposeElemEquality({gather.getParams(), gather.getResult()});
    } else if (auto reshape = dyn_cast<ReshapeOp>(op)) {
      proposeRank(reshape.getTargetShape(), 1);
      if (auto length = constVectorLengthOf(reshape.getTargetShape()))
        proposeRank(reshape.getResult(), *length);
    } else if (auto fill = dyn_cast<FillOp>(op)) {
      proposeRank(fill.getShapeInput(), 1);
      proposeRank(fill.getValueInput(), 0);
      if (auto length = constVectorLengthOf(fill.getShapeInput()))
        proposeRank(fill.getResult(), *length);
    } else if (auto range = dyn_cast<RangeOp>(op)) {
      proposeRank(range.getStart(), 0);
      proposeRank(range.getLimit(), 0);
      proposeRank(range.getDelta(), 0);
      proposeRank(range.getResult(), 1);
    } else if (auto shape = dyn_cast<ShapeOp>(op)) {
      proposeRank(shape.getOutput(), 1);
    } else if (auto rankOp = dyn_cast<RankOp>(op)) {
      proposeRank(rankOp.getOutput(), 0);
    } else if (auto size = dyn_cast<SizeOp>(op)) {
      proposeRank(size.getOutput(), 0);
    } else if (auto slice = dyn_cast<StridedSliceOp>(op)) {
      // begin/end/strides are per-dimension spec vectors.
      proposeRank(slice.getBegin(), 1);
      proposeRank(slice.getEnd(), 1);
      proposeRank(slice.getStrides(), 1);
      // TF rank contract: result = input + popcount(newAxisMask) -
      // popcount(shrinkAxisMask), propagated both ways.  The begin vector
      // length is deliberately NOT used as a rank pin: TF implies an ellipsis
      // at the end of the spec, so begin may legally be shorter than the
      // input rank (tf.strided_slice on a rank-3 tensor with begin=[1],
      // end=[5] slices only dim 0 and is valid), and its length is only a
      // lower bound on the input rank.
      int64_t newAxes = popcountMask(slice.getNewAxisMask());
      int64_t shrinkAxes = popcountMask(slice.getShrinkAxisMask());
      if (auto rank = knownRankOf(slice.getInput())) {
        proposeRank(slice.getResult(), *rank + newAxes - shrinkAxes);
      } else if (auto rank = knownRankOf(slice.getResult())) {
        int64_t inputRank = *rank - newAxes + shrinkAxes;
        if (inputRank >= 0) proposeRank(slice.getInput(), inputRank);
      }
      // TF StridedSlice T covers input and output.
      proposeElemEquality({slice.getInput(), slice.getResult()});
    } else if (auto transpose = dyn_cast<TransposeOp>(op)) {
      proposeEquality(
          {transpose.getOutput(), transpose.getInput(), transpose.getResult()});
      proposeRank(transpose.getPerm(), 1);
    } else if (auto tile = dyn_cast<TileOp>(op)) {
      proposeEquality({tile.getOutput(), tile.getInput(), tile.getResult()});
      proposeRank(tile.getMultiples(), 1);
    } else if (auto sw = dyn_cast<SwitchOp>(op)) {
      proposeEquality(
          {sw.getData(), sw.getFalseOutput(), sw.getTrueOutput()});
    } else if (auto merge = dyn_cast<MergeOp>(op)) {
      SmallVector<Value> values(merge.getInputsAndControl());
      values.push_back(merge.getOutput());
      proposeEquality(values);
      proposeRank(merge.getValueIndex(), 0);
    } else if (auto enter = dyn_cast<EnterOp>(op)) {
      proposeEquality({enter.getInput(), enter.getOutput()});
    } else if (auto exit = dyn_cast<ExitOp>(op)) {
      proposeEquality({exit.getInput(), exit.getOutput()});
    } else if (auto next = dyn_cast<NextIterationOp>(op)) {
      proposeEquality({next.getInput(), next.getOutput()});
    } else if (auto cond = dyn_cast<LoopCondOp>(op)) {
      proposeEquality({cond.getInput(), cond.getOutput()});
      proposeRank(cond.getInput(), 0);
      proposeRank(cond.getOutput(), 0);
    }
  }

  bool commit() {
    bool changed = false;
    for (auto &[value, candidates] : proposals) {
      auto type = dyn_cast<TensorType>(value.getType());
      if (!type || type.hasKnownRank() || conflicted.count(value)) continue;

      Shape merged;
      bool haveCandidate = false;
      bool consistent = true;
      for (const Shape &candidate : candidates) {
        if (!haveCandidate) {
          merged = candidate;
          haveCandidate = true;
          continue;
        }
        if (!shapesCompatible(merged, candidate)) {
          consistent = false;
          break;
        }
        for (size_t i = 0; i < merged.size(); ++i) {
          if (merged[i] == ShapedType::kDynamic) merged[i] = candidate[i];
        }
      }
      if (!haveCandidate) continue;
      if (!consistent) {
        conflicted.insert(value);
        continue;
      }
      value.setType(rankedLike(type, merged));
      changed = true;
    }
    proposals.clear();

    for (auto &[value, candidates] : elemProposals) {
      auto type = dyn_cast<TensorType>(value.getType());
      if (!type || !isa<UnknownType>(type.getElementType()) ||
          conflictedElem.count(value))
        continue;
      SmallVector<Type, 2> uniqueCandidates;
      for (Type candidate : candidates) {
        if (!llvm::is_contained(uniqueCandidates, candidate))
          uniqueCandidates.push_back(candidate);
      }
      if (uniqueCandidates.size() == 1) {
        value.setType(withElemType(type, uniqueCandidates[0]));
        changed = true;
      } else if (uniqueCandidates.size() >= 2) {
        conflictedElem.insert(value);
      }
    }
    elemProposals.clear();
    return changed;
  }

  void rebuildFuncSignatures(ModuleOp module) {
    module.walk([](func::FuncOp function) {
      if (function.isExternal()) return;
      SmallVector<Type> argumentTypes;
      for (BlockArgument argument : function.getArguments())
        argumentTypes.push_back(argument.getType());
      SmallVector<Type> resultTypes;
      if (auto ret = dyn_cast<func::ReturnOp>(
              function.getBody().front().getTerminator())) {
        resultTypes.assign(ret.getOperandTypes().begin(),
                           ret.getOperandTypes().end());
      } else {
        resultTypes.assign(function.getResultTypes().begin(),
                           function.getResultTypes().end());
      }
      function.setType(
          FunctionType::get(function.getContext(), argumentTypes, resultTypes));
    });
  }
};

class AtirRankInferencePass
    : public AtirRankInferenceBase<AtirRankInferencePass> {
 public:
  void runOnOperation() override {
    RankSolver solver;
    bool changed = true;
    for (int iteration = 0; changed && iteration < kMaxIterations; ++iteration) {
      getOperation().walk(
          [&](Operation *op) { solver.collect(op); });
      changed = solver.commit();
      if (changed) solver.rebuildFuncSignatures(getOperation());
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createAtirRankInferencePass() {
  return std::make_unique<AtirRankInferencePass>();
}

}  // namespace atir
