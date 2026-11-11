#include "common/pb2_common.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <memory>

namespace pb2 {

using mlir::Operation;
using mlir::Value;
using Dims = llvm::SmallVector<int64_t, 4>;

constexpr int kMaxIterations = 100;
// TensorFlow's own tensor rank limit; proposals beyond it are malformed
// (negative ranks would otherwise become SIZE_MAX through size_t casts and
// blow up the Dims allocation).
constexpr int64_t kMaxTensorRank = 253;

static bool isSameShapeFamily(llvm::StringRef tfOp) {
  // All are same-shape ops (output shape == input shape), manually vetted.
  // OpDef shared type_attr alone does NOT imply same shape (Squeeze shares
  // T but drops a rank; caused a ConvertDynamicSqueezeOp segfault,
  // cvr_3 verified).
  return tfOp == "Cast" || tfOp == "Identity" ||
         tfOp == "FloorMod" || tfOp == "FloorDiv" || tfOp == "Rsqrt" ||
         tfOp == "Abs" || tfOp == "Neg" || tfOp == "Exp" || tfOp == "Sigmoid" ||
         tfOp == "Tanh" || tfOp == "Relu" || tfOp == "Square" || tfOp == "Ceil" ||
         tfOp == "ZerosLike" || tfOp == "StringToNumber" ||
         tfOp == "StaticRegexReplace" || tfOp == "StringToHashBucketFast" ||
         tfOp == "Softmax" || tfOp == "LogSoftmax" || tfOp == "Relu6" ||
         tfOp == "Log" || tfOp == "Sqrt" || tfOp == "Sign" || tfOp == "Round" ||
         tfOp == "Sin" || tfOp == "Cos" || tfOp == "Tan";
}

static bool isBroadcastBinary(llvm::StringRef tfOp) {
  // Element-wise binary: TF broadcasting hard-guarantees
  // rank(result) = max(rank(operands)) -- one-way, dims stay dynamic.
  return tfOp == "Add" || tfOp == "AddV2" || tfOp == "Sub" || tfOp == "Mul" ||
         tfOp == "MulNoNan" || tfOp == "Div" || tfOp == "RealDiv" ||
         tfOp == "Minimum" || tfOp == "Maximum" || tfOp == "Equal" ||
         tfOp == "NotEqual" || tfOp == "Greater" || tfOp == "GreaterEqual" ||
         tfOp == "Less" || tfOp == "LessEqual" || tfOp == "LogicalAnd" ||
         tfOp == "LogicalOr" || tfOp == "SquaredDifference";
}

static std::optional<int64_t> knownRankOf(Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t) return std::nullopt;
  return static_cast<int64_t>(t.getRank());
}

static std::optional<int64_t> intAttrOf(Operation *op, llvm::StringRef name) {
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>(name)) return a.getInt();
  return std::nullopt;
}
static int64_t popcount(int64_t v) {
  int64_t c = 0;
  while (v != 0) { v &= v - 1; ++c; }
  return c;
}

// Length of a constant int vector (tf.Const/residue "value"; i32/i64) --
// Reshape/Fill output rank is hidden in the shape vector's values;
static std::optional<int64_t> constVecLen(Value v) {
  Operation* def = v.getDefiningOp();
  if (!def) return std::nullopt;
  auto dense = def->getAttrOfType<mlir::DenseElementsAttr>("value");
  if (!dense || !dense.getElementType().isIntOrIndex()) return std::nullopt;
  return dense.getNumElements();
}

struct RankSolver {
  llvm::DenseMap<Value, llvm::SmallVector<Dims, 2>> proposals;
  llvm::DenseSet<Value> conflicted;

  void proposeRank(Value v, int64_t rank) {
    if (rank < 0 || rank > kMaxTensorRank) return;  // no proposal, no crash
    if (!mlir::isa<mlir::UnrankedTensorType>(v.getType())) return;
    if (conflicted.count(v)) return;
    Dims dims(static_cast<size_t>(rank), mlir::ShapedType::kDynamic);
    proposals[v].emplace_back(dims.begin(), dims.end());
  }

  // Rank-only equivalence for a group (dims NOT copied -- e.g. SparseSegment
  // data/result share rank but data dims are unknown).
  void proposeRankEquality(llvm::ArrayRef<Value> members) {
    std::optional<int64_t> rank;
    for (Value m : members)
      if (auto r = knownRankOf(m)) { rank = r; break; }
    if (!rank) return;
    for (Value m : members) proposeRank(m, *rank);
  }

  // TF broadcasting hard guarantee (one-way): rank(result) = max(known ranks).
  void proposeBroadcast(Operation *op) {
    if (op->getNumResults() == 0) return;
    int64_t best = -1;
    for (Value v : op->getOperands())
      if (auto r = knownRankOf(v)) best = std::max(best, *r);
    if (best >= 0) proposeRank(op->getResult(0), best);
  }

  bool commit() {
    bool changed = false;
    for (auto &[value, candidates] : proposals) {
      auto t = mlir::dyn_cast<mlir::UnrankedTensorType>(value.getType());
      if (!t || conflicted.count(value)) continue;
      Dims merged;
      bool have = false, consistent = true;
      for (const Dims &c : candidates) {
        if (!have) { merged = c; have = true; continue; }
        if (merged.size() != c.size()) { consistent = false; break; }
      }
      if (!have) continue;
      if (!consistent) { conflicted.insert(value); continue; }
      value.setType(mlir::RankedTensorType::get(merged, t.getElementType()));
      changed = true;
    }
    proposals.clear();
    return changed;
  }
};

// TF op rules (operand/result indices follow TF ODS order).
static void collect(Operation *op, RankSolver &s, llvm::StringRef tfOp) {
  unsigned numIn = op->getNumOperands(), numOut = op->getNumResults();
  auto in = [&](unsigned i) { return op->getOperand(i); };
  auto out = [&](unsigned i) { return op->getResult(i); };

  if (isSameShapeFamily(tfOp)) {
    llvm::SmallVector<Value, 4> members;
    for (unsigned i = 0; i < numIn; ++i) members.push_back(in(i));
    for (unsigned i = 0; i < numOut; ++i) members.push_back(out(i));
    s.proposeRankEquality(members);
    return;
  }
  if (isBroadcastBinary(tfOp)) { s.proposeBroadcast(op); return; }

  if (tfOp == "Unique") {  // TF asserts WithRank(x, 1): y/idx rank 1.
    if (numIn < 1 || numOut < 1) return;
    s.proposeRank(in(0), 1);
    s.proposeRank(out(0), 1);
    if (numOut >= 2) s.proposeRank(out(1), 1);
    return;
  }
  if (tfOp == "MatMul") {
    // TF MatMul is strictly rank-2. BatchMatMul(V2) supports rank>=3
    // (batched) and has NO rule here -- a wrong rank beats nothing.
    if (numIn < 2 || numOut < 1) return;
    s.proposeRank(in(0), 2);
    s.proposeRank(in(1), 2);
    s.proposeRank(out(0), 2);
    return;
  }
  if (tfOp == "SparseSegmentSum" || tfOp == "SparseSegmentMean" ||
      tfOp == "SparseSegmentSqrtN") {
    // data ≡ result (rank equation); indices/segment_ids rank 1.
    if (numIn < 1 || numOut < 1) return;
    s.proposeRankEquality({in(0), out(0)});
    if (numIn >= 2) s.proposeRank(in(1), 1);
    if (numIn >= 3) s.proposeRank(in(2), 1);
    return;
  }
  if (tfOp == "ConcatV2") {
    // ConcatV2: values..., axis (last input); all values share one rank.
    if (numOut < 1) return;
    std::optional<int64_t> rank;
    for (unsigned i = 0; i + 1 < numIn; ++i)
      if (auto r = knownRankOf(in(i))) { rank = r; break; }
    if (!rank) rank = knownRankOf(out(0));
    if (rank) {
      for (unsigned i = 0; i + 1 < numIn; ++i) s.proposeRank(in(i), *rank);
      s.proposeRank(out(0), *rank);
    }
    if (numIn >= 1) s.proposeRank(in(numIn - 1), 0);
    return;
  }
  if (tfOp == "Concat") {
    // legacy Concat: concat_dim FIRST (in(0)), then values...
    if (numIn < 1 || numOut < 1) return;
    std::optional<int64_t> rank;
    for (unsigned i = 1; i < numIn; ++i)
      if (auto r = knownRankOf(in(i))) { rank = r; break; }
    if (!rank) rank = knownRankOf(out(0));
    if (rank) {
      for (unsigned i = 1; i < numIn; ++i) s.proposeRank(in(i), *rank);
      s.proposeRank(out(0), *rank);
    }
    s.proposeRank(in(0), 0);
    return;
  }
  if (tfOp == "Pack") {
    // inputs share rank r, result is r+1 (axis inserted).
    if (numOut < 1) return;
    std::optional<int64_t> rank;
    for (unsigned i = 0; i < numIn; ++i)
      if (auto r = knownRankOf(in(i))) { rank = r; break; }
    if (!rank && knownRankOf(out(0)))
      rank = *knownRankOf(out(0)) - 1;
    if (rank && *rank >= 0) {
      for (unsigned i = 0; i < numIn; ++i) s.proposeRank(in(i), *rank);
      s.proposeRank(out(0), *rank + 1);
    }
    return;
  }
  if (tfOp == "ExpandDims") {
    if (numIn < 1 || numOut < 1) return;
    if (auto r = knownRankOf(in(0))) s.proposeRank(out(0), *r + 1);
    else if (auto r = knownRankOf(out(0)))
      if (*r >= 1) s.proposeRank(in(0), *r - 1);
    return;
  }
  if (tfOp == "TopK") {
    if (numIn < 1 || numOut < 1) return;
    if (auto r = knownRankOf(in(0))) {
      s.proposeRank(out(0), *r);
      if (numOut >= 2) s.proposeRank(out(1), *r);
    }
    if (numIn >= 2) s.proposeRank(in(1), 0);
    return;
  }
  if (tfOp == "GatherV2") {
    // batch_dims=0 form ONLY (general: -batch_dims-1); other forms skip.
    auto bd = intAttrOf(op, "batch_dims");
    if (numIn == 3 && bd && *bd == 0) {
      auto p = knownRankOf(in(0));
      auto i = knownRankOf(in(1));
      auto r = knownRankOf(out(0));
      if (p && i) s.proposeRank(out(0), *p + *i - 1);
      if (p && r) s.proposeRank(in(1), *r - *p + 1);
      if (i && r) s.proposeRank(in(0), *r - *i + 1);
      s.proposeRank(in(2), 0);
    }
    return;
  }
  if (tfOp == "Reshape") {
    // Constant shape vector: output rank = vector length (exact, sound).
    // (lands the "needs constant evaluation" note above).
    if (numIn < 2 || numOut < 1) return;
    if (auto cl = constVecLen(in(1))) s.proposeRank(out(0), *cl);
    // shape vector itself is rank 1. Output rank = element COUNT of the
    // shape vector (its dim-0 VALUE), which needs constant evaluation --
    // NOT the vector's own rank. A wrong rank is worse than no rank.
    s.proposeRank(in(1), 1);
    return;
  }
  if (tfOp == "Fill") {
    // dims vector is rank 1; output rank = element count (needs constant
    // evaluation -- same reasoning as Reshape).
    if (numIn < 1 || numOut < 1) return;
    if (auto cl = constVecLen(in(0))) s.proposeRank(out(0), *cl);
    s.proposeRank(in(0), 1);
    return;
  }
  if (tfOp == "Range") {
    if (numOut < 1) return;
    for (unsigned i = 0; i < numIn; ++i) s.proposeRank(in(i), 0);
    s.proposeRank(out(0), 1);
    return;
  }
  if (numOut < 1) return;  // Shape/Rank/Size and every rule below read out(0)
  if (tfOp == "Shape") { s.proposeRank(out(0), 1); return; }
  if (tfOp == "Rank") { s.proposeRank(out(0), 0); return; }
  if (tfOp == "Size") { s.proposeRank(out(0), 0); return; }
  if (tfOp == "StridedSlice") {
    // begin/end/strides rank 1; hard equation propagated BOTH ways:
    // rank(result) = rank(x) + popcount(new_axis) - popcount(shrink_axis).
    // Masks must be present and non-negative: a negative mask is malformed
    // IR and popcount(-1)=64 would fabricate a rank+64 equation.
    if (numIn < 4 || numOut < 1) return;
    s.proposeRank(in(1), 1);
    s.proposeRank(in(2), 1);
    s.proposeRank(in(3), 1);
    auto na = intAttrOf(op, "new_axis_mask");
    auto sa = intAttrOf(op, "shrink_axis_mask");
    if (!na || *na < 0 || !sa || *sa < 0) return;
    if (auto r = knownRankOf(in(0)))
      s.proposeRank(out(0), *r + popcount(*na) - popcount(*sa));
    else if (auto r = knownRankOf(out(0)))
      s.proposeRank(in(0), *r - popcount(*na) + popcount(*sa));
    return;
  }
  if (tfOp == "Transpose" || tfOp == "Tile") {
    if (numIn < 2 || numOut < 1) return;
    s.proposeRankEquality({in(0), out(0)});
    s.proposeRank(in(1), 1);
    return;
  }
  if (tfOp == "Switch") {
    if (numIn < 1 || numOut < 2) return;
    s.proposeRankEquality({in(0), out(0), out(1)});
    return;
  }
  if (tfOp == "Merge") {
    if (numIn < 1 || numOut < 1) return;
    llvm::SmallVector<Value, 4> members;
    for (unsigned i = 0; i < numIn; ++i) members.push_back(in(i));
    members.push_back(out(0));
    s.proposeRankEquality(members);
    if (numOut >= 2) s.proposeRank(out(1), 0);
    return;
  }
  if (tfOp == "Enter" || tfOp == "Exit" || tfOp == "NextIteration") {
    if (numIn < 1 || numOut < 1) return;
    s.proposeRankEquality({in(0), out(0)});
    return;
  }
  if (tfOp == "LoopCond") {
    if (numIn < 1 || numOut < 1) return;
    s.proposeRankEquality({in(0), out(0)});
    s.proposeRank(in(0), 0);
    return;
  }
}
// Every other op: no hard guarantee, no proposal. A missing rank degrades to
// residue / TF-kernel fallback; a wrong rank would not -- prefer missing.

class TfRankInferPass
    : public mlir::PassWrapper<TfRankInferPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TfRankInferPass)
  llvm::StringRef getArgument() const override { return "tf-rank-infer"; }
  void runOnOperation() override {
    RankSolver solver;
    bool changed = true;
    for (int i = 0; changed && i < kMaxIterations; ++i) {
      getOperation()->walk([&](Operation *op) {
        llvm::StringRef name = op->getName().getStringRef();
        if (name == "tf.residue") {
          // KP/TF op hidden in the residue carrier: dispatch by
          // call_target_name (strip the "tf." prefix for the rule table).
          if (auto ct = op->getAttrOfType<mlir::StringAttr>("call_target_name")) {
            auto t = ct.getValue();
            if (t.starts_with("tf."))
              collect(op, solver, t.drop_front(3));
          }
          return;
        }
        if (!name.starts_with("tf.")) return;
        collect(op, solver, name.drop_front(3));
      });
      changed = solver.commit();
    }
  }
};

std::unique_ptr<mlir::Pass> createTfRankInferPass() {
  return std::make_unique<TfRankInferPass>();
}

}  // namespace pb2
