#include <string>

#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/FusionV2/CarrierMatchers.h"
#include "Dialect/Atir/Passes/FusionV2/FusionV2Outline.h"
#include "Dialect/Atir/Passes/FusionV2/Naming.h"
#include "Dialect/Atir/Passes/FusionV2/Patterns.h"
#include "Dialect/Atir/Passes/FusionV2/Provenance.h"
#include "Support/Log.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace atir::fusionv2;

namespace atir {
namespace {

struct HashBucketMatch {
  atir::StringToHashBucketFastV2Op anchor;
  Value input;
  Value weight;
  Value output;
  SmallVector<Operation *, 40> members;
};

bool isStringTensor(Value v) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  return ty && isa<atir::StringType>(ty.getElementType());
}

bool isResidueOf(Operation *op, llvm::StringRef opType) {
  auto residue = dyn_cast_or_null<atir::UnsupportedOp>(op);
  return residue && residue.getOpType() == opType;
}

// Walks the string-domain chain back to the single string boundary (block
// arg or placeholder residue). Only identity wrappers (Cast residues) may
// sit in between: a real string op would change the hashed value, so it
// invalidates the site instead of being absorbed as a member.
Value findStringInput(Value root, bool &invalid) {
  Value v = root;
  for (int hop = 0; hop < 16; ++hop) {
    Operation *def = v.getDefiningOp();
    if (!def) {
      if (!isStringTensor(v)) break;
      return v;
    }
    if (auto residue = dyn_cast<atir::UnsupportedOp>(def)) {
      if (residue.getOpType() == "Placeholder") {
        if (!isStringTensor(v)) break;
        return v;
      }
      if (residue.getOpType() == "Cast" ||
          residue.getOpType() == "CastV2") {
        v = def->getOperand(0);
        continue;
      }
    }
    break;
  }
  invalid = true;
  return Value();
}

// True when from reaches target through a convert/cast-only chain (the
// Unique#1 -> segment_mean indices contract).
bool reachesThroughCast(Value from, Operation *target) {
  SmallVector<Value, 8> work{from};
  SmallPtrSet<Operation *, 8> seen;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (user == target) return true;
      if (!isa<stablehlo::ConvertOp, arith::IndexCastOp>(user) &&
          !isResidueOf(user, "Cast") && !isResidueOf(user, "CastV2"))
        continue;
      if (seen.insert(user).second) work.push_back(user->getResult(0));
    }
  }
  return false;
}

// Fail-closed match for the dnn_embedding_hash_bucket site: hash anchor ->
// Unique -> gather -> segment mean; output = block-order-last reshape of the
// segment result; string constants join the closure, not the boundary.
LogicalResult matchHashBucket(atir::StringToHashBucketFastV2Op anchor,
                              HashBucketMatch &m) {
  if (anchor.getNumBuckets() == 0) return failure();
  m.anchor = anchor;

  Operation *unique = nullptr;
  for (Operation *user : anchor.getResult().getUsers()) {
    if (!isResidueOf(user, "Unique")) continue;
    if (unique) return failure();
    unique = user;
  }
  if (!unique || unique->getNumResults() != 2) return failure();

  auto isGatherLike = [](Operation *op) {
    return isa_and_nonnull<stablehlo::TorchIndexSelectOp>(op) ||
           isResidueOf(op, "GatherV2") || isResidueOf(op, "Gather");
  };
  Operation *gather = nullptr;
  for (Operation *user : unique->getResult(0).getUsers()) {
    if (!isGatherLike(user)) continue;
    if (gather) return failure();
    gather = user;
  }
  if (!gather) return failure();
  m.weight = gather->getOperand(0);
  auto weightTy = dyn_cast<RankedTensorType>(m.weight.getType());
  if (!weightTy || weightTy.getRank() != 2 ||
      !weightTy.getElementType().isF32())
    return failure();
  // Row-wise lookup only: the kernel indexes weight rows directly, so the
  // gather must select along dim 0 (a bare TF Gather residue is row-wise by
  // definition).
  if (auto view = GatherLike::match(gather)) {
    if (!view.axisIsZero()) return failure();
  } else if (!isResidueOf(gather, "Gather")) {
    return failure();
  }

  Operation *segMean = nullptr;
  for (Operation *user : gather->getResult(0).getUsers()) {
    if (!isa<atir::SparseSegmentMeanV2Op>(user) &&
        !isResidueOf(user, "SparseSegmentMean"))
      continue;
    if (segMean) return failure();
    segMean = user;
  }
  if (!segMean || segMean->getNumResults() != 1) return failure();
  if (!reachesThroughCast(unique->getResult(1), segMean)) return failure();
  // num_segments must stay inferred (0): the kernel derives the output row
  // count itself; an explicit count changes the segment semantics (V1
  // verifyConstants parity). The TF residue form carries no such operand.
  if (auto ssm = dyn_cast<atir::SparseSegmentMeanV2Op>(segMean)) {
    Operation *numDef = ssm.getNumSegments().getDefiningOp();
    auto numConst = dyn_cast_or_null<stablehlo::ConstantOp>(numDef);
    auto numAttr =
        numConst ? dyn_cast<DenseElementsAttr>(numConst.getValue())
                 : DenseElementsAttr();
    if (!numAttr || !numAttr.isSplat() ||
        !numAttr.getElementType().isInteger() ||
        numAttr.getSplatValue<APInt>().getSExtValue() != 0)
      return failure();
  }

  Operation *outputOp = nullptr;
  for (Operation &op : segMean->getParentOfType<func::FuncOp>()
                           .getBody()
                           .front()
                           .getOperations()) {
    auto reshape = fusionv2::ReshapeLike::match(&op);
    if (reshape && reshape.src() == segMean->getResult(0)) outputOp = &op;
  }
  if (!outputOp) return failure();
  m.output = outputOp->getResult(0);
  auto outTy = dyn_cast<RankedTensorType>(m.output.getType());
  if (!outTy || outTy.getRank() != 2 || !outTy.getElementType().isF32())
    return failure();

  bool invalid = false;
  m.input = findStringInput(anchor.getInput(), invalid);
  if (invalid || !m.input) return failure();

  SmallPtrSet<Operation *, 32> seen;
  SmallVector<Value, 32> work{m.output};
  while (!work.empty()) {
    Value v = work.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def) continue;
    if (isa<stablehlo::ConstantOp, arith::ConstantOp>(def)) continue;
    if (auto residue = dyn_cast<atir::UnsupportedOp>(def))
      if (residue.getOpType() == "Placeholder") continue;
    if (def->getName().getStringRef() != "atir.constant_v2" &&
        def->getNumOperands() == 0)
      return failure();
    if (!seen.insert(def).second) continue;
    SmallVector<Value> deps(def->getOperands().begin(),
                            def->getOperands().end());
    def->walk([&](Operation *inner) {
      if (inner != def)
        for (Value o : inner->getOperands()) deps.push_back(o);
    });
    work.append(deps.begin(), deps.end());
    m.members.push_back(def);
  }
  llvm::sort(m.members, [&](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  if (m.members.empty()) return failure();

  SmallPtrSet<Operation *, 32> memberSet(m.members.begin(), m.members.end());
  auto insideSite = [&](Operation *user) {
    for (Operation *anc = user; anc != nullptr; anc = anc->getParentOp())
      if (memberSet.contains(anc)) return true;
    return false;
  };
  for (Operation *op : m.members) {
    for (Value r : op->getResults()) {
      if (r == m.output) continue;
      for (Operation *user : r.getUsers()) {
        if (insideSite(user)) continue;
        if (user->getDialect() && user->getDialect()->getNamespace() == "shape")
          continue;
        return failure();
      }
    }
  }
  return success();
}

// Outlines the site with ABI (input, embedding_weight); num_buckets rides
// the context_i64_attrs slot for the CustomizeCallLowering cross-check.
class FuseDnnEmbeddingHashBucketV2Pattern
    : public OpRewritePattern<atir::StringToHashBucketFastV2Op> {
 public:
  FuseDnnEmbeddingHashBucketV2Pattern(MLIRContext *ctx,
                                      fusionv2::CallPortTable *portTable)
      : OpRewritePattern(ctx), portTable(portTable) {
    assert(portTable && "hash_bucket requires the pass-owned port table");
  }

 private:
  fusionv2::CallPortTable *portTable;

  LogicalResult matchAndRewrite(atir::StringToHashBucketFastV2Op anchor,
                                PatternRewriter &rewriter) const override {
    HashBucketMatch m;
    if (failed(matchHashBucket(anchor, m))) return failure();

    func::FuncOp main = anchor->getParentOfType<func::FuncOp>();
    Operation *tailOps[] = {m.output.getDefiningOp()};
    Value outputs[] = {m.output};
    SmallVector<std::string> tailAnchors;
    tailAnchors.push_back(fusionv2::anchorFromLoc(m.output.getLoc()) + ":0");

    SmallVector<ProbeArg> args;
    if (failed(collectBoundary(main, m.members, args, {}, {}, portTable)))
      return failure();
    if (args.size() != 2) {
      ANNC_LOG_DEBUG("op-fusion-v2")
          << "dnn_embedding_hash_bucket: args=" << args.size()
          << " (want 2); dropped\n";
      return failure();
    }
    SmallVector<ProbeArg> ordered;
    auto pick = [&](Value v) {
      for (auto it = args.begin(); it != args.end(); ++it) {
        if (it->value != v) continue;
        ordered.push_back(*it);
        args.erase(it);
        return true;
      }
      return false;
    };
    if (!pick(m.input) || !pick(m.weight)) return failure();
    args.assign(ordered.begin(), ordered.end());

    ModuleOp module = main->getParentOfType<ModuleOp>();
    std::string name =
        uniquifySymbolName(module, "KPFusedDnnEmbeddingWithHashBucket");
    SmallVector<int64_t, 1> contextAttrs{
        static_cast<int64_t>(anchor.getNumBuckets())};
    if (failed(outlineSite(main, rewriter, args, m.members, tailOps, outputs,
                           name, "dnn_embedding_hash_bucket", tailAnchors,
                           contextAttrs, *portTable)))
      return failure();
    return success();
  }
};

}  // namespace
}  // namespace atir

void atir::addDnnEmbeddingHashBucketV2Pattern(
    RewritePatternSet &patterns, fusionv2::CallPortTable *portTable) {
  patterns.add<FuseDnnEmbeddingHashBucketV2Pattern>(
      patterns.getContext(), portTable);
}
