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
// arg or placeholder residue). Identity wrappers may sit in between: Cast
// residues and the string-transport carriers (encode/decode + gather /
// reshape) the odml frontend lowers to_sparse_input's ExpandDims and
// GatherNd into. They preserve the hashed value domain and correspond to
// V1 site members; a real string op would change the hashed value, so it
// invalidates the site instead of being absorbed as a member.
Value findStringInput(Value root, bool &invalid) {
  Value v = root;
  for (int hop = 0; hop < 24; ++hop) {
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
          residue.getOpType() == "CastV2" ||
          residue.getOpType() == "ExpandDims") {
        v = def->getOperand(0);
        continue;
      }
    }
    if (isa<atir::EncodeStringV2Op, atir::DecodeStringV2Op>(def) ||
        isa<stablehlo::DynamicGatherOp, stablehlo::DynamicReshapeOp>(def)) {
      v = def->getOperand(0);
      continue;
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

// Locates the Unique residue downstream of the hash anchor. The odml
// frontend lowers the pre-hash filtering (GreaterEqual compare + Where +
// GatherV2) and SparseFillEmptyRows' feeding chain into stablehlo computes
// and v2 carriers between the anchor and Unique; the walk traverses that
// chain along the data edge only. Fail-closed: an unknown op stops its
// branch, two distinct Uniques or an exhausted budget drop the site.
Operation *findUniqueDownstream(Value hashValue) {
  SmallPtrSet<Value, 16> seen;
  SmallVector<Value, 8> work{hashValue};
  Operation *unique = nullptr;
  int budget = 128;
  while (!work.empty() && budget-- > 0) {
    Value v = work.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (isResidueOf(user, "Unique")) {
        if (unique && unique != user) return nullptr;
        unique = user;
        continue;
      }
      bool traverse = isa<stablehlo::CompareOp, stablehlo::ConvertOp,
                          stablehlo::DynamicBroadcastInDimOp,
                          stablehlo::DynamicGatherOp,
                          stablehlo::TorchIndexSelectOp,
                          arith::IndexCastOp>(user) ||
                      isa<atir::SparseFillEmptyRowsV2Op>(user) ||
                      isResidueOf(user, "Where") || isResidueOf(user, "Reshape");
      if (!traverse) continue;
      for (Value r : user->getResults())
        if (seen.insert(r).second) work.push_back(r);
    }
  }
  return (work.empty() || budget > 0) ? unique : nullptr;
}

// Fail-closed match for the dnn_embedding_hash_bucket site: hash anchor ->
// Unique -> gather -> segment mean; output = block-order-last reshape of the
// segment result; string constants join the closure, not the boundary.
LogicalResult matchHashBucket(atir::StringToHashBucketFastV2Op anchor,
                              HashBucketMatch &m) {
  auto bail = [&](const char *why) {
    ANNC_LOG_DEBUG("op-fusion-v2") << "dnn_embedding_hash_bucket reject: "
                                    << why << "\n";
    return failure();
  };
  if (anchor.getNumBuckets() == 0) return bail("numBuckets=0");
  m.anchor = anchor;

  Operation *unique = findUniqueDownstream(anchor.getResult());
  if (!unique || unique->getNumResults() != 2)
    return bail("no Unique residue downstream");

  auto isGatherLike = [](Operation *op) {
    return isa_and_nonnull<stablehlo::TorchIndexSelectOp>(op) ||
           isResidueOf(op, "GatherV2") || isResidueOf(op, "Gather");
  };
  Operation *gather = nullptr;
  for (Operation *user : unique->getResult(0).getUsers()) {
    if (!isGatherLike(user)) continue;
    if (gather) return bail("multi gather user");
    gather = user;
  }
  if (!gather) return bail("no gather-like user of Unique#0");
  m.weight = gather->getOperand(0);
  auto weightTy = dyn_cast<RankedTensorType>(m.weight.getType());
  if (!weightTy || weightTy.getRank() != 2 ||
      !weightTy.getElementType().isF32())
    return bail("weight not rank2 f32");
  // Row-wise lookup only: the kernel indexes weight rows directly, so the
  // gather must select along dim 0 (a bare TF Gather residue is row-wise by
  // definition).
  if (auto view = GatherLike::match(gather)) {
    if (!view.axisIsZero()) return bail("gather axis!=0");
  } else if (!isResidueOf(gather, "Gather")) {
    return bail("gather not GatherLike/Gather residue");
  }

  Operation *segMean = nullptr;
  for (Operation *user : gather->getResult(0).getUsers()) {
    if (!isa<atir::SparseSegmentMeanV2Op>(user) &&
        !isResidueOf(user, "SparseSegmentMean"))
      continue;
    if (segMean) return failure();
    segMean = user;
  }
  if (!segMean || segMean->getNumResults() != 1)
    return bail("no SparseSegmentMean tail");
  if (!reachesThroughCast(unique->getResult(1), segMean))
    return bail("Unique#1 does not reach segment mean");
  // num_segments must stay inferred (0): the kernel derives the output row
  // count itself; an explicit count changes the segment semantics (V1
  // verifyConstants parity). The TF residue form carries no such operand.
  if (auto ssm = dyn_cast<atir::SparseSegmentMeanV2Op>(segMean)) {
    // The inferred marker is a splat zero constant; the wheel's odml run
    // emits arith constants here, handcrafted tests stablehlo ones.
    Operation *numDef = ssm.getNumSegments().getDefiningOp();
    DenseElementsAttr numAttr;
    if (auto stableConst = dyn_cast_or_null<stablehlo::ConstantOp>(numDef))
      numAttr = dyn_cast<DenseElementsAttr>(stableConst.getValue());
    else if (auto arithConst = dyn_cast_or_null<arith::ConstantOp>(numDef))
      numAttr = dyn_cast<DenseElementsAttr>(arithConst.getValue());
    if (!numAttr || !numAttr.isSplat() ||
        !numAttr.getElementType().isInteger() ||
        numAttr.getSplatValue<APInt>().getSExtValue() != 0)
      return bail("num_segments not inferred-0");
  }

  Operation *outputOp = nullptr;
  for (Operation &op : segMean->getParentOfType<func::FuncOp>()
                           .getBody()
                           .front()
                           .getOperations()) {
    auto reshape = fusionv2::ReshapeLike::match(&op);
    if (reshape && reshape.src() == segMean->getResult(0)) outputOp = &op;
  }
  if (!outputOp) {
    outputOp = segMean;
  }
  m.output = outputOp->getResult(0);
  auto outTy = dyn_cast<RankedTensorType>(m.output.getType());
  if (!outTy || outTy.getRank() != 2 || !outTy.getElementType().isF32())
    return bail("output not rank2 f32");

  bool invalid = false;
  m.input = findStringInput(anchor.getInput(), invalid);
  if (invalid || !m.input) return bail("string input chain invalid");

  SmallPtrSet<Operation *, 32> seen;
  SmallVector<Value, 32> work{m.output};
  // Values defined inside a member's region (e.g. a stablehlo.reduce body)
  // are not members: they relocate with their hosting op; collecting them
  // separately would strand region arguments after the outline move.
  Block *topBlock = &anchor->getParentOfType<func::FuncOp>()
                         .getBody()
                         .front();
  while (!work.empty()) {
    Value v = work.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def) continue;
    if (def->getBlock() != topBlock) continue;
    if (isa<stablehlo::ConstantOp, arith::ConstantOp>(def)) continue;
    if (auto residue = dyn_cast<atir::UnsupportedOp>(def))
      if (residue.getOpType() == "Placeholder") continue;
    if (def->getName().getStringRef() != "atir.constant_v2" &&
        def->getNumOperands() == 0) {
      // Weight reads (variable_v2) are dynamic boundaries handed to the
      // kernel through the ABI, not closure members.
      if (isa<atir::VariableV2Op>(def)) continue;
      return bail("closure hits non-v2 constant");
    }
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
      // SparseFillEmptyRows' auxiliary outputs (empty_row_indicator,
      // reverse_index_map) may feed out-of-site consumers: the V1 kernel
      // leaves them to the host graph, and the outline stay-set clones
      // the carrier for the kernel side.
      if (auto sfer = dyn_cast<atir::SparseFillEmptyRowsV2Op>(op))
        if (r == sfer.getEmptyRowIndicator() ||
            r == sfer.getReverseIndexMap())
          continue;
      // Aux-dialect members (shape/arith/tensor scaffolding) may leak into
      // out-of-site consumers: the outline stay-set clones those for the
      // kernel while the originals keep serving main.
      StringRef memberNs = op->getDialect()
                               ? op->getDialect()->getNamespace()
                               : StringRef();
      bool auxLikeMember =
          memberNs == "shape" || memberNs == "arith" || memberNs == "tensor";
      for (Operation *user : r.getUsers()) {
        if (insideSite(user)) continue;
        if (user->getDialect() && user->getDialect()->getNamespace() == "shape")
          continue;
        if (auxLikeMember) continue;
        return bail("member result leaks outside site");
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
    if (!pick(m.input) || !pick(m.weight)) {
      ANNC_LOG_DEBUG("op-fusion-v2")
          << "dnn_embedding_hash_bucket: boundary args do not cover "
          << "input/weight (args=" << args.size() << "); dropped\n";
      return failure();
    }
    args.assign(ordered.begin(), ordered.end());

    ModuleOp module = main->getParentOfType<ModuleOp>();
    std::string name =
        uniquifySymbolName(module, "KPFusedDnnEmbeddingWithHashBucket");
    SmallVector<int64_t, 1> contextAttrs{
        static_cast<int64_t>(anchor.getNumBuckets())};
    ANNC_LOG_DEBUG("op-fusion-v2")
        << "dnn_embedding_hash_bucket: pre-outline args=" << args.size()
        << " members=" << m.members.size() << "\n";
    if (failed(outlineSite(main, rewriter, args, m.members, tailOps, outputs,
                           name, "dnn_embedding_hash_bucket", tailAnchors,
                           contextAttrs, *portTable))) {
      ANNC_LOG_DEBUG("op-fusion-v2")
          << "dnn_embedding_hash_bucket: outlineSite failed\n";
      return failure();
    }
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
