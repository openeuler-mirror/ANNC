#include <optional>
#include <string>

#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/FusionV2/CarrierMatchers.h"
#include "Dialect/Atir/Passes/FusionV2/FusionV2Outline.h"
#include "Dialect/Atir/Passes/FusionV2/Naming.h"
#include "Dialect/Atir/Passes/FusionV2/Patterns.h"
#include "Support/Log.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
// The atir dialect and the pass live in the top-level ::atir; fusionv2 view
// references go through the alias.
namespace fusionv2 = atir::fusionv2;
using namespace atir::fusionv2;

namespace atir {
namespace {

struct StitchMatch {
  atir::UnsupportedOp stitch;     // the ParallelDynamicStitch residue
  PartitionLike posPartition;     // position DP (range buckets, reorder idx)
  PartitionLike rowNumPartition;  // row-id DP (ids/N buckets, gather idx)
  int64_t n = 0;
  Value x;                               // unique id (kernel contract)
  SmallVector<Value, 16> variables;      // table partitions (kernel contract)
  SmallVector<Operation *, 16> gathers;  // torch_index_select chain
  bool posShared = false;  // position DP has other stitch users
};

static bool matchStructure(atir::UnsupportedOp stitch, StitchMatch &m) {
  const char *why = "";
  auto fail = [&]() {
    ANNC_LOG_DEBUG("op-fusion-v2")
        << "kp_sparse_dynamic_stitch: reject(" << why << ")\n";
    return false;
  };
  if (stitch->getNumResults() != 1) {
    why = "result-count";
    return fail();
  }
  int64_t numInputs = stitch.getNumOperands();
  if (numInputs < 4 || numInputs % 2 != 0) {
    why = "operand-count";
    return fail();
  }
  int64_t n = numInputs / 2;
  if (n < 2 || n > 16) {
    why = "n-out-of-range";
    return fail();
  }
  auto posDP = PartitionLike::match(stitch.getOperand(0).getDefiningOp());
  if (!posDP) {
    why = "posDP";
    return fail();
  }
  if (posDP.count() != n) {
    why = "posDP-results";
    return fail();
  }
  bool posShared = false;
  for (int64_t k = 0; k < n; ++k) {
    if (stitch.getOperand(k) != posDP.buckets[k]) {
      why = "posDP-operand";
      return fail();
    }
    for (Operation *user : posDP.buckets[k].getUsers()) {
      if (user == stitch.getOperation()) continue;
      auto residue = dyn_cast<atir::UnsupportedOp>(user);
      if (!residue || residue.getOpType() != "ParallelDynamicStitch") {
        why = "posDP-user";
        return fail();
      }
      posShared = true;
    }
  }
  m.stitch = stitch;
  m.posPartition = posDP;
  m.n = n;
  m.posShared = posShared;

  PartitionLike rowNumDP;
  for (int64_t k = 0; k < n; ++k) {
    auto gather =
        fusionv2::GatherLike::match(stitch.getOperand(n + k).getDefiningOp());
    if (!gather || !gather.axisIsZero()) {
      why = "gather";
      return fail();
    }
    Value table = gather.params();
    Value idx = gather.indices();
    PartitionLike rn = PartitionLike::match(idx.getDefiningOp());
    if (!rn) {
      why = "rowNumDP";
      return fail();
    }
    if (k == 0) {
      rowNumDP = rn;
      if (rowNumDP.count() != n) {
        why = "rowNumDP-results";
        return fail();
      }
      // Both DPs share the partition ids (same pidx carrier).
      if (rowNumDP.partitions != posDP.partitions) {
        why = "pidx-diverge";
        return fail();
      }
    } else if (idx != rowNumDP.buckets[k]) {
      why = "rowNumDP-bucket";
      return fail();
    }
    // The rowNumDP result is consumed only by this gather; the gather only
    // by the anchor.
    for (Operation *user : idx.getUsers())
      if (user != gather.op) {
        why = "idx-user";
        return fail();
      }
    for (Operation *user : gather.result().getUsers())
      if (user != stitch.getOperation()) {
        why = "gather-user";
        return fail();
      }
    m.variables.push_back(table);
    m.gathers.push_back(gather.op);
  }
  m.rowNumPartition = rowNumDP;
  return true;
}

// divConsts records the constant divisors of stablehlo.divide ops along
// the walk; matchOperands uses them to validate the x/N kernel relation.
static Value traceUniqueIds(Value x, SmallVectorImpl<int64_t> &divConsts) {
  SmallVector<Value, 16> work{x};
  SmallPtrSet<Operation *, 32> seen;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def || !seen.insert(def).second) continue;
    if (auto div = dyn_cast<stablehlo::DivOp>(def))
      if (auto rhs = div.getRhs().getDefiningOp<stablehlo::ConstantOp>())
        if (auto attr = rhs.getValue().dyn_cast<DenseElementsAttr>();
            attr && attr.isSplat() &&
            isa<IntegerType>(attr.getElementType()))
          divConsts.push_back(attr.getSplatValue<APInt>().getSExtValue());
    if (auto residue = dyn_cast<atir::UnsupportedOp>(def))
      if (residue.getOpType() == "Unique" && def->getNumResults() >= 1)
        return def->getResult(0);  // Unique:0 = the sorted unique ids
    if (UniqueLike unique = UniqueLike::match(def)) return unique.values;
    if (auto assuming = dyn_cast<shape::AssumingOp>(def)) {
      for (auto &region : assuming.getRegion())
        for (Operation &op : region)
          for (Value o : op.getOperands())
            if (!isa<BlockArgument>(o)) work.push_back(o);
      continue;
    }
    // Reverse enqueue so pop_back takes operand 0 (main chain first).
    for (Value o : llvm::reverse(def->getOperands())) work.push_back(o);
  }
  return Value();
}

static Value adoptIdView(Value ids) {
  Operation *first = fusionv2::singleDataUser(ids);
  Operation *last = first;
  if (auto getDim = dyn_cast_or_null<atir::GetDimV2Op>(first))
    last = fusionv2::singleDataUser(getDim.getFlat());
  auto restore = dyn_cast_or_null<atir::SetDimV2Op>(last);
  if (!restore) return ids;
  if (restore.getResult().getType() != ids.getType()) return ids;
  if (fusionv2::anchorFromLoc(restore->getLoc()).empty()) return ids;
  return restore.getResult();
}

// Contract checks: x = rank1 i64 (the kernel dtype; an i32 x would mix
// types in the synthesized select chain); variables = rank2 f32 tables
// (dim1 consistent when static); pidx integral.

// The kernel hard-errors on x < 0; the synthesized Rem/Div body truncates
// instead, so the x >= 0 precondition is inherited, not re-checked.
static bool matchOperands(StitchMatch &m) {
  auto fail = [](const char *why) {
    ANNC_LOG_DEBUG("op-fusion-v2")
        << "kp_sparse_dynamic_stitch: reject(match: " << why << ")\n";
    return false;
  };
  SmallVector<int64_t, 4> divConsts;
  Value x = traceUniqueIds(m.rowNumPartition.data, divConsts);
  if (!x) return fail("unique-ids");
  // The kernel computes table = x % N, row = x / N; when the row-id DP's
  // data carries explicit divides, their divisor must be N (a mismatch
  // silently mis-rows). No divides on the direct form - nothing to check.
  if (!divConsts.empty() && !llvm::is_contained(divConsts, m.n))
    return fail("divisor-mismatch");
  x = adoptIdView(x);
  auto xTy = dyn_cast<RankedTensorType>(x.getType());
  if (!xTy || xTy.getRank() != 1) return fail("x-rank");
  auto xElem = dyn_cast<IntegerType>(xTy.getElementType());
  if (!xElem || !xElem.isSignlessInteger(64))
    return fail("x-dtype");
  Value pidx = m.posPartition.partitions;
  auto pidxTy = dyn_cast<RankedTensorType>(pidx.getType());
  if (!pidxTy || !isa<IntegerType>(pidxTy.getElementType()))
    return fail("pidx-dtype");

  std::optional<int64_t> width;
  for (int64_t k = 0; k < m.n; ++k) {
    Value v = m.variables[k];
    auto vTy = dyn_cast<RankedTensorType>(v.getType());
    if (!vTy || !vTy.getElementType().isF32() || vTy.getRank() != 2)
      return fail("table-type");
    // A dynamic width would mix <?x?xf32> and <?xWxf32> gathers and break
    // the synthesized select chain, so require a uniform static width.
    if (vTy.isDynamicDim(1)) return fail("table-width-dynamic");
    if (width && *width != vTy.getDimSize(1)) return fail("table-width");
    width = vTy.getDimSize(1);
  }
  m.x = x;
  return true;
}

class FuseKpSparseDynamicStitchV2Pattern
    : public OpRewritePattern<atir::UnsupportedOp> {
 public:
  FuseKpSparseDynamicStitchV2Pattern(MLIRContext *ctx,
                                     fusionv2::CallPortTable *portTable)
      : OpRewritePattern(ctx), portTable(portTable) {
    assert(portTable && "kp_sparse_dynamic_stitch requires a port table");
  }

  LogicalResult matchAndRewrite(atir::UnsupportedOp anchor,
                                PatternRewriter &rewriter) const override {
    if (anchor.getOpType() != "ParallelDynamicStitch") return failure();
    StitchMatch m;
    if (!matchStructure(anchor, m)) return failure();
    if (!matchOperands(m)) return failure();

    func::FuncOp main = anchor->getParentOfType<func::FuncOp>();
    if (!main) return failure();
    auto module = anchor->getParentOfType<mlir::ModuleOp>();

    // ---- Synthesize the lowerable site body (inserted before the anchor;
    // failure paths roll the synthesized chain back in reverse). ----
    rewriter.setInsertionPoint(anchor.getOperation());
    Location loc = anchor.getLoc();
    SmallVector<Operation *, 40> synthesized;
    auto create = [&](auto &&op) {
      synthesized.push_back(op.getOperation());
      return op;
    };
    // FE-style: broadcast the (1,) splats to x's shape first - a raw (1,)
    // operand would let the type inference collapse the chain to one row.

    Value xi = m.x;
    auto xTy = cast<RankedTensorType>(m.x.getType());
    if (!xTy.getElementType().isInteger(64)) {
      xi = create(
               rewriter.create<stablehlo::ConvertOp>(
                   loc,
                   RankedTensorType::get(xTy.getShape(), rewriter.getI64Type()),
                   m.x))
               .getResult();
    }
    auto i64Splat = [&](int64_t v) -> Value {
      auto ty = RankedTensorType::get({1}, rewriter.getI64Type());
      return create(rewriter.create<stablehlo::ConstantOp>(
                        loc, DenseElementsAttr::get(ty, ArrayRef<int64_t>{v})))
          .getResult();
    };
    Value nSplat = i64Splat(m.n);
    Value zero = i64Splat(0);
    Value xRows =
        create(rewriter.create<stablehlo::GetDimensionSizeOp>(
                   loc, RankedTensorType::get({}, rewriter.getI32Type()), xi,
                   0))
            .getResult();
    Value bcastDims =
        create(rewriter.create<stablehlo::ReshapeOp>(
                   loc, RankedTensorType::get({1}, rewriter.getI32Type()),
                   xRows))
            .getResult();
    auto toXShape = [&](Value v) -> Value {
      return create(rewriter.create<stablehlo::DynamicBroadcastInDimOp>(
                        loc, xTy, v, bcastDims,
                        rewriter.getDenseI64ArrayAttr({0})))
          .getResult();
    };
    nSplat = toXShape(nSplat);
    zero = toXShape(zero);
    Value rem =
        create(rewriter.create<stablehlo::RemOp>(loc, xi, nSplat)).getResult();
    Value rid =
        create(rewriter.create<stablehlo::DivOp>(loc, xi, nSplat)).getResult();

    Value acc;
    for (int64_t k = 0; k < m.n; ++k) {
      Value pred =
          create(rewriter.create<stablehlo::CompareOp>(
                     loc, rem, toXShape(i64Splat(k)),
                     stablehlo::ComparisonDirection::EQ))
              .getResult();
      Value sel =
          create(rewriter.create<stablehlo::SelectOp>(loc, pred, rid, zero))
              .getResult();
      auto vTy = cast<RankedTensorType>(m.variables[k].getType());
      auto gTy = RankedTensorType::get({xTy.getShape()[0], vTy.getDimSize(1)},
                                       rewriter.getF32Type());
      Value gathered = create(rewriter.create<stablehlo::TorchIndexSelectOp>(
                                  loc, gTy, m.variables[k], sel,
                                  /*dim=*/0, /*batch_dims=*/0))
                           .getResult();
      if (k == 0) {
        acc = gathered;
        continue;
      }
      auto pred2d = RankedTensorType::get(gTy.getShape(), rewriter.getI1Type());
      // Broadcast shape from stablehlo itself (no shape-dialect
      // dependency in the synthesized chain).
      Value rows =
          create(rewriter.create<stablehlo::GetDimensionSizeOp>(
                     loc, RankedTensorType::get({}, rewriter.getI32Type()),
                     gathered, 0))
              .getResult();
      Value dims =
          create(
              rewriter.create<stablehlo::ReshapeOp>(
                  loc, RankedTensorType::get({1}, rewriter.getI32Type()), rows))
              .getResult();
      Value predWide =
          create(
              rewriter.create<stablehlo::DynamicBroadcastInDimOp>(
                  loc, pred2d, pred, dims, rewriter.getDenseI64ArrayAttr({0})))
              .getResult();
      acc = create(rewriter.create<stablehlo::SelectOp>(loc, predWide, gathered,
                                                        acc))
                .getResult();
    }

    // ---- Boundary collection with ABI-order validation (failure rolls
    // the synthesized chain back; the main graph is untouched). ----
    SmallVector<ProbeArg> args;
    SmallVector<Value> fixed(m.variables.begin(), m.variables.end());
    auto roll = [&](const char *why) {
      for (Operation *op : llvm::reverse(synthesized)) rewriter.eraseOp(op);
      ANNC_LOG_DEBUG("op-fusion-v2")
          << "kp_sparse_dynamic_stitch: reject(post: " << why << ")\n";
      return failure();
    };
    if (failed(collectBoundary(
            main, synthesized, args,
            {{m.x, fusionv2::abiNameFor(main, m.x, anchorFromLoc(loc), "_x")}},
            fixed)))
      return roll("collectBoundary");
    // ABI order = (x, v_0..v_N-1); x lands last, so reorder by value
    // identity to avoid same-type mis-wiring.
    if (args.size() != static_cast<size_t>(m.n) + 1) {
      for (const ProbeArg &a : args)
        ANNC_LOG_DEBUG("op-fusion-v2")
            << "kp_sparse_dynamic_stitch: arg " << a.tfName << "\n";
      return roll("args-count");
    }
    SmallVector<ProbeArg> ordered;
    auto takeValue = [&](Value want) {
      for (auto it = args.begin(); it != args.end(); ++it)
        if (it->value == want) {
          ordered.push_back(*it);
          args.erase(it);
          return true;
        }
      return false;
    };
    if (!takeValue(m.x)) return roll("x-arg");
    for (int64_t k = 0; k < m.n; ++k)
      if (!takeValue(m.variables[k])) return roll("table-arg");
    args.assign(ordered.begin(), ordered.end());

    // ---- Commit: rebind anchor users to the site output; dead-residue
    // deletion; outline. ----
    if (anchor.getResult(0).getType() != acc.getType())
      return roll("result-type");
    rewriter.replaceAllUsesWith(anchor.getResult(0), acc);
    rewriter.eraseOp(anchor.getOperation());
    auto eraseGroupMembers = [&](const PartitionLike &view) {
      for (auto it = view.members.rbegin(); it != view.members.rend(); ++it) {
        Operation *op = *it;
        if (op->getBlock() == nullptr) continue;
        bool dead = llvm::all_of(op->getResults(),
                                 [](Value r) { return r.use_empty(); });
        if (dead) rewriter.eraseOp(op);
      }
    };
    for (Operation *g : m.gathers) rewriter.eraseOp(g);
    eraseGroupMembers(m.rowNumPartition);
    if (!m.posShared) eraseGroupMembers(m.posPartition);

    Operation *tailOps[] = {synthesized.back()};
    Value outputs[] = {acc};
    // The schema keys on the unified name with N as the sole i64 attribute.
    std::string name = uniquifySymbolName(module, "KPFusedSparseDynamicStitch");
    std::string tailAnchor = fusionv2::anchorFromLoc(loc);
    if (failed(outlineSite(main, rewriter, args, synthesized, tailOps, outputs,
                           name, "kp_sparse_dynamic_stitch", {tailAnchor},
                           {m.n}, *portTable))) {
      // The rewrite so far is semantics-preserving (the synthesized chain
      // replaced the anchor), so keep it and report success instead of
      // violating the driver's failure-means-unchanged contract.
      ANNC_LOG_DEBUG("op-fusion-v2")
          << "kp_sparse_dynamic_stitch: outline failed, site dropped\n";
    }
    return success();
  }

 private:
  fusionv2::CallPortTable *portTable;
};

}  // namespace

void addSparseDynamicStitchV2Pattern(RewritePatternSet &patterns,
                                     fusionv2::CallPortTable *portTable) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<FuseKpSparseDynamicStitchV2Pattern>(ctx, portTable);
}
}  // namespace atir
