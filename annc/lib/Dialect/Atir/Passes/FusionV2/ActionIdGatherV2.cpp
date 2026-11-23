#include <optional>
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
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace atir::fusionv2;

namespace atir {
namespace {

struct ActionIdGatherMatch {
  fusionv2::GatherLike g1, g2;
  fusionv2::ReshapeLike reshape;
  fusionv2::ConcatLike concat;
  Value params, indices1, indices2;
  Value packDim;
  std::optional<int64_t> packDimConst;
  Value pack;
  std::optional<int64_t> packConst;
  Value w;
  SmallVector<Operation *, 16> members;
};

// Peels a dynamic_broadcast_in_dim wrapper down to its data operand; null
// stays null (callers feed it optional fields).
Value peelBcast(Value v) {
  if (!v) return v;
  auto bcast = v.getDefiningOp<stablehlo::DynamicBroadcastInDimOp>();
  return bcast ? bcast.getOperand() : v;
}

// One pack_shape/pack_dims field: a reshape(x,[1]) wrapper or a bare
// length-1 rank-1 value (static folding drops the wrapper).
Value packFieldOf(Value v) {
  if (auto reshape = v.getDefiningOp<stablehlo::ReshapeOp>())
    return reshape.getOperand();
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  if (ty && ty.getRank() == 1 && !ty.isDynamicDim(0) && ty.getDimSize(0) == 1)
    return v;
  return Value();
}

// Single-element i32 constant, piercing a reshape wrapper.
std::optional<int64_t> constantInt(Value v) {
  if (auto reshape = v.getDefiningOp<stablehlo::ReshapeOp>())
    return constantInt(reshape.getOperand());
  auto cst = v.getDefiningOp<stablehlo::ConstantOp>();
  if (!cst) return std::nullopt;
  auto attr = dyn_cast<DenseElementsAttr>(cst.getValue());
  auto ty = dyn_cast<RankedTensorType>(cst.getType());
  if (!attr || !ty || !ty.getElementType().isSignlessInteger(32) ||
      attr.getNumElements() != 1)
    return std::nullopt;
  return attr.getValues<APInt>()[0].getSExtValue();
}

// i32 constant vector (the folded Pack([k, -1]) dense form).
std::optional<SmallVector<int64_t, 2>> constantIntVector(Value v) {
  auto cst = v.getDefiningOp<stablehlo::ConstantOp>();
  if (!cst) return std::nullopt;
  auto attr = dyn_cast<DenseElementsAttr>(cst.getValue());
  auto ty = dyn_cast<RankedTensorType>(cst.getType());
  if (!attr || !ty || !ty.getElementType().isSignlessInteger(32))
    return std::nullopt;
  SmallVector<int64_t, 2> values;
  for (const APInt &x : attr.getValues<APInt>())
    values.push_back(x.getSExtValue());
  return values;
}

// Zero-tensor constant, piercing a tensor.cast dynamic-type wrapper.
stablehlo::ConstantOp peelCastToConstant(Value v) {
  if (auto cst = v.getDefiningOp<stablehlo::ConstantOp>()) return cst;
  if (auto cast = v.getDefiningOp<tensor::CastOp>())
    return peelCastToConstant(cast.getSource());
  return nullptr;
}

// pack_dim value: the runtime concat chain or a static shape constant; the
// reshape target must end with -1 in both dynamic forms.
LogicalResult matchPackDim(fusionv2::ReshapeLike reshape, Value &runtime,
                           std::optional<int64_t> &constant) {
  if (Value target = reshape.targetShape()) {
    if (auto cc = target.getDefiningOp<stablehlo::ConcatenateOp>();
        cc && cc.getNumOperands() == 2 && cc.getDimension() == 0) {
      auto tail = constantInt(cc.getOperand(1));
      if (!tail || *tail != -1) return failure();
      runtime = packFieldOf(cc.getOperand(0));
      if (!runtime) return failure();
    } else if (auto vec = constantIntVector(target)) {
      if (vec->size() != 2 || (*vec)[1] != -1) return failure();
      constant = (*vec)[0];
    } else {
      return failure();
    }
  } else {
    auto ty = dyn_cast<RankedTensorType>(reshape.result().getType());
    if (!ty || ty.getRank() != 2 || !ty.hasStaticShape()) return failure();
    constant = ty.getDimSize(0);
  }
  auto resTy = dyn_cast<RankedTensorType>(reshape.result().getType());
  if (constant && resTy && resTy.getRank() == 2 && !resTy.isDynamicDim(0) &&
      resTy.getDimSize(0) != *constant)
    return failure();
  return success();
}

// Fail-closed match for the kp_embedding_action_id_gather site: gather x2 ->
// pack reshape -> concat-with-zeros; pack values arrive runtime or
// static-folded. The five ABI inputs never join the closure.
LogicalResult matchActionIdGather(Operation *anchorOp, ActionIdGatherMatch &m) {
  m.g1 = fusionv2::GatherLike::match(anchorOp);
  if (!m.g1) return failure();
  Value g1v = m.g1.result();
  if (!g1v.hasOneUse()) return failure();
  m.g2 = fusionv2::GatherLike::match(*g1v.getUsers().begin());
  if (!m.g2 || !m.g2.result().hasOneUse()) return failure();

  fusionv2::ReshapeLike reshape =
      fusionv2::ReshapeLike::match(*m.g2.result().getUsers().begin());
  if (!reshape || !reshape.result().hasOneUse()) return failure();

  m.concat = fusionv2::ConcatLike::match(*reshape.result().getUsers().begin());
  if (!m.concat || m.concat.inputs().size() != 2) return failure();
  auto dim = m.concat.dimension();
  auto wTy = dyn_cast<RankedTensorType>(m.concat.result().getType());
  if (!dim || *dim != 1) return failure();
  if (!wTy || !wTy.getElementType().isF32() || wTy.getRank() != 2)
    return failure();

  Value packDim;
  std::optional<int64_t> packDimConst;
  if (failed(matchPackDim(reshape, packDim, packDimConst))) return failure();
  if (packDim) {
    auto ty = dyn_cast<RankedTensorType>(packDim.getType());
    if (!ty || ty.getRank() != 0 || !ty.getElementType().isSignlessInteger(32))
      return failure();
  }

  Value pack;
  std::optional<int64_t> packConst;
  if (auto fill = m.concat.inputs()[1]
                      .getDefiningOp<stablehlo::DynamicBroadcastInDimOp>()) {
    auto fillVal = fill.getOperand().getDefiningOp<stablehlo::ConstantOp>();
    if (!fillVal) return failure();
    auto fillTy = dyn_cast<RankedTensorType>(fillVal.getType());
    auto fillAttr = fillVal.getValue().dyn_cast<DenseElementsAttr>();
    if (!fillTy || !fillTy.getElementType().isF32() || !fillAttr ||
        fillAttr.getValues<float>()[0] != 0.0f) {
      return failure();
    }
    auto packDimsConcat =
        fill.getOutputDimensions().getDefiningOp<stablehlo::ConcatenateOp>();
    if (!packDimsConcat || packDimsConcat.getNumOperands() != 2 ||
        packDimsConcat.getDimension() != 0)
      return failure();
    Value head = packFieldOf(packDimsConcat.getOperand(0));
    if (!head) return failure();
    if (packDimConst) {
      auto headConst = constantInt(head);
      if (!headConst || *headConst != *packDimConst) return failure();
    } else if (head != packDim) {
      return failure();
    }
    Value tail = peelBcast(packFieldOf(packDimsConcat.getOperand(1)));
    if (!tail) return failure();
    if (auto tailConst = constantInt(tail)) {
      packConst = *tailConst;
    } else if (tail.getDefiningOp<stablehlo::ConstantOp>()) {
      return failure();
    } else {
      pack = tail;
    }
  } else if (auto fillConst = peelCastToConstant(m.concat.inputs()[1])) {
    auto ty = dyn_cast<RankedTensorType>(fillConst.getType());
    auto attr = dyn_cast<DenseElementsAttr>(fillConst.getValue());
    if (!ty || ty.getRank() != 2 || !ty.getElementType().isF32() || !attr ||
        !attr.isSplat() ||
        attr.getSplatValue<APFloat>().convertToDouble() != 0.0)
      return failure();
    if (ty.isDynamicDim(0) || ty.isDynamicDim(1) || !packDimConst ||
        ty.getDimSize(0) != *packDimConst)
      return failure();
    packConst = ty.getDimSize(1);
  } else {
    return failure();
  }

  if (pack) {
    auto ty = dyn_cast<RankedTensorType>(pack.getType());
    if (!ty || ty.getRank() != 0 || !ty.getElementType().isSignlessInteger(32))
      return failure();
  }

  m.packDim = packDim;
  m.packDimConst = packDimConst;
  m.pack = pack;
  m.packConst = packConst;

  m.params = m.g1.params();
  m.indices1 = m.g1.indices();
  m.indices2 = m.g2.indices();
  auto paramsTy = dyn_cast<RankedTensorType>(m.params.getType());
  if (!paramsTy || !paramsTy.getElementType().isF32() ||
      paramsTy.getRank() != 2)
    return failure();

  m.reshape = reshape;

  SmallPtrSet<Operation *, 16> seen;
  SmallVector<Operation *, 16> work;
  auto push = [&](Value v) {
    if (v == m.params || v == m.indices1 || v == m.indices2 || v == m.pack ||
        v == m.packDim)
      return;
    Operation *def = v.getDefiningOp();
    if (def && !isa<stablehlo::ConstantOp>(def) && seen.insert(def).second)
      work.push_back(def);
  };
  for (Value o : m.concat.inputs()) push(o);
  push(reshape.targetShape());
  push(m.w = m.concat.result());
  while (!work.empty()) {
    Operation *op = work.pop_back_val();
    m.members.push_back(op);
    for (Value o : op->getOperands()) push(o);
  }
  llvm::sort(m.members, [&](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });

  SmallPtrSet<Operation *, 16> memberSet(m.members.begin(), m.members.end());
  for (Operation *op : m.members) {
    for (Value r : op->getResults()) {
      if (r == m.w) continue;
      for (Operation *u : r.getUsers()) {
        bool inside = false;
        for (Operation *anc = u; anc != nullptr; anc = anc->getParentOp())
          if (memberSet.contains(anc)) {
            inside = true;
            break;
          }
        if (!inside) return failure();
      }
    }
  }
  if (m.w.use_empty()) return failure();
  return success();
}

// Outlines the site with ABI (params, indices1, indices2, pack_dim, pack);
// static pack values become rank-0 i32 promoted constants. The kernel name
// keeps the V1 rank specialization (R{r1}x{r2}), uniquified per module.
static LogicalResult fuseActionIdGather(
    Operation *anchorOp, PatternRewriter &rewriter,
    fusionv2::CallPortTable *portTable) {
  ActionIdGatherMatch m;
  if (failed(matchActionIdGather(anchorOp, m))) return failure();

  func::FuncOp main = anchorOp->getParentOfType<func::FuncOp>();
  Operation *tailOps[] = {m.concat.op};
  Value outputs[] = {m.w};

  std::string siteAnchor = fusionv2::anchorFromLoc(m.concat.op->getLoc());
  if (siteAnchor.empty()) return failure();
  SmallVector<std::pair<Value, std::string>> promoted;
  SmallVector<Operation *, 4> synth;
  auto cleanup = [&]() {
    for (Operation *op : llvm::reverse(synth)) rewriter.eraseOp(op);
  };
  auto scalarI32 = [&](int64_t value, llvm::StringRef tfName) -> Value {
    rewriter.setInsertionPoint(m.concat.op);
    auto ty = RankedTensorType::get({}, rewriter.getI32Type());
    auto cst = rewriter.create<stablehlo::ConstantOp>(
        NameLoc::get(rewriter.getStringAttr(tfName)),
        DenseElementsAttr::get(ty,
                               ArrayRef<int32_t>{static_cast<int32_t>(value)}));
    synth.push_back(cst.getOperation());
    promoted.emplace_back(cst.getResult(), tfName.str());
    return cst.getResult();
  };
  Value packDim = m.packDim;
  if (!packDim) {
    if (!m.packDimConst) return failure();
    packDim = scalarI32(*m.packDimConst, siteAnchor + "/pack_dim");
  }
  Value pack = m.pack;
  if (!pack) {
    if (!m.packConst) {
      cleanup();
      return failure();
    }
    pack = scalarI32(*m.packConst, siteAnchor + "/pack");
  }

  SmallVector<ProbeArg> args;
  if (failed(collectBoundary(main, m.members, args, promoted, {}, portTable))) {
    cleanup();
    return failure();
  }

  SmallVector<std::string> tailAnchors;
  for (Value out : outputs) {
    OpResult res = cast<OpResult>(out);
    tailAnchors.push_back(fusionv2::anchorFromLoc(res.getOwner()->getLoc()) +
                          ":" + std::to_string(res.getResultNumber()));
  }

  if (args.size() != 5) {
    cleanup();
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
  for (Value v : {m.params, m.indices1, m.indices2, packDim, pack})
    if (!pick(v)) {
      cleanup();
      return failure();
    }
  args.assign(ordered.begin(), ordered.end());

  ModuleOp module = main->getParentOfType<ModuleOp>();
  auto idx1Ty = dyn_cast<RankedTensorType>(m.indices1.getType());
  auto idx2Ty = dyn_cast<RankedTensorType>(m.indices2.getType());
  // Schema names only cover rank 1..2 (BuiltinKernelSchema); unranked or
  // deeper indices would outline a kernel the second level never lowers.
  if (!idx1Ty || !idx2Ty || idx1Ty.getRank() < 1 || idx1Ty.getRank() > 2 ||
      idx2Ty.getRank() < 1 || idx2Ty.getRank() > 2) {
    cleanup();
    return failure();
  }
  std::string name = uniquifySymbolName(
      module, "KPFusedEmbeddingActionIdGatherR" +
                  std::to_string(idx1Ty.getRank()) + "x" +
                  std::to_string(idx2Ty.getRank()));
  if (failed(outlineSite(main, rewriter, args, m.members, tailOps, outputs,
                         name, "kp_embedding_action_id_gather", tailAnchors,
                         {}, *portTable))) {
    cleanup();
    return failure();
  }
  return success();
}

// Native-carrier anchor (stablehlo.torch_index_select).
class FuseActionIdGatherV2Pattern
    : public OpRewritePattern<stablehlo::TorchIndexSelectOp> {
 public:
  FuseActionIdGatherV2Pattern(MLIRContext *ctx,
                              fusionv2::CallPortTable *portTable)
      : OpRewritePattern(ctx), portTable(portTable) {
    assert(portTable && "action_id_gather requires the pass-owned port table");
  }

 private:
  fusionv2::CallPortTable *portTable;

  LogicalResult matchAndRewrite(stablehlo::TorchIndexSelectOp anchor,
                                PatternRewriter &rewriter) const override {
    return fuseActionIdGather(anchor.getOperation(), rewriter, portTable);
  }
};

// Residue-carrier anchor (atir.unsupported{GatherV2}).
class FuseActionIdGatherResiduePattern
    : public OpRewritePattern<atir::UnsupportedOp> {
 public:
  FuseActionIdGatherResiduePattern(MLIRContext *ctx,
                                   fusionv2::CallPortTable *portTable)
      : OpRewritePattern(ctx), portTable(portTable) {
    assert(portTable && "action_id_gather requires the pass-owned port table");
  }

 private:
  fusionv2::CallPortTable *portTable;

  LogicalResult matchAndRewrite(atir::UnsupportedOp anchor,
                                PatternRewriter &rewriter) const override {
    if (anchor.getOpType() != "GatherV2") return failure();
    return fuseActionIdGather(anchor.getOperation(), rewriter, portTable);
  }
};

}  // namespace
}  // namespace atir

void atir::addActionIdGatherV2Pattern(RewritePatternSet &patterns,
                                      fusionv2::CallPortTable *portTable) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<FuseActionIdGatherV2Pattern>(ctx, portTable);
  patterns.add<FuseActionIdGatherResiduePattern>(ctx, portTable);
}
