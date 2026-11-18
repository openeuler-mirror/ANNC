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
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace atir::fusionv2;

namespace atir {
namespace {

// True for shape-carrying glue: every atir residue except the three
// semantic anchors, plus the stablehlo view triple.
static bool isScaffold(Operation *op) {
  if (auto residue = dyn_cast_or_null<atir::UnsupportedOp>(op)) {
    llvm::StringRef opType = residue.getOpType();
    return opType != "Fill" && opType != "Reshape" && opType != "ConcatV2";
  }
  return isa_and_nonnull<stablehlo::ConvertOp, stablehlo::ReshapeOp,
                         stablehlo::SliceOp>(op);
}

// Walks scaffold glue upwards and returns the first value matching accept;
// a non-scaffold op, a missing definition or too many hops ends the walk.
template <typename Accept>
static Value traceScaffold(Value v, Accept &&accept,
                           SmallVectorImpl<Operation *> *path) {
  for (int hop = 0; hop < 64; ++hop) {
    if (accept(v)) return v;
    Operation *def = v.getDefiningOp();
    if (!def || !isScaffold(def) || def->getNumOperands() == 0)
      return Value();
    if (path) path->push_back(def);
    v = def->getOperand(0);
  }
  return Value();
}

static bool isOriginShape(Value v) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  return ty && ty.getRank() == 1 && ty.getElementType().isSignlessInteger(64);
}

static bool isRowCount(Value v) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  return ty && ty.getRank() == 0 && ty.getElementType().isSignlessInteger(32);
}

// The getDim carrier in either spelling: the UnsupportedOp residue
// (opType "annc.getDim...") or the native atir.get_dim_v2 op. Shared by
// bridge piercing and the A-form origin terminal check.
static bool isGetDimCarrier(Operation *op) {
  if (auto residue = dyn_cast_or_null<atir::UnsupportedOp>(op))
    return residue.getOpType().starts_with("annc.getDim");
  return isa<atir::GetDimV2Op>(op);
}

// Movement contract: members move into the kernel; origin/rows/pack
// terminals and the anchor (+ anchorPath glue) stay in main reading the
// call results; glue with outside users is cloned by the stays mechanism.
struct PaddingMatch {
  ReshapeLike reshape;                    // anchor (Reshape residue)
  ConcatLike concat;                      // output-1 IR value (tail)
  SmallVector<Operation *, 8> anchorPath; // anchor -> concat view glue
  Operation *sub;                         // padding tail (sub)
  Value originVec;                        // origin_shape contract value
  Value inputRows;                        // input_rows contract value
  Value packConst;                        // pack contract value
  std::string packSource;                 // pack pb name (annc.const_source)
  Value data;                             // data contract value
  SmallVector<Operation *, 24> members;   // func body order
};

// One sub side: the optional bcast, the traced scaffold path and the role
// terminal (origin rank1 i64 / rows rank0 i32).
struct SideTrace {
  Operation *bcast = nullptr;
  Value terminal;
  SmallVector<Operation *, 8> path;
};

// Traces one sub operand to its role terminal; a shared bcast shape_of
// input is legal (the stays mechanism clones it later), anything else that
// breaks the walk returns an empty value.
static Value traceSideRole(Value subOperand, bool wantOrigin, SideTrace &t) {
  Operation *def = subOperand.getDefiningOp();
  Value root = subOperand;
  t.bcast = nullptr;
  t.path.clear();
  if (auto bcast = dyn_cast_or_null<stablehlo::DynamicBroadcastInDimOp>(def)) {
    if (!subOperand.hasOneUse()) return Value();
    t.bcast = def;
    root = bcast.getOperand();
  }
  return traceScaffold(
      root,
      [&](Value v) { return wantOrigin ? isOriginShape(v) : isRowCount(v); },
      &t.path);
}

// Fail-closed kp_embedding_padding match: anchor residue -> [view glue] ->
// concat(data, zero-fill); sub sides carry annc.prov, pack terminals are
// constant-or-blockarg only, member closure covers glue paths + sub + dims
// chain + fill + concat, and every closure member must stay escape-free.
// Escape handling is two-tier by design: spine ops (concat/dims/sub)
// reject on any outside user, while glue-path members with outside users
// rely on the outlineSite stays clone instead.
static LogicalResult matchKpEmbeddingPadding(Operation *anchorOp,
                                             PaddingMatch &m) {
  std::string locStr;
  if (annc::log::levelEnabled(annc::log::Level::Debug))
    llvm::raw_string_ostream(locStr) << anchorOp->getLoc();

  ReshapeLike anchor = ReshapeLike::match(anchorOp);
  if (!anchor || !anchor.targetShape()) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": anchor invalid\n";
    return failure();
  }
  m.reshape = anchor;

  m.anchorPath.clear();
  Value scan = anchor.op->getOperand(0);
  for (int hop = 0; hop < 16; ++hop) {
    if (auto c = ConcatLike::match(scan.getDefiningOp());
        c && c.inputs().size() == 2 && c.dimension() == 0) {
      m.concat = c;
      break;
    }
    Operation *def = scan.getDefiningOp();
    if (!def || !isScaffold(def) || def->getNumOperands() == 0) {
      ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": concat chain break\n";
      return failure();
    }
    m.anchorPath.push_back(def);
    scan = def->getOperand(0);
  }
  if (!m.concat) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": concat hop limit\n";
    return failure();
  }
  ConcatLike concat = m.concat;

  m.data = concat.inputs()[0];
  auto dataTy = dyn_cast<RankedTensorType>(m.data.getType());
  if (!dataTy || dataTy.getRank() != 2 || !dataTy.getElementType().isF32()) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": data type\n";
    return failure();
  }
  if (!isa<BlockArgument>(m.data)) {
    Operation *def = m.data.getDefiningOp();
    bool carrierLike = def && (def->getDialect()->getNamespace() == "atir" ||
                               isa<func::CallOp>(def));
    if (!carrierLike) {
      ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": data carrier\n";
      return failure();
    }
  }

  FillLike fill = FillLike::match(concat.inputs()[1].getDefiningOp());
  if (!fill) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": fill form\n";
    return failure();
  }
  Operation *fillOp = fill.op;
  auto fillConst =
      fill.value().getDefiningOp<stablehlo::ConstantOp>();
  auto fillAttr = fillConst
                      ? dyn_cast<DenseElementsAttr>(fillConst.getValue())
                      : DenseElementsAttr();
  if (!fillAttr || !fillAttr.isSplat() ||
      !fillAttr.getElementType().isF32() ||
      fillAttr.getSplatValue<APFloat>().convertToDouble() != 0.0) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": fill zero splat\n";
    return failure();
  }

  SmallVector<Operation *, 8> dimsPath;
  Value dimsV = traceScaffold(
      fill.shape(),
      [](Value v) {
        auto c = ConcatLike::match(v.getDefiningOp());
        return static_cast<bool>(c) && c.inputs().size() == 2 &&
               c.dimension() == 0;
      },
      &dimsPath);
  ConcatLike dimsConcat = ConcatLike::match(dimsV ? dimsV.getDefiningOp()
                                                  : nullptr);
  if (!dimsConcat) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": dims concat\n";
    return failure();
  }

  SmallVector<Operation *, 8> subPath;
  Value subV = traceScaffold(
      dimsConcat.inputs()[0],
      [](Value v) {
        return isa_and_nonnull<stablehlo::SubtractOp>(v.getDefiningOp());
      },
      &subPath);
  auto sub =
      subV ? dyn_cast<stablehlo::SubtractOp>(subV.getDefiningOp()) : nullptr;
  if (!sub) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": sub\n";
    return failure();
  }
  m.sub = sub.getOperation();

  SmallVector<Operation *, 8> packPath;
  auto isPackTerm = [&](Value v) {
    auto ty = dyn_cast<RankedTensorType>(v.getType());
    if (!ty || !ty.getElementType().isSignlessInteger(32)) return false;
    bool constLike =
        isa<BlockArgument>(v) ||
        v.getDefiningOp<stablehlo::ConstantOp>() != nullptr;
    if (!constLike) return false;
    if (ty.getRank() == 0) return true;
    return ty.getRank() == 1 && !ty.isDynamicDim(0) && ty.getDimSize(0) == 1;
  };
  m.packConst = traceScaffold(dimsConcat.inputs()[1], isPackTerm, &packPath);
  if (!m.packConst || isScaffold(m.packConst.getDefiningOp())) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": pack terminal\n";
    return failure();
  }
  if (auto cst = m.packConst.getDefiningOp<stablehlo::ConstantOp>())
    if (auto src = cst->getAttrOfType<StringAttr>("annc.const_source"))
      m.packSource = src.getValue().str();
  // FE contract (TagConcatConstSources): CSE merges pack constants, so the
  // per-site dims concat carries the nearest pb Const name instead.
  if (m.packSource.empty())
    if (auto src = dimsConcat.op->getAttrOfType<StringAttr>(
            "annc.const_source"))
      m.packSource = src.getValue().str();

  ProvRecord provSink0, provSink1;
  if (!operandProvOf(sub.getOperation(), 0, provSink0) ||
      !operandProvOf(sub.getOperation(), 1, provSink1)) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": prov missing\n";
    return failure();
  }
  SideTrace lhs, rhs;
  Value originTerm = traceSideRole(sub.getLhs(), /*wantOrigin=*/true, lhs);
  Value rowsTerm = traceSideRole(sub.getRhs(), /*wantOrigin=*/false, rhs);
  if (!originTerm || !rowsTerm) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": side terminal\n";
    return failure();
  }
  m.originVec = originTerm;
  m.inputRows = rowsTerm;

  if (!isa<BlockArgument>(m.originVec)) {
    Operation *originDef = m.originVec.getDefiningOp();
    bool cForm = isa<atir::SparseReshapeV2Op>(originDef) &&
                 m.originVec == originDef->getResult(1);
    if (!cForm) {
      if (!originDef || originDef->getNumOperands() != 1 ||
          !isGetDimCarrier(originDef)) {
        ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": origin form\n";
        return failure();
      }
      Value originCarrier = originDef->getOperand(0);
      if (!originCarrier.getDefiningOp() ||
          originCarrier.getDefiningOp()->getDialect()->getNamespace() !=
              "atir" ||
          !isa<UnrankedTensorType>(originCarrier.getType())) {
        ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": origin carrier\n";
        return failure();
      }
    }
  }

  if (!fillOp->getResult(0).hasOneUse()) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": fill escape\n";
    return failure();
  }
  if (anchor.result().use_empty()) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": anchor dead\n";
    return failure();
  }

  SmallPtrSet<Operation *, 24> chainSet;
  chainSet.insert(dimsPath.begin(), dimsPath.end());
  chainSet.insert(subPath.begin(), subPath.end());
  chainSet.insert(packPath.begin(), packPath.end());
  chainSet.insert(lhs.path.begin(), lhs.path.end());
  chainSet.insert(rhs.path.begin(), rhs.path.end());
  if (lhs.bcast) chainSet.insert(lhs.bcast);
  if (rhs.bcast) chainSet.insert(rhs.bcast);
  chainSet.insert(sub.getOperation());
  chainSet.insert(dimsConcat.op);
  chainSet.insert(fillOp);
  chainSet.insert(concat.op);

  auto noEscape = [&](Operation *op) {
    for (Operation *user : op->getUsers()) {
      if (user == anchorOp || chainSet.contains(user)) continue;
      if (llvm::is_contained(m.anchorPath, user)) continue;
      return false;
    }
    return true;
  };
  if (!noEscape(concat.op) || !noEscape(dimsConcat.op) ||
      !noEscape(sub.getOperation())) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": member escape\n";
    return failure();
  }
  for (Operation &op : anchorOp->getParentOfType<func::FuncOp>()
                           .getBody()
                           .front()
                           .getOperations())
    if (chainSet.contains(&op)) m.members.push_back(&op);
  return success();
}

// Lands the site: synthesizes the padding scalar and the [-1, C]
// reshape_sizes, promotes origin/rows/pack with producer-provenance names,
// orders the ABI slots by semantic value and outlines through outlineSite.
// Failure paths roll every synthesized op back.
static LogicalResult fuseKpEmbeddingPadding(
    Operation *anchorOp, PatternRewriter &rewriter,
    fusionv2::CallPortTable *portTable) {
  PaddingMatch m;
  if (failed(matchKpEmbeddingPadding(anchorOp, m))) return failure();
  func::FuncOp main = anchorOp->getParentOfType<func::FuncOp>();
  if (!main) return failure();

  std::string anchorName = anchorFromLoc(m.reshape.op->getLoc());
  if (anchorName.empty()) anchorName = anchorFromLoc(m.concat.op->getLoc());
  if (anchorName.empty()) return failure();

  SmallVector<std::pair<Value, std::string>> promoted;
  SmallVector<Operation *, 4> synth;
  auto cleanup = [&]() {
    for (Operation *op : llvm::reverse(synth)) {
      llvm::erase(m.members, op);
      rewriter.eraseOp(op);
    }
  };
  rewriter.setInsertionPointAfter(m.sub);

  auto dataTy = cast<RankedTensorType>(m.data.getType());
  if (dataTy.isDynamicDim(1)) {
    ANNC_LOG_DEBUG("op-fusion-v2") << "dynamic data cols; dropped\n";
    cleanup();
    return failure();
  }

  Value reshapeSizes = m.reshape.targetShape();
  std::string rsName;
  if (auto blockArg = dyn_cast<BlockArgument>(reshapeSizes)) {
    auto entryNames = entryInputNames(main);
    if (blockArg.getArgNumber() < entryNames.size())
      rsName = entryNames[blockArg.getArgNumber()];
  }
  if (rsName.empty()) {
    if (static_cast<uint64_t>(dataTy.getDimSize(1)) > 0x7fffffff) {
      cleanup();
      return failure();
    }
    auto rsTy = RankedTensorType::get({2}, rewriter.getI32Type());
    auto synthesized = rewriter.create<stablehlo::ConstantOp>(
        NameLoc::get(rewriter.getStringAttr(anchorName + "/shape")),
        DenseElementsAttr::get(
            rsTy, ArrayRef<int32_t>{
                      -1, static_cast<int32_t>(dataTy.getDimSize(1))}));
    synth.push_back(synthesized.getOperation());
    reshapeSizes = synthesized.getResult();
    rsName = anchorName + "/shape";
  }
  promoted.emplace_back(reshapeSizes, rsName);

  auto subTy = cast<RankedTensorType>(m.sub->getResult(0).getType());
  auto padScalar = rewriter.create<stablehlo::ReshapeOp>(
      m.sub->getLoc(),
      TypeRange{RankedTensorType::get({}, subTy.getElementType())},
      ValueRange{m.sub->getResult(0)});
  synth.push_back(padScalar.getOperation());
  auto memIt = llvm::find(m.members, m.sub);
  if (memIt == m.members.end()) {
    cleanup();
    return failure();
  }
  m.members.insert(std::next(memIt), padScalar.getOperation());

  if (cast<RankedTensorType>(m.packConst.getType()).getRank() == 1) {
    rewriter.setInsertionPointAfterValue(m.packConst);
    auto packScalar = rewriter.create<stablehlo::ReshapeOp>(
        m.packConst.getDefiningOp()->getLoc(),
        TypeRange{RankedTensorType::get({}, rewriter.getI32Type())},
        ValueRange{m.packConst});
    synth.push_back(packScalar.getOperation());
    m.packConst = packScalar.getResult();
  }

  if (!isa<BlockArgument>(m.originVec)) {
    Value nv = m.originVec;
    for (int hop = 0; hop < 8 && isGetDimCarrier(nv.getDefiningOp()); ++hop)
      nv = nv.getDefiningOp()->getOperand(0);
    std::string originName = sourceNameOf(nv);
    if (originName.empty()) {
      ANNC_LOG_DEBUG("op-fusion-v2")
          << "kp_embedding_padding: origin name empty\n";
      cleanup();
      return failure();
    }
    promoted.emplace_back(m.originVec, originName);
  }
  if (!isa<BlockArgument>(m.inputRows)) {
    std::string rowsName = sourceNameOf(m.inputRows);
    if (rowsName.empty()) {
      ANNC_LOG_DEBUG("op-fusion-v2")
          << "kp_embedding_padding: rows name empty\n";
      cleanup();
      return failure();
    }
    promoted.emplace_back(m.inputRows, rowsName);
  }
  if (!isa<BlockArgument>(m.packConst)) {
    std::string packName = m.packSource;
    if (packName.empty())
      packName = singleAnchorName(m.packConst.getDefiningOp()->getLoc());
    if (packName.empty()) packName = anchorName + "/pack";
    promoted.emplace_back(m.packConst, packName);
  }

  SmallVector<Operation *> tailOps = {m.sub, padScalar.getOperation(),
                                      m.concat.op};
  Value outputs[] = {padScalar.getResult(), m.concat.result()};
  SmallVector<std::string> tailAnchors;
  for (Value out : outputs) {
    auto res = mlir::dyn_cast<mlir::OpResult>(out);
    if (!res) {
      cleanup();
      return failure();
    }
    std::string anchor = fusionv2::anchorFromLoc(res.getOwner()->getLoc());
    if (anchor.empty()) {
      // Output names are converter wiring: a synthesized fallback would be
      // dropped downstream, so an anonymous loc drops the site here.
      cleanup();
      return failure();
    }
    tailAnchors.push_back(anchor + ":" +
                          std::to_string(res.getResultNumber()));
  }

  SmallVector<ProbeArg> args;
  if (failed(collectBoundary(main, m.members, args, promoted, {},
                           portTable))) {
    cleanup();
    return failure();
  }
  if (args.size() != 5) {
    ANNC_LOG_DEBUG("op-fusion-v2")
        << "args=" << args.size() << " (want 5); dropped\n";
    cleanup();
    return failure();
  }

  SmallVector<ProbeArg> ordered;
  auto takeValue = [&](Value v) -> bool {
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i].value != v) continue;
      ordered.push_back(args[i]);
      args.erase(args.begin() + i);
      return true;
    }
    return false;
  };
  if (!takeValue(m.originVec) || !takeValue(m.inputRows) ||
      !takeValue(m.packConst) || !takeValue(m.data) ||
      !takeValue(reshapeSizes) || !args.empty()) {
    ANNC_LOG_DEBUG("op-fusion-v2")
        << "kp_embedding_padding: slot mismatch; dropped\n";
    cleanup();
    return failure();
  }
  args.assign(ordered.begin(), ordered.end());

  ModuleOp module = main->getParentOfType<ModuleOp>();
  std::string base = sanitizeName("KPFusedEmbeddingPadding_" + anchorName);
  std::string name =
      uniquifySymbolName(module, base + "_" + hash4(base));
  if (failed(outlineSite(main, rewriter, args, m.members, tailOps, outputs,
                         name, "kp_embedding_padding", tailAnchors, {},
                         *portTable))) {
    ANNC_LOG_DEBUG("op-fusion-v2")
        << "kp_embedding_padding: outlineSite failed\n";
    cleanup();
    return failure();
  }
  return success();
}

// Residue-form anchor (opType="Reshape" with a runtime shape operand);
// kernel bodies are never re-entered.
class FuseKpEmbeddingPaddingV2Pattern
    : public OpRewritePattern<atir::UnsupportedOp> {
 public:
  FuseKpEmbeddingPaddingV2Pattern(MLIRContext *ctx,
                                  fusionv2::CallPortTable *portTable)
      : OpRewritePattern(ctx), portTable(portTable) {
    assert(portTable && "kp_embedding_padding requires the pass port table");
  }

 private:
  fusionv2::CallPortTable *portTable;

  LogicalResult matchAndRewrite(atir::UnsupportedOp anchor,
                                PatternRewriter &rewriter) const override {
    if (anchor.getOpType() != "Reshape") return failure();
    if (auto fn = anchor->getParentOfType<func::FuncOp>())
      if (fn->hasAttr("annc.kernel")) return failure();
    return fuseKpEmbeddingPadding(anchor.getOperation(), rewriter, portTable);
  }
};

}  // namespace
}  // namespace atir

void atir::addEmbeddingPaddingV2Pattern(mlir::RewritePatternSet &patterns,
                                        fusionv2::CallPortTable *portTable) {
  patterns.add<FuseKpEmbeddingPaddingV2Pattern>(
      patterns.getContext(), portTable);
}
