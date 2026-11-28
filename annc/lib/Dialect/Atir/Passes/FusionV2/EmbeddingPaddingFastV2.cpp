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
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace atir::fusionv2;

namespace atir {
namespace {

struct PaddingFastMatch {
  ReshapeLike padded;
  Value originVec, inputRows, pack, data, reshapeSizes;
  Value paddingRows;  // site output 0 (sub, or its 0-D form)
  Value reshapeRows;  // site output 1 (the anchor result)
  SmallVector<Operation *, 24> members;  // func body order
};

static bool isOriginShape(Value v) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  return ty && ty.getRank() == 1 && ty.getElementType().isSignlessInteger(64);
}

static bool isRowCount(Value v) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  return ty && ty.getRank() == 0 && ty.getElementType().isSignlessInteger(32);
}

// Scalar/vector carrying glue: the stablehlo view triple, operand 0 only.
static bool isScalarScaffold(Operation *op) {
  return isa_and_nonnull<stablehlo::ReshapeOp, stablehlo::ConvertOp,
                         stablehlo::SliceOp>(op);
}

// Walks scalar glue upwards to the first value matching accept.
template <typename Accept>
static Value traceScalar(Value v, Accept &&accept) {
  for (int hop = 0; hop < 32; ++hop) {
    if (accept(v)) return v;
    Operation *def = v.getDefiningOp();
    if (!def || !isScalarScaffold(def) || def->getNumOperands() == 0)
      return Value();
    v = def->getOperand(0);
  }
  return Value();
}

// Origin-side walk: rank1 i64 terminals only (an intermediate scalar
// reshape would otherwise be mistaken for the row count).
static Value traceOriginShape(Value v) {
  for (int hop = 0; hop < 32; ++hop) {
    if (isOriginShape(v)) return v;
    Operation *def = v.getDefiningOp();
    if (!def || !isScalarScaffold(def) || def->getNumOperands() == 0)
      return Value();
    v = def->getOperand(0);
  }
  return Value();
}

// Reverse walk from the anchor to shape_of(padded) through the slice /
// index_cast / reshape scaffolding.
static Value traceToShapeOf(Value v) {
  for (int hop = 0; hop < 16; ++hop) {
    if (auto shapeOf = v.getDefiningOp<shape::ShapeOfOp>())
      return shapeOf.getOperand();
    Operation *def = v.getDefiningOp();
    if (!def ||
        !isa<arith::IndexCastOp, stablehlo::SliceOp, stablehlo::ReshapeOp>(
            def) ||
        def->getNumOperands() == 0)
      return Value();
    v = def->getOperand(0);
  }
  return Value();
}

// StridedSlice residue mask contract (shrink=1, single-element windows).
static bool matchShrinkSlice(atir::UnsupportedOp slice) {
  if (slice.getOpType() != "StridedSlice" || slice.getNumOperands() != 4)
    return false;
  auto meta = slice->getAttrOfType<DictionaryAttr>("metadata");
  auto maskIs = [&](llvm::StringRef name, int64_t want) {
    auto attr = meta ? meta.getAs<IntegerAttr>(name) : IntegerAttr();
    return attr && attr.getValue().getSExtValue() == want;
  };
  return maskIs("shrink_axis_mask", 1) && maskIs("begin_mask", 0) &&
         maskIs("end_mask", 0) && maskIs("ellipsis_mask", 0) &&
         maskIs("new_axis_mask", 0);
}

// Fail-closed kp_embedding_padding_fast match: anchor -> shape_of(padded)
// -> concat(data, zero-fill); the pack_dims concat hides sub + pack; the
// sub sides are prov-tagged with mirrored order fallback; the member
// closure runs through collectDataMembers and must stay escape-free.
static LogicalResult matchKpPaddingFast(Value rowsValue, Operation *anchorOp,
                                        PaddingFastMatch &m) {
  auto rowsTy = dyn_cast<RankedTensorType>(rowsValue.getType());
  if (!rowsTy || rowsTy.getRank() != 0 ||
      !rowsTy.getElementType().isSignlessInteger(32))
    return failure();
  std::string locStr;
  if (annc::log::levelEnabled(annc::log::Level::Debug))
    llvm::raw_string_ostream(locStr) << anchorOp->getLoc();

  Value paddedValue;
  if (isa<stablehlo::ReshapeOp>(anchorOp)) {
    paddedValue = traceToShapeOf(anchorOp->getOperand(0));
  } else if (auto slice = dyn_cast<atir::UnsupportedOp>(anchorOp)) {
    paddedValue = traceToShapeOf(slice.getOperand(0));
  } else {
    return failure();
  }
  if (!paddedValue) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": shape_of trace\n";
    return failure();
  }
  m.padded = ReshapeLike::match(paddedValue.getDefiningOp());
  if (!m.padded) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": padded form\n";
    return failure();
  }

  auto concat = m.padded.src().getDefiningOp<stablehlo::ConcatenateOp>();
  if (!concat || concat.getNumOperands() != 2 || concat.getDimension() != 0) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": concat\n";
    return failure();
  }
  m.data = concat.getOperand(0);
  auto dataTy = dyn_cast<RankedTensorType>(m.data.getType());
  if (!dataTy || dataTy.getRank() != 2 || !dataTy.getElementType().isF32()) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": data\n";
    return failure();
  }

  auto fill =
      concat.getOperand(1)
          .getDefiningOp<stablehlo::DynamicBroadcastInDimOp>();
  if (!fill) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": fill form\n";
    return failure();
  }
  auto fillVal = fill.getOperand().getDefiningOp<stablehlo::ConstantOp>();
  auto fillAttr = fillVal
                      ? dyn_cast<DenseElementsAttr>(fillVal.getValue())
                      : DenseElementsAttr();
  if (!fillAttr || !fillAttr.isSplat() ||
      !fillAttr.getElementType().isF32() ||
      fillAttr.getSplatValue<APFloat>().convertToDouble() != 0.0) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": fill zero\n";
    return failure();
  }
  auto packDims =
      fill.getOutputDimensions()
          .getDefiningOp<stablehlo::ConcatenateOp>();
  if (!packDims || packDims.getNumOperands() != 2 ||
      packDims.getDimension() != 0) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": pack dims\n";
    return failure();
  }

  Value subValue = traceScalar(packDims.getOperand(0), [](Value v) {
    return isa_and_nonnull<stablehlo::SubtractOp>(v.getDefiningOp());
  });
  if (!subValue) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": sub locate\n";
    return failure();
  }
  auto sub = cast<stablehlo::SubtractOp>(subValue.getDefiningOp());
  m.pack = traceScalar(packDims.getOperand(1), isRowCount);
  if (!m.pack) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": pack locate\n";
    return failure();
  }

  ProvRecord provSink0, provSink1;
  if (!operandProvOf(sub.getOperation(), 0, provSink0) ||
      !operandProvOf(sub.getOperation(), 1, provSink1)) {
    ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": prov missing\n";
    return failure();
  }
  Value originL = traceOriginShape(sub.getLhs());
  Value rowsR = traceScalar(sub.getRhs(), isRowCount);
  if (originL && rowsR) {
    m.originVec = originL;
    m.inputRows = rowsR;
  } else {
    Value originR = traceOriginShape(sub.getRhs());
    Value rowsL = traceScalar(sub.getLhs(), isRowCount);
    if (!originR || !rowsL) {
      ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": side terminal\n";
      return failure();
    }
    m.originVec = originR;
    m.inputRows = rowsL;
  }

  m.reshapeSizes = m.padded.targetShape();
  if (m.reshapeSizes) {
    auto rsTy = dyn_cast<RankedTensorType>(m.reshapeSizes.getType());
    if (!rsTy || rsTy.getRank() != 1 ||
        !rsTy.getElementType().isSignlessInteger(32))
      return failure();
  }

  m.paddingRows = sub.getResult();
  auto padTy = dyn_cast<RankedTensorType>(m.paddingRows.getType());
  if (!padTy) return failure();
  if (padTy.getRank() != 0 &&
      !(padTy.getRank() == 1 && !padTy.isDynamicDim(0) &&
        padTy.getDimSize(0) == 1))
    return failure();

  m.reshapeRows = rowsValue;

  SmallPtrSet<Operation *, 24> seen;
  if (isa<atir::UnsupportedOp>(anchorOp)) {
    seen.insert(anchorOp);
    m.members.push_back(anchorOp);
    for (Value operand : anchorOp->getOperands())
      collectDataMembers(operand, seen, m.members);
  } else {
    collectDataMembers(m.reshapeRows, seen, m.members);
  }
  collectDataMembers(m.paddingRows, seen, m.members);
  llvm::sort(m.members, [&](Operation *lhsOp, Operation *rhsOp) {
    return lhsOp->isBeforeInBlock(rhsOp);
  });
  if (m.members.empty()) return failure();

  SmallPtrSet<Operation *, 24> memberSet(m.members.begin(), m.members.end());
  auto insideSite = [&](Operation *user) {
    for (Operation *anc = user; anc != nullptr; anc = anc->getParentOp())
      if (memberSet.contains(anc)) return true;
    return false;
  };
  for (Operation *op : m.members) {
    for (Value r : op->getResults()) {
      if (r == m.paddingRows || r == m.reshapeRows) continue;
      for (Operation *user : r.getUsers()) {
        if (insideSite(user)) continue;
        if (user->getDialect()->getNamespace() == "shape")
          continue;
        ANNC_LOG_DEBUG("op-fusion-v2") << locStr << ": member escape\n";
        return failure();
      }
    }
  }
  return success();
}

// Lands the fast site: 0-D normalization of padding_rows, [-1, C]
// reshape_sizes synthesis, constant promotion via annc.const_source and
// producer-provenance naming for origin / input_rows. Failure paths roll
// every synthesized op back.
static LogicalResult fusePaddingFast(Value rowsValue, Operation *anchorOp,
                                     PatternRewriter &rewriter,
                                     fusionv2::CallPortTable *portTable) {
  if (auto fn = anchorOp->getParentOfType<func::FuncOp>())
    if (fn->hasAttr("annc.kernel")) return failure();
  PaddingFastMatch m;
  if (failed(matchKpPaddingFast(rowsValue, anchorOp, m))) return failure();
  func::FuncOp main = anchorOp->getParentOfType<func::FuncOp>();
  if (!main) return failure();
  std::string anchorName = anchorFromLoc(anchorOp->getLoc());
  if (anchorName.empty()) return failure();

  SmallVector<std::pair<Value, std::string>> promoted;
  SmallVector<Operation *, 2> synth;
  auto cleanup = [&]() {
    for (Operation *op : llvm::reverse(synth)) {
      llvm::erase(m.members, op);
      rewriter.eraseOp(op);
    }
  };
  rewriter.setInsertionPoint(anchorOp);

  Value paddingRows = m.paddingRows;
  auto padTy = cast<RankedTensorType>(paddingRows.getType());
  if (padTy.getRank() == 1) {
    auto scalar = rewriter.create<stablehlo::ReshapeOp>(
        paddingRows.getDefiningOp()->getLoc(),
        TypeRange{RankedTensorType::get({}, padTy.getElementType())},
        ValueRange{paddingRows});
    synth.push_back(scalar.getOperation());
    paddingRows = scalar.getResult();
    m.members.push_back(scalar.getOperation());
    llvm::sort(m.members, [&](Operation *lhsOp, Operation *rhsOp) {
      return lhsOp->isBeforeInBlock(rhsOp);
    });
  }

  bool synthesizedReshapeSizes = false;
  if (!m.reshapeSizes) {
    auto dataTy = cast<RankedTensorType>(m.data.getType());
    if (dataTy.isDynamicDim(1)) {
      cleanup();
      return failure();
    }
    auto rsTy = RankedTensorType::get({2}, rewriter.getI32Type());
    auto rs = rewriter.create<stablehlo::ConstantOp>(
        NameLoc::get(rewriter.getStringAttr(anchorName + "/shape")),
        DenseElementsAttr::get(
            rsTy, ArrayRef<int32_t>{
                      -1, static_cast<int32_t>(dataTy.getDimSize(1))}));
    synth.push_back(rs.getOperation());
    m.reshapeSizes = rs.getResult();
    promoted.emplace_back(m.reshapeSizes, anchorName + "/shape");
    synthesizedReshapeSizes = true;
  }

  auto constSourceName = [](Value v) -> std::string {
    Operation *def = v.getDefiningOp();
    if (!def) return "";
    if (auto src = def->getAttrOfType<StringAttr>("annc.const_source"))
      return src.getValue().str();
    return "";
  };
  Operation *packDef = m.pack.getDefiningOp();
  if (packDef && isa<stablehlo::ConstantOp>(packDef)) {
    std::string packName = constSourceName(m.pack);
    if (packName.empty()) packName = anchorName + "/pack";
    promoted.emplace_back(m.pack, packName);
  }
  if (!synthesizedReshapeSizes) {
    Operation *rsDef = m.reshapeSizes.getDefiningOp();
    if (rsDef && isa<stablehlo::ConstantOp>(rsDef)) {
      std::string rsName = constSourceName(m.reshapeSizes);
      if (rsName.empty()) rsName = anchorName + "/shape";
      promoted.emplace_back(m.reshapeSizes, rsName);
    }
  }

  auto dimBridge = [](Operation *d) {
    if (!d || d->getNumOperands() != 1) return false;
    auto residue = dyn_cast<atir::UnsupportedOp>(d);
    return residue && residue.getOpType().starts_with("annc.getDim");
  };
  if (!isa<BlockArgument>(m.originVec)) {
    Value nv = m.originVec;
    for (int hop = 0; hop < 8 && dimBridge(nv.getDefiningOp()); ++hop)
      nv = nv.getDefiningOp()->getOperand(0);
    std::string originName = sourceNameOf(nv);
    if (originName.empty()) {
      cleanup();
      return failure();
    }
    promoted.emplace_back(m.originVec, originName);
  }
  if (!isa<BlockArgument>(m.inputRows)) {
    std::string rowsName = sourceNameOf(m.inputRows);
    if (rowsName.empty()) {
      cleanup();
      return failure();
    }
    promoted.emplace_back(m.inputRows, rowsName);
  }

  Operation *tailOps[] = {paddingRows.getDefiningOp(), anchorOp};
  Value outputs[] = {paddingRows, m.reshapeRows};
  SmallVector<std::string> tailAnchors;
  for (Value out : outputs) {
    auto res = mlir::dyn_cast<mlir::OpResult>(out);
    if (!res) return failure();
    std::string anchor = fusionv2::anchorFromLoc(res.getOwner()->getLoc());
    if (anchor.empty()) return failure();
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
  auto pick = [&](Value v) {
    for (auto it = args.begin(); it != args.end(); ++it) {
      if (it->value != v) continue;
      ordered.push_back(*it);
      args.erase(it);
      return true;
    }
    return false;
  };
  for (Value v : {m.originVec, m.inputRows, m.pack, m.data, m.reshapeSizes})
    if (!pick(v)) {
      cleanup();
      return failure();
    }
  args.assign(ordered.begin(), ordered.end());

  ModuleOp module = main->getParentOfType<ModuleOp>();
  std::string base =
      sanitizeName("KPFusedEmbeddingPaddingFast_" + anchorName);
  std::string name =
      uniquifySymbolName(module, base + "_" + hash4(base));
  if (failed(outlineSite(main, rewriter, args, m.members, tailOps, outputs,
                         name, "kp_embedding_padding_fast", tailAnchors, {},
                         *portTable))) {
    cleanup();
    return failure();
  }
  return success();
}

// Scalar-reshape anchor (the import-folded StridedSlice form).
class FusePaddingFastScalarPattern
    : public OpRewritePattern<stablehlo::ReshapeOp> {
 public:
  FusePaddingFastScalarPattern(MLIRContext *ctx,
                               fusionv2::CallPortTable *portTable)
      : OpRewritePattern(ctx), portTable(portTable) {
    assert(portTable && "padding_fast requires the pass port table");
  }

 private:
  fusionv2::CallPortTable *portTable;

  LogicalResult matchAndRewrite(stablehlo::ReshapeOp anchor,
                                PatternRewriter &rewriter) const override {
    auto ty = dyn_cast<RankedTensorType>(anchor.getType());
    if (!ty || ty.getRank() != 0 || !ty.getElementType().isSignlessInteger(32))
      return failure();
    return fusePaddingFast(anchor.getResult(), anchor.getOperation(),
                           rewriter, portTable);
  }
};

// StridedSlice residue anchor (the unfolded V1 form).
class FusePaddingFastSlicePattern
    : public OpRewritePattern<atir::UnsupportedOp> {
 public:
  FusePaddingFastSlicePattern(MLIRContext *ctx,
                              fusionv2::CallPortTable *portTable)
      : OpRewritePattern(ctx), portTable(portTable) {
    assert(portTable && "padding_fast requires the pass port table");
  }

 private:
  fusionv2::CallPortTable *portTable;

  LogicalResult matchAndRewrite(atir::UnsupportedOp anchor,
                                PatternRewriter &rewriter) const override {
    if (anchor.getOpType() != "StridedSlice") return failure();
    if (!matchShrinkSlice(anchor)) return failure();
    return fusePaddingFast(anchor.getResult(0), anchor.getOperation(),
                           rewriter, portTable);
  }
};

}  // namespace
}  // namespace atir

void atir::addEmbeddingPaddingFastV2Pattern(
    mlir::RewritePatternSet &patterns,
    fusionv2::CallPortTable *portTable) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<FusePaddingFastScalarPattern, FusePaddingFastSlicePattern>(
      ctx, portTable);
}
