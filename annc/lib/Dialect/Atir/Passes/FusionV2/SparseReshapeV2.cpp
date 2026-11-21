#include <array>
#include <string>

#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/FusionV2/FusionV2Outline.h"
#include "Dialect/Atir/Passes/FusionV2/Naming.h"
#include "Dialect/Atir/Passes/FusionV2/Patterns.h"
#include "Support/Log.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace atir::fusionv2;

namespace atir {
namespace {

struct SparseReshapeMatch {
  atir::SparseReshapeV2Op sr;
  Value keys, begin, newShape, packConst;
  Value beginConst;
  Value newShapeConst;
  stablehlo::ConstantOp packSrcConst;
  Value savedInShape;  // original input_shape for failure rollback
  SmallVector<Operation *, 16> members;
  std::string beginSourceName;
};

static bool isResidueOp(Value v, llvm::StringRef opType) {
  auto residue = dyn_cast_or_null<atir::UnsupportedOp>(v.getDefiningOp());
  return residue && residue.getOpType() == opType;
}

static bool matchStridedSliceMask(atir::UnsupportedOp slice, int64_t shrink,
                                  int64_t beginMask, int64_t endMask) {
  auto meta = slice->getAttrOfType<DictionaryAttr>("metadata");
  if (!meta) return false;
  auto maskIs = [&](llvm::StringRef name, int64_t want) {
    auto attr = meta.getAs<IntegerAttr>(name);
    return attr && attr.getValue().getSExtValue() == want;
  };
  return maskIs("shrink_axis_mask", shrink) &&
         maskIs("begin_mask", beginMask) && maskIs("end_mask", endMask) &&
         maskIs("ellipsis_mask", 0) && maskIs("new_axis_mask", 0);
}

static bool constInt1D(Value v, SmallVectorImpl<int64_t> &out) {
  auto def = v.getDefiningOp<stablehlo::ConstantOp>();
  if (!def) return false;
  auto attr = def.getValue().dyn_cast<DenseElementsAttr>();
  if (!attr || !attr.getElementType().isInteger()) return false;
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  if (!ty || ty.getRank() != 1) return false;
  out.clear();
  for (APInt v2 : attr.getValues<APInt>()) out.push_back(v2.getSExtValue());
  return true;
}

static bool reshapeToColumn(atir::UnsupportedOp reshape) {
  if (reshape.getNumOperands() != 2) return false;
  SmallVector<int64_t, 2> shape;
  if (!constInt1D(reshape.getOperand(1), shape)) return false;
  return shape.size() == 2 && shape[0] == -1 && shape[1] == 1;
}

static bool matchSegColumn(Value segCol, Value &keys, Value &begin,
                           std::string &beginSourceName) {
  beginSourceName.clear();
  if (auto reshape =
          dyn_cast_or_null<atir::UnsupportedOp>(segCol.getDefiningOp())) {
    if (reshape.getOpType() != "Reshape" || !reshapeToColumn(reshape))
      return false;
    auto slice = dyn_cast_or_null<atir::UnsupportedOp>(
        reshape.getOperand(0).getDefiningOp());
    if (!slice || slice.getOpType() != "StridedSlice" ||
        slice.getNumOperands() != 4)
      return false;
    if (!matchStridedSliceMask(slice, /*shrink=*/2, /*beginMask=*/1,
                               /*endMask=*/1))
      return false;
    SmallVector<int64_t, 2> strides;
    if (!constInt1D(slice.getOperand(3), strides) ||
        strides != SmallVector<int64_t, 2>{1, 1})
      return false;
    begin = slice.getOperand(1);
    auto beginTy = dyn_cast<RankedTensorType>(begin.getType());
    if (!beginTy || !beginTy.getElementType().isInteger(32) ||
        beginTy.getRank() != 1 || beginTy.getShape()[0] != 2)
      return false;
    keys = slice.getOperand(0);
    auto keysTy = dyn_cast<RankedTensorType>(keys.getType());
    if (!keysTy || !keysTy.getElementType().isInteger(64) ||
        keysTy.getRank() != 2)
      return false;
    return true;
  }
  // set_dim_v2(get_dim_v2(dynamic_reshape(real_dynamic_slice(keys, begin,
  // limit, strides), shape)), [-1,1])
  if (auto setDim =
          dyn_cast_or_null<atir::SetDimV2Op>(segCol.getDefiningOp())) {
    SmallVector<int64_t, 2> targetShape;
    if (!constInt1D(setDim.getOperand(1), targetShape) ||
        targetShape != SmallVector<int64_t, 2>{-1, 1})
      return false;
    auto getDim = dyn_cast_or_null<atir::GetDimV2Op>(
        setDim.getOperand(0).getDefiningOp());
    if (!getDim) return false;
    auto dynReshape = dyn_cast_or_null<stablehlo::DynamicReshapeOp>(
        getDim.getOperand().getDefiningOp());
    if (!dynReshape) return false;
    auto realSlice = dyn_cast_or_null<stablehlo::RealDynamicSliceOp>(
        dynReshape.getOperand().getDefiningOp());
    if (!realSlice) return false;
    SmallVector<int64_t, 2> strides;
    if (!constInt1D(realSlice.getStrides(), strides) ||
        strides != SmallVector<int64_t, 2>{1, 1})
      return false;
    keys = realSlice.getOperand();
    auto keysTy = dyn_cast<RankedTensorType>(keys.getType());
    if (!keysTy || !keysTy.getElementType().isInteger(64) ||
        keysTy.getRank() != 2)
      return false;
    begin = realSlice.getStartIndices();
    auto beginTy = dyn_cast<RankedTensorType>(begin.getType());
    if (!beginTy || !beginTy.getElementType().isInteger(32) ||
        beginTy.getRank() != 1 || beginTy.getShape()[0] != 2)
      return false;
    if (auto bs = realSlice->getAttrOfType<StringAttr>("annc.begin_source"))
      beginSourceName = bs.getValue().str();
    return true;
  }
  {
    Value sliceVal = segCol;
    if (auto reshape =
            dyn_cast_or_null<stablehlo::ReshapeOp>(segCol.getDefiningOp()))
      sliceVal = reshape.getOperand();
    Operation *sliceOp = sliceVal.getDefiningOp();
    if (auto slice = dyn_cast_or_null<atir::UnsupportedOp>(sliceOp)) {
      if (slice.getOpType() != "StridedSlice" || slice.getNumOperands() != 4)
        return false;
      if (!matchStridedSliceMask(slice, /*shrink=*/2, /*beginMask=*/1,
                                 /*endMask=*/1))
        return false;
      SmallVector<int64_t, 2> strides;
      if (!constInt1D(slice.getOperand(3), strides) ||
          strides != SmallVector<int64_t, 2>{1, 1})
        return false;
      begin = slice.getOperand(1);
      auto beginTy = dyn_cast<RankedTensorType>(begin.getType());
      if (!beginTy || !beginTy.getElementType().isInteger(32) ||
          beginTy.getRank() != 1 || beginTy.getShape()[0] != 2)
        return false;
      keys = slice.getOperand(0);
      auto keysTy = dyn_cast<RankedTensorType>(keys.getType());
      if (!keysTy || !keysTy.getElementType().isInteger(64) ||
          keysTy.getRank() != 2)
        return false;
      return true;
    }
    auto st = dyn_cast_or_null<stablehlo::SliceOp>(sliceOp);
    if (!st) return false;
    keys = st.getOperand();
    auto srcTy = dyn_cast<RankedTensorType>(keys.getType());
    auto starts = st.getStartIndices();
    auto limits = st.getLimitIndices();
    auto strides = st.getStrides();
    if (!srcTy || srcTy.getRank() != 2 || starts.size() != 2 ||
        limits.size() != 2 || strides.size() != 2)
      return false;
    if (starts[0] != 0 || strides[0] != 1) return false;
    if (!srcTy.isDynamicDim(0) && limits[0] != srcTy.getDimSize(0))
      return false;
    if (strides[1] != 1 || limits[1] - starts[1] != 1) return false;
    if (starts[1] != 0) return false;
    if (auto bs = st->getAttrOfType<StringAttr>("annc.begin_source"))
      beginSourceName = bs.getValue().str();
    if (beginSourceName.empty()) return false;
    auto keysTy = dyn_cast<RankedTensorType>(keys.getType());
    if (!keysTy || !keysTy.getElementType().isInteger(64) ||
        keysTy.getRank() != 2)
      return false;
    begin = Value();
    return true;
  }
  return false;
}

static bool matchInShapeAB(Value inShape, stablehlo::ConstantOp &packSrcConst) {
  auto convert = inShape.getDefiningOp<stablehlo::ConvertOp>();
  if (!convert) return false;
  if (auto indexCast =
          convert.getOperand().getDefiningOp<arith::IndexCastOp>()) {
    if (auto shapeOf =
            indexCast.getOperand().getDefiningOp<shape::ShapeOfOp>()) {
      (void)shapeOf;
      packSrcConst = nullptr;
      return true;
    }
  }
  auto concat = convert.getOperand().getDefiningOp<stablehlo::ConcatenateOp>();
  if (!concat || concat.getDimension() != 0 || concat.getNumOperands() != 2)
    return false;
  auto rowSrcTy = dyn_cast<RankedTensorType>(concat.getOperand(0).getType());
  if (!rowSrcTy || rowSrcTy.getRank() != 1 ||
      !rowSrcTy.getElementType().isInteger())
    return false;
  auto pTy = dyn_cast<RankedTensorType>(concat.getOperand(1).getType());
  if (!pTy || pTy.getRank() != 1 || !pTy.getElementType().isInteger())
    return false;
  auto packOp = concat.getOperand(1).getDefiningOp();
  if (!packOp) return false;
  Value packVal = concat.getOperand(1);
  for (int hop = 0; hop < 4; ++hop) {
    Operation *def = packVal.getDefiningOp();
    if (!def || def->getNumOperands() != 1 || isa<stablehlo::ConstantOp>(def))
      break;
    packVal = def->getOperand(0);
  }
  if (auto c = packVal.getDefiningOp<stablehlo::ConstantOp>()) packSrcConst = c;
  return true;
}


LogicalResult matchKpSparseReshape(atir::SparseReshapeV2Op sr,
                                   SparseReshapeMatch &m) {
  SmallPtrSet<Operation *, 32> closure;
  m.sr = sr;
  auto concat = sr.getIndices().getDefiningOp<stablehlo::ConcatenateOp>();
  if (!concat || concat.getDimension() != 1 || concat.getNumOperands() != 2)
    return failure();
  Value keys, begin;
  if (!matchSegColumn(concat.getOperand(1), keys, begin, m.beginSourceName))
    return failure();
  stablehlo::ConstantOp packSrcConst;
  if (!matchInShapeAB(sr.getInputShape(), packSrcConst)) return failure();
  Value newShape = sr.getNewShape();
  auto nTy = dyn_cast<RankedTensorType>(newShape.getType());
  if (!nTy || !nTy.getElementType().isInteger(64) || nTy.getRank() != 1 ||
      nTy.getShape()[0] != 2)
    return failure();
  if (auto newDef = newShape.getDefiningOp<stablehlo::ConstantOp>())
    m.newShapeConst = newShape;
  else if (auto arg = llvm::dyn_cast<BlockArgument>(newShape))
    (void)arg;
  else
    return failure();
  m.keys = keys;
  m.beginConst = begin;
  m.packSrcConst = packSrcConst;

  auto func = sr->getParentOfType<func::FuncOp>();
  for (Operation &op : func.getBody().front().getOperations())
    if (closure.contains(&op)) m.members.push_back(&op);
  return success();
}

class FuseKpSparseReshapeV2Pattern
    : public OpRewritePattern<atir::SparseReshapeV2Op> {
 public:
  FuseKpSparseReshapeV2Pattern(MLIRContext *ctx,
                               fusionv2::CallPortTable *portTableArg)
      : OpRewritePattern(ctx), portTable(portTableArg) {
    assert(portTable && "requires a port table");
  }

  LogicalResult matchAndRewrite(atir::SparseReshapeV2Op anchor,
                                PatternRewriter &rewriter) const override {
    SparseReshapeMatch m;
    if (failed(matchKpSparseReshape(anchor, m))) return failure();

    func::FuncOp main = anchor->getParentOfType<func::FuncOp>();

    SmallVector<std::pair<Value, std::string>> promoted;

    SmallVector<Operation *, 4> abSynth;
    stablehlo::ConstantOp synthPack;
    bool abRewired = false;
    auto cleanup = [&]() {
      if (abRewired)
        rewriter.modifyOpInPlace(m.sr.getOperation(), [&] {
          m.sr.getOperation()->setOperand(1, m.savedInShape);
        });
      for (Operation *op : llvm::reverse(abSynth)) rewriter.eraseOp(op);
      abSynth.clear();
      abRewired = false;
    };
    {
      std::string anchorName = fusionv2::anchorFromLoc(anchor.getLoc());
      rewriter.setInsertionPoint(anchor.getOperation());
      std::string packName;
      if (m.packSrcConst) {
        auto srcTy = cast<RankedTensorType>(m.packSrcConst.getType());
        auto synthTy = RankedTensorType::get({}, srcTy.getElementType());
        std::string src;
        if (auto convert = dyn_cast_or_null<stablehlo::ConvertOp>(
                m.sr.getInputShape().getDefiningOp())) {
          if (auto concat = dyn_cast_or_null<stablehlo::ConcatenateOp>(
                  convert.getOperand().getDefiningOp())) {
            if (auto s = concat->getAttrOfType<StringAttr>("annc.const_source"))
              src = s.getValue().str();
          }
        }
        if (src.empty()) {
          if (auto s = m.packSrcConst->getAttrOfType<StringAttr>(
                  "annc.const_source"))
            src = s.getValue().str();
        }
        if (src.empty()) return failure();
        packName = src;
        auto packAttr = m.packSrcConst.getValue().cast<DenseElementsAttr>();
        if (!packAttr.isSplat()) {
          // A multi-element vector packs no single width; reject instead
          // of tripping the splat assertion.
          return failure();
        }
        // The registry only defines the I64 builtin (the Customize
        // rewrite rejects i32 packs), so fail closed on the source type.
        if (!srcTy.getElementType().isInteger(64)) return failure();
        synthPack = rewriter.create<stablehlo::ConstantOp>(
            NameLoc::get(rewriter.getStringAttr(packName)),
            DenseElementsAttr::get(synthTy, packAttr.getSplatValue<APInt>()));
      } else {
        return failure();
      }
      abSynth.push_back(synthPack.getOperation());
      promoted.emplace_back(synthPack.getResult(), packName);
      m.packConst = synthPack.getResult();
      std::string keysName =
          fusionv2::abiNameFor(main, m.keys, anchorName, "_indices");
      promoted.emplace_back(m.keys, keysName);
      m.savedInShape = m.sr.getInputShape();
      rewriter.modifyOpInPlace(m.sr.getOperation(), [&] {
        m.sr.getOperation()->setOperand(0, m.keys);
        m.sr.getOperation()->setOperand(1, m.sr.getNewShape());
      });
      abRewired = true;
    }

    if (!m.newShapeConst) {
      auto newArg = llvm::dyn_cast<BlockArgument>(m.sr.getNewShape());
      SmallVector<std::string> entryInputs = entryInputNames(main);
      if (!newArg || newArg.getOwner()->getParentOp() != main ||
          newArg.getArgNumber() >= entryInputs.size()) {
        cleanup();
        return failure();
      }
      promoted.emplace_back(m.sr.getNewShape(),
                            entryInputs[newArg.getArgNumber()]);
    }

    if (m.beginConst) {
      const std::string anchorName = fusionv2::anchorFromLoc(anchor.getLoc());
      std::string beginName = m.beginSourceName;
      if (beginName.empty()) {
        Operation *beginDef = m.beginConst.getDefiningOp();
        if (beginDef) {
          if (auto bs =
                  beginDef->getAttrOfType<StringAttr>("annc.begin_source"))
            beginName = bs.getValue().str();
          if (beginName.empty())
            beginName = fusionv2::anchorFromLoc(beginDef->getLoc());
        }
      }
      if (beginName.empty())
        beginName =
            fusionv2::abiNameFor(main, m.beginConst, anchorName, "_begin");
      promoted.emplace_back(m.beginConst, beginName);
    } else if (!m.beginSourceName.empty()) {
      auto begin = rewriter.create<stablehlo::ConstantOp>(
          NameLoc::get(rewriter.getStringAttr(m.beginSourceName)),
          DenseElementsAttr::get(
              RankedTensorType::get({2}, rewriter.getI32Type()),
              ArrayRef<int32_t>{0, 0}));
      abSynth.push_back(begin.getOperation());
      promoted.emplace_back(begin.getResult(), m.beginSourceName);
    }
    if (m.newShapeConst) {
      Operation *newShapeDef = m.newShapeConst.getDefiningOp();
      std::string newName;
      if (auto src = newShapeDef ? newShapeDef->getAttrOfType<StringAttr>(
                                       "annc.const_source")
                                 : nullptr)
        newName = src.getValue().str();
      if (newName.empty() && newShapeDef) {
        newName = fusionv2::anchorFromLoc(newShapeDef->getLoc());
        std::string srAnchor = fusionv2::anchorFromLoc(m.sr.getLoc());
        if (!newName.empty() && newName == srAnchor)
          newName.clear();
      }
      // A synthesized fallback name resolves to no pb node and the
      // converter drops the whole site, so fail closed instead.
      if (newName.empty()) {
        cleanup();
        return failure();
      }
      promoted.emplace_back(m.newShapeConst, newName);
    }

    Operation *tailOps[] = {anchor.getOperation()};
    Value outputs[] = {m.sr.getOutputIndices(), m.sr.getOutputShape()};
    SmallVector<ProbeArg> args;
    if (failed(collectBoundary(main, m.members, args, promoted))) {
      cleanup();
      return failure();
    }
    SmallVector<std::string> tailAnchors;
    for (auto [idx, out] : llvm::enumerate(outputs)) {
      std::string a = fusionv2::anchorFromLoc(out.getLoc());
      tailAnchors.push_back(a.empty() ? a : a + ":" + std::to_string(idx));
    }

    if (args.size() != 4) {
      ANNC_LOG_DEBUG("op-fusion-v2")
          << "kp_sparse_reshape: args=" << args.size()
          << " (want 4); dropped\n";
      for (const ProbeArg &a : args)
        ANNC_LOG_DEBUG("op-fusion-v2") << "  - " << a.tfName << "\n";
      cleanup();
      return failure();
    }
    SmallVector<ProbeArg> ordered;
    auto take = [&](std::function<bool(const ProbeArg &)> pred) {
      for (auto it = args.begin(); it != args.end(); ++it) {
        if (!pred(*it)) continue;
        ordered.push_back(*it);
        args.erase(it);
        return true;
      }
      return false;
    };
    auto rankOf = [](const ProbeArg &a, unsigned bits, int rank,
                     int dim0 = -1) {
      return a.type.getRank() == rank &&
             a.type.getElementType().isInteger(bits) &&
             (dim0 < 0 || a.type.getShape()[0] == dim0);
    };
    if (!take([&](const ProbeArg &a) { return rankOf(a, 64, 2); })) {  // keys
      cleanup();
      return failure();
    }
    if (!take(
            [&](const ProbeArg &a) { return rankOf(a, 32, 1, 2); })) {  // begin
      cleanup();
      return failure();
    }
    if (!take([&](const ProbeArg &a) {
          if (a.type.getRank() == 0) return true;
          return a.type.getRank() == 1 &&
                 a.type.getElementType().isInteger(64) &&
                 a.type.getShape()[0] != 2;
        })) {  // pack_const
      cleanup();
      return failure();
    }
    if (!take([&](const ProbeArg &a) {
          return rankOf(a, 64, 1, 2);
        })) {  // new_shape
      cleanup();
      return failure();
    }
    args.assign(ordered.begin(), ordered.end());

    const bool isI64 = m.packConst.getType()
                           .cast<RankedTensorType>()
                           .getElementType()
                           .isInteger(64);
    auto module = anchor->getParentOfType<mlir::ModuleOp>();
    std::string name = uniquifySymbolName(
        module, std::string("KPFusedSparseReshape") +
                    (isI64 ? "I64" : "I32"));
    if (failed(outlineSite(main, rewriter, args, m.members, tailOps, outputs,
                           name, "kp_sparse_reshape", tailAnchors, {},
                           *portTable))) {
      cleanup();
      return failure();
    }
    return success();
  }

 private:
  fusionv2::CallPortTable *portTable;
};

}  // namespace
}  // namespace atir

void atir::addSparseReshapeV2Pattern(RewritePatternSet &patterns,
                               fusionv2::CallPortTable *portTable) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<FuseKpSparseReshapeV2Pattern>(ctx, portTable);
}
