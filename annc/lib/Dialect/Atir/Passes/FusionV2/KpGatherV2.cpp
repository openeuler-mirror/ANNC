#include <string>

#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/FusionV2/CarrierMatchers.h"
#include "Dialect/Atir/Passes/FusionV2/FusionV2Outline.h"
#include "Dialect/Atir/Passes/FusionV2/Naming.h"
#include "Dialect/Atir/Passes/FusionV2/Provenance.h"
#include "Support/Log.h"
#include "llvm/Support/raw_ostream.h"
#include "Dialect/Atir/Passes/FusionV2/Patterns.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace atir::fusionv2;

namespace atir {
namespace {

// Keys column flatten / residue column select feeding the 1-D Unique
// input; fills src (keys2d) and the StridedSlice-form begin const with
// its pb name (the kernel reads begin[1]; masks live in metadata).
static bool isColumnFlatten(Operation *op, Value &src, Value &beginConst,
                            std::string &beginSrc) {
  beginConst = nullptr;
  beginSrc.clear();
  if (!op) return false;
  if (auto reshape = dyn_cast<stablehlo::ReshapeOp>(op)) {
    auto rt = dyn_cast<RankedTensorType>(reshape.getResult().getType());
    if (!rt || rt.getRank() != 1) return false;
    src = reshape.getOperand();
    return true;
  }
  if (op->getNumOperands() == 4) {
    auto residue = dyn_cast<atir::UnsupportedOp>(op);
    if (residue && residue.getOpType() == "StridedSlice") {
      auto md = residue->getAttrOfType<DictionaryAttr>("metadata");
      auto mask = [&](llvm::StringRef name) {
        auto a = md ? md.getAs<mlir::IntegerAttr>(name) : nullptr;
        return a ? a.getInt() : int64_t{-1};
      };
      if (mask("ellipsis_mask") != 0 || mask("new_axis_mask") != 0 ||
          mask("begin_mask") != 1 || mask("end_mask") != 1 ||
          mask("shrink_axis_mask") != 2)
        return false;
      auto inTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      if (!inTy || inTy.getRank() != 2) return false;
      auto beginOp =
          dyn_cast<stablehlo::ConstantOp>(op->getOperand(1).getDefiningOp());
      auto strideOp =
          dyn_cast<stablehlo::ConstantOp>(op->getOperand(3).getDefiningOp());
      if (!beginOp || !strideOp) return false;
      auto beginVal = dyn_cast<DenseIntElementsAttr>(beginOp.getValue());
      auto strideVal = dyn_cast<DenseIntElementsAttr>(strideOp.getValue());
      if (!beginVal || !strideVal) return false;
      auto vec2 = [](DenseIntElementsAttr a) {
        auto t = a.getType();
        return t.getRank() == 1 && t.getShape()[0] == 2 &&
               t.getElementType().isInteger(32);
      };
      if (!vec2(beginVal) || !vec2(strideVal)) return false;
      SmallVector<int64_t, 2> b, s;
      for (const APInt &p : beginVal.getValues<APInt>())
        b.push_back(p.getSExtValue());
      for (const APInt &p : strideVal.getValues<APInt>())
        s.push_back(p.getSExtValue());
      if (b.size() != 2 || s.size() != 2) return false;
      if (b[0] != 0 || s[0] != 1 || s[1] != 1) return false;
      beginSrc = fusionv2::sourceNameOf(op->getOperand(1));
      if (beginSrc.empty()) return false;
      src = op->getOperand(0);
      beginConst = op->getOperand(1);
      return true;
    }
    return false;
  }
  if (op->getNumOperands() != 2) return false;
  if (auto residue = dyn_cast<atir::UnsupportedOp>(op)) {
    if (residue.getOpType() != "Reshape") return false;
  } else if (!isa<stablehlo::DynamicReshapeOp>(op)) {
    return false;
  }
  auto ty = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!ty || ty.getRank() != 1 || ty.getShape()[0] != 1 ||
      !ty.getElementType().isInteger())
    return false;
  src = op->getOperand(0);
  return true;
}

struct GatherMatch {
  UniqueLike outer;             // outer Unique (result-replacement target)
  UniqueLike unique;            // the fused Unique (anchor / inner)
  bool doubleLayer = false;     // outer feeds on inner (two layers chained)
  bool hasColumn = false;       // column known (begin / static slice attr)
  int64_t beginCol = 0;         // static-slice column (for synthesized begin)
  Value keys2d;                 // ABI keys (rank2 i64)
  Value table;                  // ABI data (rank2 f32)
  Value beginConst;             // double: the slice's real begin constant
  std::string beginSourceName;  // double: begin's pb node name
  GatherLike gather;            // members: table[uniq#0]
  GatherLike restore;           // stays in main: gathered[uniq#1]
  Value gatherOut, uniqValues, uniqIndices;
};

// Fail-closed kp_gather match: single (flatten) or double layer (inner
// Unique, outer re-uniques idempotently); keys2d rank2 i64 needs dim1==1
// unless a begin selects the column (static slice: col 0 + begin_source).
// The indices slot takes only the anchor group's port-1 (annc.source_ports)
// value - expansion intermediates are indistinguishable, so no tag = drop.
LogicalResult matchKpGather(Operation *anchorOp, GatherMatch &m) {
  UniqueLike outer = UniqueLike::match(anchorOp);
  if (!outer) return failure();
  m.outer = outer;

  UniqueLike fused = outer;
  Operation *flatten = outer.input.getDefiningOp();
  Value keys2d, beginConst;
  std::string beginSource;
  if (!isColumnFlatten(flatten, keys2d, beginConst, beginSource)) {
    UniqueLike inner = UniqueLike::match(flatten);
    if (!inner) {
      std::string locStr;
      llvm::raw_string_ostream(locStr) << anchorOp->getLoc();
      ANNC_LOG_DEBUG("gather")
          << locStr << ": inner-unique-miss flatten="
          << (flatten ? flatten->getName().getStringRef().str()
             : std::string("<null>"))
          << "\n";
      return failure();
    }
    flatten = inner.input.getDefiningOp();
    if (!isColumnFlatten(flatten, keys2d, beginConst, beginSource)) {
      std::string locStr;
      llvm::raw_string_ostream(locStr) << anchorOp->getLoc();
      ANNC_LOG_DEBUG("gather")
          << locStr << ": flatten-miss "
          << (flatten ? flatten->getName().getStringRef().str()
             : std::string("<null>"))
          << "\n";
      return failure();
    }
    if (beginConst) {
      m.beginConst = beginConst;
      m.beginSourceName = beginSource;
      m.hasColumn = true;
    } else {
      auto slice = SliceLike::match(keys2d.getDefiningOp());
      if (!slice) return failure();
      keys2d = slice.source();
      if (Value begin = slice.start()) {
        auto beginTy = dyn_cast<RankedTensorType>(begin.getType());
        if (!beginTy || beginTy.getRank() != 1 || beginTy.getShape()[0] != 2 ||
            !beginTy.getElementType().isInteger(32))
          return failure();
        if (!begin.getDefiningOp<stablehlo::ConstantOp>()) return failure();
        m.beginConst = begin;
        m.hasColumn = true;
        if (auto src = slice.op->getAttrOfType<StringAttr>("annc.begin_source"))
          m.beginSourceName = src.getValue().str();
      } else if (auto st = dyn_cast<stablehlo::SliceOp>(slice.op)) {
        auto srcTy = dyn_cast<RankedTensorType>(keys2d.getType());
        auto starts = st.getStartIndices();
        auto limits = st.getLimitIndices();
        auto strides = st.getStrides();
        if (!srcTy || srcTy.getRank() != 2 || starts.size() != 2 ||
            limits.size() != 2 || strides.size() != 2)
          return failure();
        if (starts[0] != 0 || strides[0] != 1) return failure();
        if (!srcTy.isDynamicDim(0) && limits[0] != srcTy.getDimSize(0))
          return failure();
        if (strides[1] != 1 || limits[1] - starts[1] != 1) return failure();
        if (starts[1] != 0) return failure();
        if (auto src = slice.op->getAttrOfType<StringAttr>("annc.begin_source"))
          m.beginSourceName = src.getValue().str();
        if (m.beginSourceName.empty()) return failure();
        m.beginCol = starts[1];
        m.hasColumn = true;
      } else {
        return failure();
      }
    }
    fused = inner;
    m.doubleLayer = true;
  }
  const bool realIndices = static_cast<bool>(outer.indices);
  if (beginConst && !m.doubleLayer) {
    m.beginConst = beginConst;
    m.beginSourceName = beginSource;
    m.hasColumn = true;
  }
  m.keys2d = keys2d;

  auto kTy = dyn_cast<RankedTensorType>(m.keys2d.getType());
  if (!kTy || kTy.getRank() != 2 || !kTy.getElementType().isSignlessInteger(64))
    return failure();
  if (!m.hasColumn && kTy.getDimSize(1) != 1) return failure();

  if (!fused.residue) {
    SmallVector<Operation *, 8> chain;
    for (Value v = fused.input; v != m.keys2d;) {
      Operation *def = v.getDefiningOp();
      if (!def || def->getNumOperands() != 1) return failure();
      chain.push_back(def);
      v = def->getOperand(0);
    }
    if (chain.empty()) return failure();
    fused.members.insert(fused.members.begin(), chain.rbegin(), chain.rend());
  }

  auto soleUser = [&](Value v) -> Operation * {
    Operation *single = nullptr;
    for (Operation *u : v.getUsers()) {
      if (single) return nullptr;
      single = u;
    }
    return single;
  };
  auto gather = soleUser(outer.values);
  m.gather = GatherLike::match(gather);
  if (!m.gather || !m.gather.axisIsZero() || m.gather.indices() != outer.values)
    return failure();
  m.table = m.gather.params();
  auto tTy = dyn_cast<RankedTensorType>(m.table.getType());
  if (!tTy || tTy.getRank() != 2 || !tTy.getElementType().isF32())
    return failure();
  auto gTy = dyn_cast<RankedTensorType>(m.gather.result().getType());
  if (!gTy || gTy.getRank() != 2 || !gTy.getElementType().isF32())
    return failure();

  if (realIndices) {
    fused.indices = outer.indices;
    if (!outer.indices) return failure();
    auto restore = soleUser(outer.indices);
    m.restore = GatherLike::match(restore);
    if (!m.restore || m.restore.params() != m.gather.result() ||
        m.restore.indices() != outer.indices)
      return failure();
  } else {
    return failure();
  }
  const int64_t resultWidths[] = {64, 32};
  Value fusedResults[] = {fused.values, fused.indices};
  for (auto [idx, want] : llvm::enumerate(resultWidths)) {
    auto ty = dyn_cast<RankedTensorType>(fusedResults[idx].getType());
    if (!ty || ty.getRank() != 1 ||
        !ty.getElementType().isSignlessInteger(want))
      return failure();
  }

  m.unique = fused;
  m.uniqValues = fused.values;
  m.uniqIndices = fused.indices;
  m.gatherOut = m.gather.result();
  return success();
}

// Site fusion body for the residue and 1:N forms; rewires roll back on
// failure. ABI = (data, keys, begin) with a promoted or synthesized [0,0]
// begin; double layer re-binds gather indices to the inner unique_values.
// 1:N sites absorb the anchor group's cast/gte shell so the port-1 value
// becomes an in-site output; main keeps only the restore gather.
static LogicalResult fuseKpGather(Operation *anchorOp,
                                  PatternRewriter &rewriter,
                                  fusionv2::CallPortTable *portTable) {
  GatherMatch m;
  if (failed(matchKpGather(anchorOp, m))) return failure();
  func::FuncOp main = anchorOp->getParentOfType<func::FuncOp>();
  if (!main) return failure();

  SmallVector<std::pair<Value, std::string>> promoted;
  SmallVector<Operation *, 4> synth;
  struct Rewire {
    Operation *op;
    unsigned index;
    Value old;
  };
  SmallVector<Rewire, 8> rewires;
  Value savedGatherIdx;
  bool gatherRewired = false;
  auto cleanup = [&]() {
    for (auto &r : llvm::reverse(rewires))
      rewriter.modifyOpInPlace(r.op, [&] { r.op->setOperand(r.index, r.old); });
    if (gatherRewired)
      rewriter.modifyOpInPlace(
          m.gather.op, [&] { m.gather.op->setOperand(1, savedGatherIdx); });
    for (Operation *op : llvm::reverse(synth)) rewriter.eraseOp(op);
  };
  std::string anchorName = fusionv2::anchorFromLoc(anchorOp->getLoc());
  rewriter.setInsertionPoint(anchorOp);

  if (m.beginConst) {
    promoted.emplace_back(m.beginConst, m.beginSourceName.empty()
                                            ? anchorName + "_begin"
                                            : m.beginSourceName);
  } else {
    std::string beginName = m.beginSourceName;
    if (beginName.empty())
      beginName =
          (anchorName.empty() ? std::string("Unique") : anchorName) + "_begin";
    auto begin = rewriter.create<stablehlo::ConstantOp>(
        NameLoc::get(rewriter.getStringAttr(beginName)),
        DenseElementsAttr::get(
            RankedTensorType::get({2}, rewriter.getI32Type()),
            ArrayRef<int32_t>{0, static_cast<int32_t>(m.beginCol)}));
    synth.push_back(begin.getOperation());
    promoted.emplace_back(begin.getResult(), beginName);
  }

  auto promotedName = [&](Value v, const char *suffix) {
    promoted.emplace_back(v, fusionv2::abiNameFor(main, v, anchorName, suffix));
  };
  promotedName(m.keys2d, "_keys");
  promotedName(m.table, "_table");

  if (m.doubleLayer) {
    savedGatherIdx = m.gather.indices();
    rewriter.modifyOpInPlace(
        m.gather.op, [&] { m.gather.op->setOperand(1, m.unique.values); });
    gatherRewired = true;
  }

  if (m.unique.residue) {
    for (Operation *op : m.unique.members)
      for (unsigned i = 0; i < op->getNumOperands(); ++i)
        if (op->getOperand(i) == m.unique.input)
          rewires.push_back({op, i, m.unique.input});
    for (auto &r : rewires)
      rewriter.modifyOpInPlace(r.op,
                               [&] { r.op->setOperand(r.index, m.keys2d); });
  }

  SmallPtrSet<Operation *, 16> memberSet(m.unique.members.begin(),
                                         m.unique.members.end());
  memberSet.insert(m.outer.members.begin(), m.outer.members.end());
  memberSet.insert(m.gather.op);
  SmallVector<Operation *, 16> members;
  for (Operation &op : main.getBody().front().getOperations())
    if (memberSet.contains(&op)) members.push_back(&op);

  SmallVector<ProbeArg> args;
  if (failed(collectBoundary(main, members, args, promoted, {},
                           portTable))) {
    cleanup();
    return failure();
  }
  if (args.size() != 3) {
    cleanup();
    return failure();
  }

  SmallVector<ProbeArg> ordered;
  auto take = [&](auto &&pred) -> bool {
    for (size_t i = 0; i < args.size(); ++i) {
      if (pred(args[i])) {
        ordered.push_back(args[i]);
        args.erase(args.begin() + i);
        return true;
      }
    }
    return false;
  };
  if (!take([&](const ProbeArg &a) {
        return a.type.getRank() == 2 && a.type.getElementType().isF32();
      })) {
    cleanup();
    return failure();
  }
  if (!take([&](const ProbeArg &a) {
        return a.type.getRank() == 2 &&
               a.type.getElementType().isSignlessInteger(64);
      })) {
    cleanup();
    return failure();
  }
  if (!take([&](const ProbeArg &a) {
        return a.type.getRank() == 1 && a.type.getShape()[0] == 2 &&
               a.type.getElementType().isInteger(32);
      })) {
    cleanup();
    return failure();
  }
  args.assign(ordered.begin(), ordered.end());

  SmallVector<Operation *, 16> tailOps(m.unique.members.begin(),
                                       m.unique.members.end());
  tailOps.append(m.outer.members.begin(), m.outer.members.end());
  tailOps.push_back(m.gather.op);
  Value outputs[] = {m.uniqValues, m.uniqIndices, m.gatherOut};
  // DotGemmV2-style naming: anchor-derived hash is stable across runs in
  // long-lived processes; module uniquification guards collisions.
  ModuleOp module = main->getParentOfType<ModuleOp>();
  std::string base = fusionv2::sanitizeName("KPFusedGather_" + anchorName);
  std::string name =
      fusionv2::uniquifyFusionName(module, base + "_" + fusionv2::hash4(base));
  SmallVector<std::string> tailAnchors;
  auto portName = [&](Value out, int port) {
    std::string a = fusionv2::anchorFromLoc(out.getLoc());
    if (a.empty()) a = anchorName.empty() ? std::string("Unique") : anchorName;
    return a + ":" + std::to_string(port);
  };
  tailAnchors.push_back(portName(m.uniqValues, 0));
  tailAnchors.push_back(portName(m.uniqIndices, 1));
  tailAnchors.push_back(portName(m.gatherOut, 0));
  if (failed(outlineSite(main, rewriter, args, members, tailOps, outputs,
                         name, "kp_gather", tailAnchors, {}, *portTable))) {
    cleanup();
    return failure();
  }
  return success();
}

// Residue form: the anchor is atir.unsupported{opType="Unique"} itself.
class FuseKpGatherV2Pattern : public OpRewritePattern<atir::UnsupportedOp> {
 public:
  FuseKpGatherV2Pattern(MLIRContext *ctx, fusionv2::CallPortTable *portTable)
      : OpRewritePattern(ctx), portTable(portTable) {
    assert(portTable && "kp_gather requires the pass-owned port table");
  }

 private:
  fusionv2::CallPortTable *portTable;

  LogicalResult matchAndRewrite(atir::UnsupportedOp anchor,
                                PatternRewriter &rewriter) const override {
    return fuseKpGather(anchor.getOperation(), rewriter, portTable);
  }
};

// 1:N form: the anchor is a semantic group; only the member producing
// port 1 (indices) triggers once. The fn guard rejects outlined kernel
// bodies (source tags survive cloning - prevents kernel nesting).
class FuseKpGatherV2GroupPattern : public RewritePattern {
 public:
  FuseKpGatherV2GroupPattern(MLIRContext *ctx, PatternBenefit benefit,
                             fusionv2::CallPortTable *portTable)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, ctx),
        portTable(portTable) {
    assert(portTable && "kp_gather requires the pass-owned port table");
  }

 private:
  fusionv2::CallPortTable *portTable;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (fusionv2::semanticOf(op) != "tf.Unique") return failure();
    func::FuncOp fn = op->getParentOfType<func::FuncOp>();
    if (!fn || fn.getName().starts_with("KPFused") ||
        fn->hasAttr("annc.kernel"))
      return failure();
    auto indices = fusionv2::groupPortValue(op, 1);
    if (!indices) return failure();
    bool representative = false;
    for (Value r : op->getResults())
      if (r == *indices) representative = true;
    if (!representative) return failure();
    return fuseKpGather(op, rewriter, portTable);
  }
};

}  // namespace

// Registers the residue-anchor and 1:N group patterns; the group form gets
// benefit 1 over the residue default 0.
void addKpGatherV2Pattern(RewritePatternSet &patterns,
                          fusionv2::CallPortTable *portTable) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<FuseKpGatherV2Pattern>(ctx, portTable);
  patterns.add<FuseKpGatherV2GroupPattern>(ctx, PatternBenefit(1), portTable);
}
}  // namespace atir
