#include <optional>
#include <string>

#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/FusionV2/CarrierMatchers.h"
#include "Dialect/Atir/Passes/FusionV2/FusionV2Outline.h"
#include "Dialect/Atir/Passes/FusionV2/Naming.h"
#include "Dialect/Atir/Passes/FusionV2/Patterns.h"
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

struct SparseSelectMatch {
  fusionv2::ReshapeLike reshapeA, reshapeB, reshapeC;
  Value a, b, c;
  Value gt, eq1, eq2, eq3;
  Value x, y, w;
  SmallVector<Operation *, 32> members;
};

Value peelBroadcast(Value v) {
  while (auto bcast = v.getDefiningOp<stablehlo::DynamicBroadcastInDimOp>())
    v = bcast.getOperand();
  return v;
}

Value scalarI32(Value v) {
  if (!v) return Value();
  v = peelBroadcast(fusionv2::peelAssumingYield(v));
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  if (!ty || ty.getRank() != 0 || !ty.getElementType().isSignlessInteger(32))
    return Value();
  Operation *def = v.getDefiningOp();
  if (!def || isa<stablehlo::ConstantOp>(def)) return v;
  return Value();
}

bool matchCompareAgainstColumn(Value condValue,
                               stablehlo::ComparisonDirection dir,
                               fusionv2::ReshapeLike &column, Value &scalar) {
  if (!condValue) return false;
  auto cmp = fusionv2::peelAssumingYield(condValue)
                 .getDefiningOp<stablehlo::CompareOp>();
  if (!cmp || cmp.getComparisonDirection() != dir) return false;
  fusionv2::ReshapeLike lhs =
      fusionv2::ReshapeLike::match(cmp.getLhs().getDefiningOp());
  fusionv2::ReshapeLike rhs =
      fusionv2::ReshapeLike::match(cmp.getRhs().getDefiningOp());
  if (dir == stablehlo::ComparisonDirection::GT) {
    if (!lhs || rhs) return false;
    column = lhs;
    scalar = scalarI32(cmp.getRhs());
  } else if (lhs && !rhs) {
    column = lhs;
    scalar = scalarI32(cmp.getRhs());
  } else if (rhs && !lhs) {
    column = rhs;
    scalar = scalarI32(cmp.getLhs());
  } else {
    return false;
  }
  if (!scalar) return false;
  auto resTy = dyn_cast<RankedTensorType>(column.result().getType());
  return resTy && resTy.getRank() == 2 &&
         (resTy.isDynamicDim(1) || resTy.getDimSize(1) == 1);
}

Value matchFillOne(Value v) {
  Value src = peelBroadcast(fusionv2::peelAssumingYield(v));
  auto cst = src.getDefiningOp<stablehlo::ConstantOp>();
  if (!cst) return Value();
  auto attr = dyn_cast<DenseElementsAttr>(cst.getValue());
  if (!attr || !attr.isSplat() || !attr.getElementType().isF32())
    return Value();
  if (attr.getSplatValue<APFloat>().convertToDouble() != 1.0) return Value();
  return src;
}

bool matchFillLikeBranch(Value v, Value &fillValue) {
  Value peeled = fusionv2::peelAssumingYield(v);
  if (Value f = matchFillOne(peeled)) {
    fillValue = f;
    return true;
  }
  auto sel = peeled.getDefiningOp<stablehlo::SelectOp>();
  if (!sel) return false;
  if (!matchFillOne(sel.getOnTrue()) || !matchFillOne(sel.getOnFalse()))
    return false;
  fillValue = matchFillOne(sel.getOnTrue());
  return true;
}

Value traceShapeOfSource(Value v, int hops = 0) {
  if (hops > 32) return Value();
  if (auto shapeOf = v.getDefiningOp<shape::ShapeOfOp>())
    return shapeOf.getOperand();
  Operation *def = v.getDefiningOp();
  if (!def || def->getNumOperands() == 0) return Value();
  return traceShapeOfSource(def->getOperand(0), hops + 1);
}

stablehlo::CompareOp findEqCompareUser(Value column) {
  stablehlo::CompareOp found;
  for (Operation *user : column.getUsers()) {
    auto cmp = dyn_cast<stablehlo::CompareOp>(user);
    if (!cmp ||
        cmp.getComparisonDirection() != stablehlo::ComparisonDirection::EQ)
      continue;
    if (found) return nullptr;
    found = cmp;
  }
  return found;
}

void collectMembers(Value root, SmallPtrSetImpl<Operation *> &seen,
                    SmallVectorImpl<Operation *> &out) {
  fusionv2::collectDataMembers(root, seen, out);
}

LogicalResult matchKpSparseSelect(Operation *anchorOp, SparseSelectMatch &m) {
  fusionv2::ConcatLike concat = fusionv2::ConcatLike::match(anchorOp);
  if (!concat) return failure();
  auto dim = concat.dimension();
  if (!dim || *dim != 1) return failure();
  SmallVector<Value, 4> operands = concat.inputs();
  if (operands.size() != 2 && operands.size() != 3) return failure();
  if (operands.size() == 3) {
    auto ty = dyn_cast<RankedTensorType>(operands[2].getType());
    if (!ty || ty.getRank() != 2 || ty.isDynamicDim(1) ||
        ty.getDimSize(1) != 0 ||
        !operands[2].getDefiningOp<stablehlo::ConstantOp>())
      return failure();
  }
  auto wTy = dyn_cast<RankedTensorType>(concat.result().getType());
  if (!wTy || wTy.getRank() != 2 || !wTy.getElementType().isF32())
    return failure();

  Value yValue = operands[0];
  auto sel0 =
      fusionv2::peelAssumingYield(yValue).getDefiningOp<stablehlo::SelectOp>();
  if (!sel0) return failure();
  if (!matchFillOne(sel0.getOnTrue())) return failure();
  if (!matchCompareAgainstColumn(sel0.getPred(),
                                 stablehlo::ComparisonDirection::EQ, m.reshapeB,
                                 m.eq2))
    return failure();

  auto sel1 = fusionv2::peelAssumingYield(sel0.getOnFalse())
                  .getDefiningOp<stablehlo::SelectOp>();
  if (!sel1) return failure();
  if (!matchFillOne(sel1.getOnTrue())) return failure();
  fusionv2::ReshapeLike reshapeB1;
  if (!matchCompareAgainstColumn(
          sel1.getPred(), stablehlo::ComparisonDirection::EQ, reshapeB1, m.eq1))
    return failure();
  if (reshapeB1.op != m.reshapeB.op) return failure();

  Value gtValue = fusionv2::peelAssumingYield(sel1.getOnFalse());
  if (auto convert = gtValue.getDefiningOp<stablehlo::ConvertOp>()) {
    auto castTy = dyn_cast<RankedTensorType>(convert.getType());
    if (!castTy || !castTy.getElementType().isF32()) return failure();
    gtValue = convert.getOperand();
  }
  if (!matchCompareAgainstColumn(gtValue, stablehlo::ComparisonDirection::GT,
                                 m.reshapeA, m.gt))
    return failure();

  Value cValue = operands[1];
  Value fillValue;
  if (auto csel = fusionv2::peelAssumingYield(cValue)
                      .getDefiningOp<stablehlo::SelectOp>()) {
    if (!matchCompareAgainstColumn(csel.getPred(),
                                   stablehlo::ComparisonDirection::EQ,
                                   m.reshapeC, m.eq3))
      return failure();
    if (!matchFillLikeBranch(csel.getOnTrue(), fillValue) &&
        !matchFillLikeBranch(csel.getOnFalse(), fillValue))
      return failure();
  } else {
    if (!matchFillLikeBranch(cValue, fillValue)) return failure();
    auto bcast = fusionv2::peelAssumingYield(cValue)
                     .getDefiningOp<stablehlo::DynamicBroadcastInDimOp>();
    Value column =
        bcast ? traceShapeOfSource(bcast.getOutputDimensions()) : Value();
    if (!column) return failure();
    m.reshapeC = fusionv2::ReshapeLike::match(column.getDefiningOp());
    if (!m.reshapeC) return failure();
    auto cmp = findEqCompareUser(column);
    if (!cmp) return failure();
    fusionv2::ReshapeLike columnAgain;
    if (!matchCompareAgainstColumn(cmp.getResult(),
                                   stablehlo::ComparisonDirection::EQ,
                                   columnAgain, m.eq3) ||
        columnAgain.op != m.reshapeC.op)
      return failure();
  }

  m.a = m.reshapeA.src();
  m.b = m.reshapeB.src();
  m.c = m.reshapeC.src();
  if (m.a == m.b || m.a == m.c || m.b == m.c) return failure();
  for (Value v : {m.a, m.b, m.c}) {
    auto ty = dyn_cast<RankedTensorType>(v.getType());
    if (!ty || ty.getRank() != 1 || !ty.getElementType().isSignlessInteger(32))
      return failure();
  }
  auto xTy = dyn_cast<RankedTensorType>(m.reshapeA.result().getType());
  auto yTy = dyn_cast<RankedTensorType>(yValue.getType());
  if (!xTy || xTy.getRank() != 2 || !xTy.getElementType().isSignlessInteger(32))
    return failure();
  if (!yTy || yTy.getRank() != 2 || !yTy.getElementType().isF32())
    return failure();

  m.x = m.reshapeA.result();
  m.y = yValue;
  m.w = concat.result();

  SmallPtrSet<Operation *, 32> seen;
  SmallVector<Value> roots = {m.x, m.y, m.w, operands[1]};
  if (fillValue) roots.push_back(fillValue);
  for (Value root : roots) collectMembers(root, seen, m.members);
  llvm::sort(m.members, [&](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  if (m.members.empty()) return failure();
  if (!llvm::is_contained(m.members, anchorOp)) m.members.push_back(anchorOp);

  SmallPtrSet<Operation *, 32> memberSet(m.members.begin(), m.members.end());
  auto insideSite = [&](Operation *user) {
    for (Operation *anc = user; anc != nullptr; anc = anc->getParentOp())
      if (memberSet.contains(anc)) return true;
    return false;
  };
  for (Operation *op : m.members) {
    for (Value r : op->getResults()) {
      if (r == m.x || r == m.y || r == m.w) continue;
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

static LogicalResult fuseKpSparseSelect(Operation *anchorOp,
                                        PatternRewriter &rewriter,
                                        fusionv2::CallPortTable *portTable) {
  SparseSelectMatch m;
  if (failed(matchKpSparseSelect(anchorOp, m))) return failure();

  func::FuncOp main = anchorOp->getParentOfType<func::FuncOp>();
  std::string anchorName = fusionv2::anchorFromLoc(anchorOp->getLoc());
  if (anchorName.empty()) return failure();

  SmallVector<std::pair<Value, std::string>> promoted;
  auto promoteConst = [&](Value v, llvm::StringRef field) {
    Operation *def = v.getDefiningOp();
    if (!def || !isa<stablehlo::ConstantOp>(def)) return;
    std::string name;
    if (auto src = def->getAttrOfType<StringAttr>("annc.const_source"))
      name = src.getValue().str();
    if (name.empty()) name = anchorName + "/" + field.str();
    promoted.emplace_back(v, name);
  };
  promoteConst(m.gt, "gt");
  promoteConst(m.eq1, "eq1");
  promoteConst(m.eq2, "eq2");
  promoteConst(m.eq3, "eq3");

  Operation *tailOpsArr[] = {m.x.getDefiningOp(), m.y.getDefiningOp(),
                             anchorOp};
  SmallVector<Operation *, 3> tailOps(tailOpsArr, tailOpsArr + 3);
  Value outputsArr[] = {m.x, m.y, m.w};
  SmallVector<Value, 3> outputs(outputsArr, outputsArr + 3);
  SmallVector<std::string> tailAnchors;
  for (Value out : outputs) {
    OpResult res = cast<OpResult>(out);
    tailAnchors.push_back(fusionv2::anchorFromLoc(res.getOwner()->getLoc()) +
                          ":" + std::to_string(res.getResultNumber()));
  }

  SmallVector<ProbeArg> args;
  if (failed(collectBoundary(main, m.members, args, promoted)))
    return failure();

  if (args.size() != 7) {
    ANNC_LOG_DEBUG("op-fusion-v2") << "kp_sparse_select: args=" << args.size()
                 << " (want 7); dropped\n";
    for (const ProbeArg &a : args)
      ANNC_LOG_DEBUG("op-fusion-v2") << "  - " << a.tfName << "\n";
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
  for (Value v : {m.a, m.b, m.c, m.gt, m.eq1, m.eq2, m.eq3})
    if (!pick(v)) return failure();
  args.assign(ordered.begin(), ordered.end());

  auto module = anchorOp->getParentOfType<mlir::ModuleOp>();
  std::string name = uniquifySymbolName(module, "KPFusedSparseSelect");
  if (failed(outlineSite(main, rewriter, args, m.members, tailOps, outputs,
                         name, "kp_sparse_select", tailAnchors, {},
                         *portTable)))
    return failure();
  return success();
}

class FuseKpSparseSelectV2Pattern
    : public OpRewritePattern<stablehlo::ConcatenateOp> {
 public:
  FuseKpSparseSelectV2Pattern(MLIRContext *ctx,
                              fusionv2::CallPortTable *portTableArg)
      : OpRewritePattern(ctx), portTable(portTableArg) {
    assert(portTable && "requires a port table");
  }

  LogicalResult matchAndRewrite(stablehlo::ConcatenateOp anchor,
                                PatternRewriter &rewriter) const override {
    return fuseKpSparseSelect(anchor.getOperation(), rewriter, portTable);
  }

 private:
  fusionv2::CallPortTable *portTable;
};

class FuseKpSparseSelectResiduePattern
    : public OpRewritePattern<atir::UnsupportedOp> {
 public:
  FuseKpSparseSelectResiduePattern(MLIRContext *ctx,
                                   fusionv2::CallPortTable *portTableArg)
      : OpRewritePattern(ctx), portTable(portTableArg) {
    assert(portTable && "requires a port table");
  }

  LogicalResult matchAndRewrite(atir::UnsupportedOp anchor,
                                PatternRewriter &rewriter) const override {
    if (anchor.getOpType() != "ConcatV2") return failure();
    return fuseKpSparseSelect(anchor.getOperation(), rewriter, portTable);
  }

 private:
  fusionv2::CallPortTable *portTable;
};

}  // namespace
}  // namespace atir

void atir::addSparseSelectV2Pattern(RewritePatternSet &patterns,
                                    fusionv2::CallPortTable *portTable) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<FuseKpSparseSelectV2Pattern>(ctx, portTable);
  patterns.add<FuseKpSparseSelectResiduePattern>(ctx, portTable);
}
