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

struct TbiMatch {
  Value batchInput, weight, bias, tileInput;
  Value w;
  SmallVector<Operation *, 32> members;
};

Value peelBroadcast(Value v) {
  while (auto bcast = v.getDefiningOp<stablehlo::DynamicBroadcastInDimOp>())
    v = bcast.getOperand();
  return v;
}

bool isFloatScalar(Value v, float expected) {
  v = peelBroadcast(fusionv2::peelAssumingYield(v));
  auto cst = v.getDefiningOp<stablehlo::ConstantOp>();
  if (!cst) return false;
  auto attr = dyn_cast<DenseElementsAttr>(cst.getValue());
  auto ty = dyn_cast<RankedTensorType>(cst.getType());
  return ty && ty.getRank() == 0 && ty.getElementType().isF32() && attr &&
         attr.isSplat() &&
         attr.getSplatValue<APFloat>().convertToDouble() == expected;
}

bool matchRelu(Value v, Value &add21Value) {
  Value peeled = fusionv2::peelAssumingYield(v);
  if (auto add = peeled.getDefiningOp<stablehlo::AddOp>()) {
    for (int i = 0; i < 2; ++i) {
      Value maybeAbs = peelBroadcast(add.getOperand(i));
      Value other = peelBroadcast(add.getOperand(1 - i));
      auto abs = maybeAbs.getDefiningOp<stablehlo::AbsOp>();
      if (abs && abs.getOperand() == other) {
        add21Value = other;
        return true;
      }
    }
    return false;
  }
  if (auto max = peeled.getDefiningOp<stablehlo::MaxOp>()) {
    for (int i = 0; i < 2; ++i) {
      Value lhs = peelBroadcast(max.getOperand(i));
      Value rhs = peelBroadcast(max.getOperand(1 - i));
      if (isFloatScalar(rhs, 0.0f) && lhs.getDefiningOp()) {
        add21Value = lhs;
        return true;
      }
    }
  }
  return false;
}

LogicalResult matchKpTbi(Operation *anchorOp, TbiMatch &m) {
  fusionv2::ConcatLike concat = fusionv2::ConcatLike::match(anchorOp);
  if (!concat) return failure();
  auto wTy = dyn_cast<RankedTensorType>(concat.result().getType());
  if (!wTy || wTy.getRank() != 3 || !wTy.getElementType().isF32())
    return failure();
  auto dim = concat.dimension();
  if (!dim || *dim != wTy.getRank() - 1) return failure();
  SmallVector<Value, 6> operands = concat.inputs();
  if (operands.size() != 4) return failure();

  Value reluValue = fusionv2::peelAssumingYield(operands[0]);
  auto realdiv = reluValue.getDefiningOp<stablehlo::MulOp>();
  if (!realdiv) return failure();
  Value reluIn;
  if (isFloatScalar(realdiv.getRhs(), 0.5f)) {
    reluIn = realdiv.getLhs();
  } else if (isFloatScalar(realdiv.getLhs(), 0.5f)) {
    reluIn = realdiv.getRhs();
  } else {
    return failure();
  }

  Value add21Value;
  if (!matchRelu(reluIn, add21Value)) return failure();

  auto add21 =
      fusionv2::peelAssumingYield(add21Value).getDefiningOp<stablehlo::AddOp>();
  if (!add21) return failure();
  Value dotV, bias;
  auto takeDot = [&](Value lhs, Value rhs) {
    Value l = peelBroadcast(lhs);
    Value r = peelBroadcast(rhs);
    auto dot = l.getDefiningOp<stablehlo::DotGeneralOp>();
    if (dot) {
      dotV = l;
      bias = r;
      return true;
    }
    dot = r.getDefiningOp<stablehlo::DotGeneralOp>();
    if (dot) {
      dotV = r;
      bias = l;
      return true;
    }
    return false;
  };
  if (!takeDot(add21.getLhs(), add21.getRhs())) return failure();
  auto dot = dotV.getDefiningOp<stablehlo::DotGeneralOp>();

  auto dims = dot.getDotDimensionNumbers();
  if (dims.getLhsBatchingDimensions().size() != 1 ||
      dims.getRhsBatchingDimensions().size() != 1 ||
      dims.getLhsBatchingDimensions()[0] != 0 ||
      dims.getRhsBatchingDimensions()[0] != 0 ||
      dims.getLhsContractingDimensions().size() != 1 ||
      dims.getRhsContractingDimensions().size() != 1 ||
      dims.getLhsContractingDimensions()[0] != 2 ||
      dims.getRhsContractingDimensions()[0] != 1)
    return failure();
  m.batchInput = peelBroadcast(dot.getLhs());
  m.weight = peelBroadcast(dot.getRhs());
  m.bias = bias;

  m.tileInput = operands[1];

  auto binOp = [&](Value v, auto wanted) -> bool {
    Operation *op = fusionv2::peelAssumingYield(v).getDefiningOp();
    if (!op || !wanted(op) || op->getNumOperands() != 2) return false;
    Value x = peelBroadcast(op->getOperand(0));
    Value y = peelBroadcast(op->getOperand(1));
    return (x == realdiv.getResult() && y == m.tileInput) ||
           (y == realdiv.getResult() && x == m.tileInput);
  };
  if (!binOp(operands[2],
             [](Operation *op) { return isa<stablehlo::SubtractOp>(op); }))
    return failure();
  if (!binOp(operands[3],
             [](Operation *op) { return isa<stablehlo::MulOp>(op); }))
    return failure();

  auto batchTy = dyn_cast<RankedTensorType>(m.batchInput.getType());
  auto weightTy = dyn_cast<RankedTensorType>(m.weight.getType());
  auto biasTy = dyn_cast<RankedTensorType>(m.bias.getType());
  auto tileTy = dyn_cast<RankedTensorType>(m.tileInput.getType());
  if (!batchTy || batchTy.getRank() != 3 || !batchTy.getElementType().isF32())
    return failure();
  if (!weightTy || weightTy.getRank() != 2 ||
      !weightTy.getElementType().isF32())
    return failure();
  if (!biasTy || biasTy.getRank() != 1 || !biasTy.getElementType().isF32())
    return failure();
  if (!tileTy || tileTy.getRank() != 3 || !tileTy.getElementType().isF32())
    return failure();

  m.w = concat.result();

  SmallPtrSet<Operation *, 32> seen;
  for (Value root : operands) collectDataMembers(root, seen, m.members);
  collectDataMembers(m.w, seen, m.members);
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
      if (r == m.w) continue;
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

static LogicalResult fuseKpTbi(Operation *anchorOp, PatternRewriter &rewriter,
                               fusionv2::CallPortTable *portTable) {
  TbiMatch m;
  if (failed(matchKpTbi(anchorOp, m))) return failure();

  func::FuncOp main = anchorOp->getParentOfType<func::FuncOp>();
  Operation *tailOpsArr[] = {anchorOp};
  SmallVector<Operation *, 1> tailOps(tailOpsArr, tailOpsArr + 1);
  Value outputsArr[] = {m.w};
  SmallVector<Value, 1> outputs(outputsArr, outputsArr + 1);
  SmallVector<std::string> tailAnchors;
  tailAnchors.push_back(fusionv2::anchorFromLoc(m.w.getLoc()) + ":0");

  SmallVector<ProbeArg> args;
  if (failed(collectBoundary(main, m.members, args))) return failure();

  if (args.size() != 4) {
    ANNC_LOG_DEBUG("op-fusion-v2") << "kp_target_behavior_interaction: args="
                 << args.size() << " (want 4); dropped\n";
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
  for (Value v : {m.batchInput, m.weight, m.bias, m.tileInput})
    if (!pick(v)) return failure();
  args.assign(ordered.begin(), ordered.end());

  auto module = anchorOp->getParentOfType<mlir::ModuleOp>();
  std::string name =
      uniquifySymbolName(module, "KPFusedTargetBehaviorInteraction");
  if (failed(outlineSite(main, rewriter, args, m.members, tailOps, outputs,
                         name, "kp_target_behavior_interaction", tailAnchors,
                         {}, *portTable)))
    return failure();
  return success();
}

class FuseKpTbiV2Pattern : public OpRewritePattern<stablehlo::ConcatenateOp> {
 public:
  explicit FuseKpTbiV2Pattern(MLIRContext *ctx,
                              fusionv2::CallPortTable *portTableArg)
      : OpRewritePattern(ctx), portTable(portTableArg) {
    assert(portTable && "requires a port table");
  }

  LogicalResult matchAndRewrite(stablehlo::ConcatenateOp anchor,
                                PatternRewriter &rewriter) const override {
    return fuseKpTbi(anchor.getOperation(), rewriter, portTable);
  }

 private:
  fusionv2::CallPortTable *portTable;
};

class FuseKpTbiResiduePattern : public OpRewritePattern<atir::UnsupportedOp> {
 public:
  explicit FuseKpTbiResiduePattern(MLIRContext *ctx,
                                   fusionv2::CallPortTable *portTableArg)
      : OpRewritePattern(ctx), portTable(portTableArg) {
    assert(portTable && "requires a port table");
  }

  LogicalResult matchAndRewrite(atir::UnsupportedOp anchor,
                                PatternRewriter &rewriter) const override {
    if (anchor.getOpType() != "ConcatV2") return failure();
    return fuseKpTbi(anchor.getOperation(), rewriter, portTable);
  }

 private:
  fusionv2::CallPortTable *portTable;
};

}  // namespace
}  // namespace atir

void atir::addTargetBehaviorInteractionV2Pattern(
    RewritePatternSet &patterns, fusionv2::CallPortTable *portTable) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<FuseKpTbiV2Pattern>(ctx, portTable);
  patterns.add<FuseKpTbiResiduePattern>(ctx, portTable);
}
