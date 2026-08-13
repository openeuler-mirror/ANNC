#include "Dialect/Atir/Passes/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace atir {
namespace {

constexpr llvm::StringLiteral kGemmEpilogueAttrName = "annc.gemm.epilogue";

bool isFunctionArgument(Value value, func::FuncOp funcOp) {
  auto blockArg = llvm::dyn_cast<BlockArgument>(value);
  return blockArg && funcOp && blockArg.getOwner() == &funcOp.getBody().front();
}

bool hasCompatibleBiasShape(MatMulOp matmulOp, Value bias) {
  auto outputType =
      llvm::dyn_cast<atir::TensorType>(matmulOp.getResult().getType());
  auto biasType = llvm::dyn_cast<atir::TensorType>(bias.getType());
  if (!outputType || !biasType || outputType.getShape().size() != 2 ||
      biasType.getShape().size() != 1 ||
      outputType.getElementType() != biasType.getElementType()) {
    return false;
  }

  int64_t outputN = outputType.getShape()[1];
  int64_t biasN = biasType.getShape()[0];
  return outputN == ShapedType::kDynamic || biasN == ShapedType::kDynamic ||
         outputN == biasN;
}

bool haveCompatibleTensorTypes(Type lhs, Type rhs) {
  auto lhsType = llvm::dyn_cast<atir::TensorType>(lhs);
  auto rhsType = llvm::dyn_cast<atir::TensorType>(rhs);
  return lhsType && rhsType &&
         lhsType.getElementType() == rhsType.getElementType() &&
         lhsType.getShape() == rhsType.getShape();
}

ArrayAttr buildEpilogueAttr(PatternRewriter &rewriter, ReluOp reluOp) {
  SmallVector<Attribute> steps;

  NamedAttrList biasAdd;
  biasAdd.append("kind", rewriter.getStringAttr("bias_add"));
  biasAdd.append("input", rewriter.getI64IntegerAttr(0));
  biasAdd.append("broadcast", rewriter.getStringAttr("n"));
  steps.push_back(biasAdd.getDictionary(rewriter.getContext()));

  if (reluOp) {
    NamedAttrList relu;
    relu.append("kind", rewriter.getStringAttr("relu"));
    relu.append("limit", reluOp.getReluLimitAttr());
    steps.push_back(relu.getDictionary(rewriter.getContext()));
  }

  return rewriter.getArrayAttr(steps);
}

struct FuseGemmEpilogue : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern<MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    // Legacy producers may still attach these post-op attributes. They are
    // intentionally rejected here because this pass only consumes the new
    // ordered epilogue; the legacy attributes will be removed later.
    if (matmulOp.getBias() || matmulOp.getDoRelu() || matmulOp.getWithBias() ||
        matmulOp->hasAttr(kGemmEpilogueAttrName)) {
      return failure();
    }
    if (matmulOp.getLeftTranspose() || matmulOp.getRightTranspose() ||
        matmulOp.getOutputTranspose()) {
      return failure();
    }
    if (!matmulOp.getResult().hasOneUse()) return failure();

    auto addOp =
        llvm::dyn_cast<AddOp>(*matmulOp.getResult().getUsers().begin());
    if (!addOp || addOp.getInputs().size() != 2 || addOp.getScalar() ||
        addOp.getDoRelu()) {
      return failure();
    }

    Value bias;
    if (addOp.getInputs()[0] == matmulOp.getResult()) {
      bias = addOp.getInputs()[1];
    } else if (addOp.getInputs()[1] == matmulOp.getResult()) {
      bias = addOp.getInputs()[0];
    } else {
      return failure();
    }
    if (!hasCompatibleBiasShape(matmulOp, bias)) return failure();

    ReluOp reluOp;
    if (addOp.getResult().hasOneUse()) {
      reluOp = llvm::dyn_cast<ReluOp>(*addOp.getResult().getUsers().begin());
    }
    if (reluOp && reluOp.getInput() != addOp.getResult()) return failure();

    Operation *lastOp = reluOp ? reluOp.getOperation() : addOp.getOperation();
    Value finalDestination = lastOp->getOperand(0);
    auto funcOp = matmulOp->getParentOfType<func::FuncOp>();
    if (!isFunctionArgument(finalDestination, funcOp) ||
        !haveCompatibleTensorTypes(finalDestination.getType(),
                                   matmulOp.getC().getType())) {
      return failure();
    }

    Operation *matmulBuffer = matmulOp.getC().getDefiningOp<BufferOp>();
    Operation *addBuffer = addOp.getOutput().getDefiningOp<BufferOp>();

    NamedAttrList attrs(matmulOp->getAttrs());
    attrs.set(kGemmEpilogueAttrName, buildEpilogueAttr(rewriter, reluOp));

    SmallVector<Value> operands = {finalDestination, matmulOp.getLhs(),
                                   matmulOp.getRhs(), bias};
    rewriter.setInsertionPoint(lastOp);
    auto fusedMatmul = rewriter.create<MatMulOp>(
        lastOp->getLoc(), TypeRange{lastOp->getResult(0).getType()}, operands,
        attrs.getAttrs());
    rewriter.replaceAllUsesWith(lastOp->getResult(0), fusedMatmul.getResult());

    if (reluOp) rewriter.eraseOp(reluOp);
    rewriter.eraseOp(addOp);
    rewriter.eraseOp(matmulOp);
    if (addBuffer == matmulBuffer) {
      if (addBuffer && addBuffer->use_empty()) rewriter.eraseOp(addBuffer);
    } else {
      if (addBuffer && addBuffer->use_empty()) rewriter.eraseOp(addBuffer);
      if (matmulBuffer && matmulBuffer->use_empty())
        rewriter.eraseOp(matmulBuffer);
    }
    return success();
  }
};

class AtirGemmEpilogueFusion
    : public AtirGemmEpilogueFusionBase<AtirGemmEpilogueFusion> {
 public:
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseGemmEpilogue>(&getContext());

    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createAtirGemmEpilogueFusionPass() {
  return std::make_unique<AtirGemmEpilogueFusion>();
}

}  // namespace atir
