#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace atir {
namespace {

bool hasSameDescriptorExceptName(TensorType input, TensorType result) {
  return input.getShape() == result.getShape() &&
         input.getElementType() == result.getElementType() &&
         input.getEncoding() == result.getEncoding() &&
         input.getStride() == result.getStride() &&
         input.getLayout() == result.getLayout() &&
         input.getMemType() == result.getMemType() &&
         input.getAddress() == result.getAddress() &&
         input.getDeviceParallel() == result.getDeviceParallel() &&
         input.getOnchipParallel() == result.getOnchipParallel() &&
         input.getCacheData() == result.getCacheData();
}

class EliminateIdentity final : public OpRewritePattern<IdentityOp> {
 public:
  using OpRewritePattern<IdentityOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IdentityOp identity,
                                PatternRewriter &rewriter) const override {
    Value input = identity.getInput();
    Value identityOutput = identity.getOutput();
    auto identityBuffer = identityOutput.getDefiningOp<BufferOp>();

    auto inputType = dyn_cast<TensorType>(input.getType());
    auto resultType = dyn_cast<TensorType>(identity.getResult().getType());
    if (!inputType || !resultType ||
        !hasSameDescriptorExceptName(inputType, resultType)) {
      return failure();
    }

    // Tensor names participate in TensorType identity. Preserve the Identity
    // descriptor while bypassing its redundant data movement.
    input.setType(resultType);
    rewriter.replaceOp(identity, input);
    if (identityBuffer && identityBuffer->use_empty()) {
      rewriter.eraseOp(identityBuffer);
    }
    return success();
  }
};

class AtirIdentityCanonicalizePass final
    : public AtirIdentityCanonicalizeBase<AtirIdentityCanonicalizePass> {
 public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<EliminateIdentity>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createAtirIdentityCanonicalizePass() {
  return std::make_unique<AtirIdentityCanonicalizePass>();
}

}  // namespace atir
