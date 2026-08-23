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

void preserveOutputEndpoint(IdentityOp identity, Value input,
                            PatternRewriter &rewriter) {
  auto identityMetadata = identity->getAttrOfType<DictionaryAttr>("metadata");
  auto endpoint = identityMetadata
                      ? dyn_cast_or_null<StringAttr>(
                            identityMetadata.get("tf.output_tensor"))
                      : StringAttr();
  auto producerResult = dyn_cast<OpResult>(input);
  if (!endpoint || endpoint.empty() || !producerResult) return;

  Operation *producer = producerResult.getOwner();
  SmallVector<Attribute> endpoints(producer->getNumResults(),
                                   rewriter.getStringAttr(""));
  if (auto producerMetadata =
          producer->getAttrOfType<DictionaryAttr>("metadata")) {
    if (auto existing = dyn_cast_or_null<ArrayAttr>(
            producerMetadata.get("tf.output_tensors"));
        existing && existing.size() == endpoints.size()) {
      llvm::copy(existing, endpoints.begin());
    }
  }
  endpoints[producerResult.getResultNumber()] = endpoint;

  NamedAttrList producerMetadata(
      producer->getAttrOfType<DictionaryAttr>("metadata"));
  producerMetadata.set("tf.output_tensors", rewriter.getArrayAttr(endpoints));
  producer->setAttr("metadata",
                    producerMetadata.getDictionary(rewriter.getContext()));
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

    preserveOutputEndpoint(identity, input, rewriter);

    // Tensor names participate in TensorType identity. Preserve the Identity
    // descriptor while bypassing its redundant data movement. ATIR compute
    // ops keep their output buffer as operand 0, so update that buffer too;
    // OpFusion uses it when it reconstructs the fused function boundary.
    if (auto *inputDefiningOp = input.getDefiningOp();
        inputDefiningOp && inputDefiningOp->getNumOperands() > 0) {
      Value outputBuffer = inputDefiningOp->getOperand(0);
      if (outputBuffer.getDefiningOp<BufferOp>() &&
          outputBuffer.getType() == inputType) {
        outputBuffer.setType(resultType);
      }
    }
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
