#include <iterator>
#include <memory>
#include <optional>
#include <utility>


#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "thlo/IR/thlo_ops.h"
#include "thlo/transforms/passes.h"

namespace mlir {
namespace thlo {

#define GEN_PASS_DEF_THLOLEGALIZECONCATPASS
#include "thlo/transforms/thlo_passes.h.inc"

namespace {

struct ConcatenateOpPattern : public OpRewritePattern<ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter& rewriter) const override {    
    // Rewrite must happen before bufferization.
    if (op.hasBufferSemantics())
      return op->emitError() << "expected tensor semantics";
    
    int64_t concatDim = op.getDimension().getSExtValue();

    rewriter.replaceOpWithNewOp<tensor::ConcatOp>(op, /*dim=*/ concatDim,
                                                    /*inputs=*/ op.getInputs());
    return success();
  }
};  

struct LegalizeConcatPass
    : public impl::ThloLegalizeConcatPassBase<LegalizeConcatPass> {
  // Rewrites a thlo.concatenate as a tensor.concat with identical operands and
  // semantics.
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext* ctx = f.getContext();
	
    RewritePatternSet patterns(ctx);
    patterns.add<ConcatenateOpPattern>(ctx);

    mlir::ConversionTarget target(*ctx);
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
    target.addIllegalOp<thlo::ConcatenateOp>();

    if (failed(applyPartialConversion(f, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace thlo
}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
mlir::thlo::createLegalizeConcatPass() {
  return std::make_unique<LegalizeConcatPass>();
}
