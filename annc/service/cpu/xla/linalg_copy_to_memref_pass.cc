#include <cstdint>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "transforms/passes.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"


namespace mlir {

#define GEN_PASS_DEF_LINALGCOPYTOMEMREFPASS
#include "transforms/passes.h.inc"

namespace {

struct LinalgToMemrefCopyPattern : public OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern::OpRewritePattern;
 
  LogicalResult matchAndRewrite(linalg::CopyOp op,
                                PatternRewriter& rewriter) const override {    
    // Rewrite must happens after bufferization.        
    if (!op.hasBufferSemantics()) {
	    return op->emitError() <<
        "linalg.copy on tensors cannot be converted into memref.copy";
	  }
	
    // Target can be converted, do it
    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> outputs = op.getOutputs();
    assert(inputs.size() == 1 && "expected CopyOp with one input");
    assert(outputs.size() == 1 && "expected CopyOp with one output");
 
    rewriter.replaceOpWithNewOp<memref::CopyOp>(
      op, inputs.front(), outputs.front());        

    return success();
  }
};  

struct LinalgCopyToMemrefPass
    : public impl::LinalgCopyToMemrefPassBase<LinalgCopyToMemrefPass> {

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext* ctx = f.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<LinalgToMemrefCopyPattern>(ctx);

    mlir::ConversionTarget target(*ctx);
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
    target.addIllegalOp<linalg::CopyOp>();

    if (failed(applyPartialConversion(f, target, std::move(patterns)))) {
      signalPassFailure();
    }  
  }
};

}  // namespace

} // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
mlir::hlo::createLinalgCopyToMemrefPass() {
  return std::make_unique<LinalgCopyToMemrefPass>();
}

