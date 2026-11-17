#include <memory>

#include "Conversion/AtirToLinalg/OpLowering.h"
#include "Conversion/Passes.h"
#include "Support/ContextABISwitch.h"
#include "Support/Log.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace atir {

#define GEN_PASS_DEF_CONVERTATIRV2TOLINALG
#include "Conversion/Passes.h.inc"

namespace {

class ConvertAtirV2ToLinalg
    : public impl::ConvertAtirV2ToLinalgBase<ConvertAtirV2ToLinalg> {
 public:
  void runOnOperation() override {
    ANNC_LOG_DEBUG("atir-v2-to-linalg") << "this is ConvertAtirV2ToLinalg\n";

    // Customize uses the context ABI directly, without legacy type conversion.
    if (annc::loweringContextABIEnabled()) {
      RewritePatternSet customizePatterns(&getContext());
      customizePatterns.add<ContextCustomizeLoweringToLinalg>(&getContext());
      if (failed(applyPatternsGreedily(getOperation(),
                                       std::move(customizePatterns)))) {
        signalPassFailure();
        return;
      }
    }

    // Keep this as a dialect conversion entry point so V2 patterns can be
    // added here incrementally. The legacy ATIR conversion remains separate:
    // V2 operations use builtin tensor types and must not be routed through
    // the legacy AtirTypeToLinalgConverter.
    ConversionTarget target(getContext());
    target.addLegalDialect<AtirDialect, linalg::LinalgDialect,
                           arith::ArithDialect, memref::MemRefDialect,
                           bufferization::BufferizationDialect,
                           LLVM::LLVMDialect, func::FuncDialect,
                           tensor::TensorDialect>();
    target.addIllegalOp<atir::FusedMatMulOp, atir::CustomizeOp>();
    RewritePatternSet patterns(&getContext());
    populateAtirV2ToLinalgConversionPatterns(patterns);
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return llvm::none_of(op->getOperandTypes(), [](Type type) {
        return isa<RankedTensorType, UnrankedTensorType>(type);
      });
    });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createConvertAtirV2ToLinalg() {
  return std::make_unique<ConvertAtirV2ToLinalg>();
}

}  // namespace atir
