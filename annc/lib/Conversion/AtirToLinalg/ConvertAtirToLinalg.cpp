#include <iostream>
#include <Conversion/Common/InputTypeConverter.h>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "Conversion/AtirToLinalg/AtirTypeConverter.h"
#include "Conversion/Passes.h"
#include "Conversion/AtirToLinalg/OpLowering.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace atir {

    class ConvertAtirToLinalg : public ConvertAtirToLinalgBase<ConvertAtirToLinalg> {
    public:
        ConvertAtirToLinalg() = default;

        void runOnOperation() override {
            std::cout << "this is ConvertAtirToLinalg" << std::endl;
            ConversionTarget target(getContext());
            target.addLegalDialect<linalg::LinalgDialect,
                                   arith::ArithDialect,
                                   bufferization::BufferizationDialect,
                                   memref::MemRefDialect,
                                   func::FuncDialect>();
            AtirTypeToLinalgConverter atirTypeConverter;
            InputTypeConverter inputTypeConverter;
            RewritePatternSet patterns(&getContext());
            populateAtirToLinalgConversionPatterns(inputTypeConverter, atirTypeConverter, patterns);
            target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
                return inputTypeConverter.isSignatureLegal(op.getFunctionType());
            });
            target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
                return atirTypeConverter.isLegal(op->getOperandTypes());
            });
            if (failed(applyPartialConversion(getOperation(),target,std::move(patterns)))){
                signalPassFailure();
            }
     }
 };
     std::unique_ptr<mlir::Pass> createConvertAtirToLinalg() {
         std::cout << "this is createConvertAtirToLinalgPass" << std::endl;
         return std::make_unique<ConvertAtirToLinalg>();
     }

}  // namespace atir