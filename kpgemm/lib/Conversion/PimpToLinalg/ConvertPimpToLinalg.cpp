#include <iostream>
#include <Conversion/Common/InputTypeConverter.h>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "Conversion/PimpToLinalg/PimpTypeConverter.h"
#include "Conversion/Passes.h"
#include "Conversion/PimpToLinalg/OpLowering.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace pimp {

    class ConvertPimpToLinalg : public ConvertPimpToLinalgBase<ConvertPimpToLinalg> {
    public:
        ConvertPimpToLinalg() = default;

        void runOnOperation() override {
            std::cout << "this is ConvertPimpToLinalg" << std::endl;
            ConversionTarget target(getContext());
            target.addLegalDialect<linalg::LinalgDialect,
                                   arith::ArithDialect>();
            PimpTypeToLinalgConverter pimpTypeConverter;
            InputTypeConverter inputTypeConverter;
            RewritePatternSet patterns(&getContext());
            populatePimpToLinalgConversionPatterns(inputTypeConverter, pimpTypeConverter, patterns);
            target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
                return inputTypeConverter.isSignatureLegal(op.getFunctionType());
            });
            target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
                return pimpTypeConverter.isLegal(op->getOperandTypes());
            });
            if (failed(applyPartialConversion(getOperation(),target,std::move(patterns)))){
                signalPassFailure();
            }
     }
 };
     std::unique_ptr<mlir::Pass> createConvertPimpToLinalg() {
         std::cout << "this is createConvertPimpToLinalgPass" << std::endl;
         return std::make_unique<ConvertPimpToLinalg>();
     }

}  // namespace pimp