#include <iostream>
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "Conversion/PimpToAffine/PimpTypeConverter.h"
#include "Conversion/Passes.h"
#include "Conversion/PimpToAffine/OpLowering.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace pimp {

    class ConvertPimpToAffine : public ConvertPimpToAffineBase<ConvertPimpToAffine> {
    public:
        ConvertPimpToAffine() = default;

        void runOnOperation() override {
            std::cout << "this is ConvertPimpToAffine" << std::endl;
            ConversionTarget target(getContext());
            target.addLegalDialect<affine::AffineDialect,
                                   arith::ArithDialect,
                                   bufferization::BufferizationDialect,
                                   memref::MemRefDialect>();
            PimpTypeToAffineConverter pimpTypeConverter;
            RewritePatternSet patterns(&getContext());
            populatePimpToAffineConversionPatterns(pimpTypeConverter, patterns);
            target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
                return pimpTypeConverter.isSignatureLegal(op.getFunctionType());
            });
            target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
                return pimpTypeConverter.isLegal(op->getOperandTypes());
            });
            if (failed(applyPartialConversion(getOperation(),target,std::move(patterns)))){
                signalPassFailure();
            }
     }
 };

     std::unique_ptr<mlir::Pass> createConvertPimpToAffine() {
         std::cout << "this is createConvertPimpToAffinePass" << std::endl;
         return std::make_unique<ConvertPimpToAffine>();
     }
}  // namespace pimp