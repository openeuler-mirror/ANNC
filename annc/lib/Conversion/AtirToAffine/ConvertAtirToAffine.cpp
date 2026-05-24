#include <iostream>
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "Conversion/AtirToAffine/AtirTypeConverter.h"
#include "Conversion/Passes.h"
#include "Conversion/AtirToAffine/OpLowering.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace atir {

    class ConvertAtirToAffine : public ConvertAtirToAffineBase<ConvertAtirToAffine> {
    public:
        ConvertAtirToAffine() = default;

        void runOnOperation() override {
            std::cout << "this is ConvertAtirToAffine" << std::endl;
            ConversionTarget target(getContext());
            target.addLegalDialect<affine::AffineDialect,
                                   arith::ArithDialect,
                                   bufferization::BufferizationDialect,
                                   memref::MemRefDialect,
                                   func::FuncDialect>();
            AtirTypeToAffineConverter atirTypeConverter;
            RewritePatternSet patterns(&getContext());
            populateAtirToAffineConversionPatterns(atirTypeConverter, patterns);
            target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
                return atirTypeConverter.isSignatureLegal(op.getFunctionType());
            });
            target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
                return atirTypeConverter.isLegal(op->getOperandTypes());
            });
            target.addDynamicallyLegalOp<atir::CustomizeOp>([&](atir::CustomizeOp op) {
                return op.getOpType() == "ANNCFused";
            });
            if (failed(applyPartialConversion(getOperation(),target,std::move(patterns)))){
                signalPassFailure();
            }
     }
 };

     std::unique_ptr<mlir::Pass> createConvertAtirToAffine() {
         std::cout << "this is createConvertAtirToAffinePass" << std::endl;
         return std::make_unique<ConvertAtirToAffine>();
     }
}  // namespace atir
