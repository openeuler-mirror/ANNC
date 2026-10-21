#include <Conversion/Common/InputTypeConverter.h>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "Conversion/AtirToLinalg/AtirTypeConverter.h"
#include "Conversion/Passes.h"
#include "Conversion/AtirToLinalg/OpLowering.h"
#include "Support/Log.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace atir {

    class ConvertAtirToLinalg : public ConvertAtirToLinalgBase<ConvertAtirToLinalg> {
    public:
        ConvertAtirToLinalg() = default;

        void runOnOperation() override {
            ANNC_LOG_DEBUG("atir-to-linalg") << "this is ConvertAtirToLinalg\n";
            ConversionTarget target(getContext());
            target.addLegalDialect<linalg::LinalgDialect,
                                   arith::ArithDialect,
                                   bufferization::BufferizationDialect,
                                   memref::MemRefDialect,
                                   tensor::TensorDialect,
                                   func::FuncDialect>();
            target.addIllegalOp<atir::AddOp, atir::BufferOp, atir::MatMulOp,
                                atir::ReluOp>();
            target.addDynamicallyLegalOp<atir::ReshapeOp>(
                [](atir::ReshapeOp op) {
                  return !isSupportedInsertUnitDimensionBeforeN(op);
                });
            target.addIllegalOp<atir::ConstantOp>();
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
                return;
            }
            // Emit these after pruning and constant lowering. A BMM host must
            // distinguish an unchanged initializer (zero) from a module-owned
            // folded weight (its actual data address). Missing accessors in
            // older libraries mean that the host must leave packing disabled.
            SmallVector<func::FuncOp> kernels;
            for (auto func : getOperation().getOps<func::FuncOp>())
                if (func->hasAttr("annc.kernel") &&
                    func->getAttrOfType<StringAttr>("fusion.pattern") ==
                        StringAttr::get(&getContext(), "batched_matmul_add"))
                    kernels.push_back(func);
            for (auto func : kernels) {
                std::string name = "__annc_effective_weight_" + func.getName().str();
                if (getOperation().lookupSymbol(name)) continue;
                OpBuilder builder(getOperation().getBodyRegion());
                builder.setInsertionPointToEnd(getOperation().getBody());
                auto accessor = builder.create<func::FuncOp>(
                    func.getLoc(), name,
                    builder.getFunctionType({}, {builder.getI64Type()}));
                accessor->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
                builder.setInsertionPointToStart(accessor.addEntryBlock());
                Value address;
                if (auto globalName = func->getAttrOfType<StringAttr>(
                        "annc.effective_weight_global")) {
                    auto global = getOperation().lookupSymbol<memref::GlobalOp>(
                        globalName.getValue());
                    auto buffer = builder.create<memref::GetGlobalOp>(
                        func.getLoc(), global.getType(), global.getSymName());
                    auto index = builder.create<memref::ExtractAlignedPointerAsIndexOp>(
                        func.getLoc(), buffer);
                    address = builder.create<arith::IndexCastOp>(
                        func.getLoc(), builder.getI64Type(), index);
                } else {
                    address = builder.create<arith::ConstantIntOp>(func.getLoc(), 0, 64);
                }
                builder.create<func::ReturnOp>(func.getLoc(), address);
            }
     }
 };
     std::unique_ptr<mlir::Pass> createConvertAtirToLinalg() {
         ANNC_LOG_DEBUG("atir-to-linalg") << "this is createConvertAtirToLinalgPass\n";
         return std::make_unique<ConvertAtirToLinalg>();
     }

}  // namespace atir
