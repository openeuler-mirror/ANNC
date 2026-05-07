#ifndef ATIRL_LOWERING_H
#define ATIRL_LOWERING_H
#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace atir
{

template <typename AtirOp>
class AtirLowering : public OpConversionPattern<AtirOp>
{
public:
    using OpConversionPattern<AtirOp>::OpConversionPattern;
    using Adaptor = typename AtirOp::Adaptor;
    LogicalResult matchAndRewrite(AtirOp atirOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
        Lowering(rewriter, adaptor, atirOp);
        return success();
    }

    virtual void Lowering(PatternRewriter &rewriter, Adaptor adaptor, AtirOp atirOp) const {
        llvm_unreachable("Not Implemented lowering");
    }
};

struct FuncReturnOpLowering : AtirLowering<func::ReturnOp>
{
    using AtirLowering::AtirLowering;
    void Lowering(PatternRewriter& rewriter, Adaptor adaptor, func::ReturnOp op) const override;
};
} // namespace atir
#endif //ATIRL_LOWERING_H