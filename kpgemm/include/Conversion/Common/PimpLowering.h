#ifndef PIMPL_LOWERING_H
#define PIMPL_LOWERING_H
#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace pimp
{

template <typename PimpOp>
class PimpLowering : public OpConversionPattern<PimpOp>
{
public:
    using OpConversionPattern<PimpOp>::OpConversionPattern;
    using Adaptor = typename PimpOp::Adaptor;
    LogicalResult matchAndRewrite(PimpOp pimpOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
        Lowering(rewriter, adaptor, pimpOp);
        return success();
    }

    virtual void Lowering(PatternRewriter &rewriter, Adaptor adaptor, PimpOp pimpOp) const {
        llvm_unreachable("Not Implemented lowering");
    }
};

struct FuncReturnOpLowering : PimpLowering<func::ReturnOp>
{
    using PimpLowering::PimpLowering;
    void Lowering(PatternRewriter& rewriter, Adaptor adaptor, func::ReturnOp op) const override;
};
} // namespace pimp
#endif //PIMPL_LOWERING_H