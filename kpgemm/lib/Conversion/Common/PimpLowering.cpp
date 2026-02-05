#include "Conversion/Common/PimpLowering.h"

#include <mlir/Dialect/Bufferization/IR/Bufferization.h>

namespace pimp
{
void FuncReturnOpLowering::Lowering(PatternRewriter &rewriter, Adaptor adaptor, func::ReturnOp op) const {
    auto loweringOp = rewriter.create<func::ReturnOp>(op.getLoc(), adaptor.getOperands());
    rewriter.replaceOp(op, loweringOp);
}
}
