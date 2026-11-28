#include "Conversion/Common/AtirLowering.h"

#include <mlir/Dialect/Bufferization/IR/Bufferization.h>

namespace atir
{
void FuncReturnOpLowering::Lowering(PatternRewriter &rewriter, Adaptor adaptor, func::ReturnOp op) const {
    auto loweringOp = rewriter.create<func::ReturnOp>(op.getLoc(), adaptor.getOperands());
    rewriter.replaceOp(op, loweringOp);
}
}
