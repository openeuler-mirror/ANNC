#ifndef ANNC_CONVERSION_COMMON_CUSTOMIZECALLLOWERING_H
#define ANNC_CONVERSION_COMMON_CUSTOMIZECALLLOWERING_H

#include "Dialect/Atir/AtirOps.h"
#include "mlir/IR/PatternMatch.h"

namespace atir {

void lowerCustomizeOpToFuncCall(mlir::PatternRewriter &rewriter,
                                atir::CustomizeOpAdaptor adaptor,
                                atir::CustomizeOp op);

} // namespace atir

#endif // ANNC_CONVERSION_COMMON_CUSTOMIZECALLLOWERING_H
