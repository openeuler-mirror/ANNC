#ifndef ANNC_FRONTENDS_STABLEHLO_PATTERNS_MASKEDMUL_H
#define ANNC_FRONTENDS_STABLEHLO_PATTERNS_MASKEDMUL_H
#include "mlir/IR/Attributes.h"
namespace annc::stablehlo {
struct HloGraph;
void partitionMaskedMul(HloGraph &, mlir::SmallVectorImpl<mlir::Attribute> &);
}  // namespace annc::stablehlo
#endif
