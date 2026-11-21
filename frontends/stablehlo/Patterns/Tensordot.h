#ifndef ANNC_FRONTENDS_STABLEHLO_PATTERNS_TENSORDOT_H
#define ANNC_FRONTENDS_STABLEHLO_PATTERNS_TENSORDOT_H
#include "mlir/IR/Attributes.h"
namespace annc::stablehlo {
struct HloGraph;
void partitionTensordot(HloGraph &, mlir::SmallVectorImpl<mlir::Attribute> &);
}  // namespace annc::stablehlo
#endif
