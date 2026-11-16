#ifndef ANNC_FRONTENDS_STABLEHLO_PATTERNS_LAYERNORM_H
#define ANNC_FRONTENDS_STABLEHLO_PATTERNS_LAYERNORM_H
#include "mlir/IR/Attributes.h"
namespace annc::stablehlo {
struct HloGraph;
void partitionLayerNorm(HloGraph &, mlir::SmallVectorImpl<mlir::Attribute> &);
}  // namespace annc::stablehlo
#endif
