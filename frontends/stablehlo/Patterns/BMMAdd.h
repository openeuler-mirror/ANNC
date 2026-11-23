#ifndef ANNC_FRONTENDS_STABLEHLO_PATTERNS_BMMADD_H
#define ANNC_FRONTENDS_STABLEHLO_PATTERNS_BMMADD_H
#include "mlir/IR/Attributes.h"
namespace annc::stablehlo {
struct HloGraph;
void partitionBMMAdd(HloGraph &, mlir::SmallVectorImpl<mlir::Attribute> &);
}  // namespace annc::stablehlo
#endif
