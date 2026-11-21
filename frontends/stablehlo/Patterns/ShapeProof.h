#ifndef ANNC_FRONTENDS_STABLEHLO_PATTERNS_SHAPEPROOF_H
#define ANNC_FRONTENDS_STABLEHLO_PATTERNS_SHAPEPROOF_H
#include <optional>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
namespace annc {
std::optional<mlir::SmallVector<int64_t>> evaluateStablehloShape(mlir::Value);
bool proveStablehloBroadcast(mlir::Value actual, mlir::Value original);
}  // namespace annc
#endif
