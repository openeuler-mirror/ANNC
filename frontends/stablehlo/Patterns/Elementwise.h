#ifndef ANNC_FRONTENDS_STABLEHLO_PATTERNS_ELEMENTWISE_H
#define ANNC_FRONTENDS_STABLEHLO_PATTERNS_ELEMENTWISE_H
#include <optional>
#include <string>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"
namespace annc::stablehlo {
// A proven HLO fragment. Sum preserves only the existing left add spine.
struct HloElementwise {
  std::string kind;
  mlir::SmallVector<mlir::Value> inputs;
  mlir::SmallVector<mlir::Operation *> absorbed;
};
std::optional<HloElementwise> analyzeElementwise(mlir::Value);
}  // namespace annc::stablehlo
#endif
