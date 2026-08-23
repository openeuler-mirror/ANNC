#ifndef ANNC_DIALECT_ATIR_PASSES_PATTERNS_FUSIONBOUNDARYUTILS_H
#define ANNC_DIALECT_ATIR_PASSES_PATTERNS_FUSIONBOUNDARYUTILS_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Operation;
}

namespace atir {

llvm::SmallVector<mlir::Value> collectEscapingResults(
    llvm::ArrayRef<mlir::Operation *> fusedOps);

llvm::SmallVector<mlir::Value> collectBoundaryInputs(
    llvm::ArrayRef<mlir::Operation *> fusedOps);

}  // namespace atir

#endif  // ANNC_DIALECT_ATIR_PASSES_PATTERNS_FUSIONBOUNDARYUTILS_H
