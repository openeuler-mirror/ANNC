#include "Dialect/Atir/Passes/Patterns/FusionBoundaryUtils.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"

namespace atir {

llvm::SmallVector<mlir::Value> collectEscapingResults(
    llvm::ArrayRef<mlir::Operation *> fusedOps) {
  llvm::SmallPtrSet<mlir::Operation *, 16> fusedSet(fusedOps.begin(),
                                                    fusedOps.end());
  llvm::SmallVector<mlir::Value> outputs;
  for (mlir::Operation *op : fusedOps) {
    for (mlir::Value result : op->getResults()) {
      if (llvm::any_of(result.getUsers(), [&](mlir::Operation *user) {
            return !fusedSet.contains(user);
          })) {
        outputs.push_back(result);
      }
    }
  }
  return outputs;
}

llvm::SmallVector<mlir::Value> collectBoundaryInputs(
    llvm::ArrayRef<mlir::Operation *> fusedOps) {
  llvm::SmallPtrSet<mlir::Operation *, 16> fusedSet(fusedOps.begin(),
                                                    fusedOps.end());
  llvm::SetVector<mlir::Value> inputs;
  for (mlir::Operation *op : fusedOps) {
    for (mlir::Value operand : op->getOperands()) {
      if (!fusedSet.contains(operand.getDefiningOp())) inputs.insert(operand);
    }
  }
  return llvm::SmallVector<mlir::Value>(inputs.begin(), inputs.end());
}

}  // namespace atir
