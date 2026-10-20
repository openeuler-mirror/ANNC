#include "Dialect/Atir/Passes/Patterns/FusionBoundaryUtils.h"

#include "Dialect/Atir/AtirOps.h"
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

mlir::Value rank1BiasOperandOf(MatMulOp matmul, AddOp add) {
  if (!matmul || !add || add->getNumOperands() != 3) return nullptr;
  mlir::Value bias = add.getOperand(1) == matmul.getResult()
                         ? add.getOperand(2)
                         : add.getOperand(1);
  auto biasTensor = dyn_cast<TensorType>(bias.getType());
  if (!biasTensor || biasTensor.getShape().size() != 1) return nullptr;
  return bias;
}

bool isExecutionV2BuiltinCompatible(MatMulOp matmul, AddOp add, ReluOp relu) {
  if (!matmul || !add || !relu) return false;
  if (matmul.getWithBias() || matmul.getDoRelu() ||
      matmul.getRightTranspose() || matmul.getLeftTranspose() ||
      matmul.getOutputTranspose() || matmul.getMStart() || matmul.getNStart() ||
      matmul.getKStart() || matmul.getMSize() || matmul.getNSize() ||
      matmul.getKSize() || matmul->getAttr("rhs_format"))
    return false;
  if (add.getDoRelu() || add.getScalar()) return false;
  if (!rank1BiasOperandOf(matmul, add)) return false;
  return relu.getReluLimit().convertToFloat() == -1.0f;
}

}  // namespace atir
