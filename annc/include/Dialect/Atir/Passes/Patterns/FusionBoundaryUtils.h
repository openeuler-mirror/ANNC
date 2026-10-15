#ifndef ANNC_DIALECT_ATIR_PASSES_PATTERNS_FUSIONBOUNDARYUTILS_H
#define ANNC_DIALECT_ATIR_PASSES_PATTERNS_FUSIONBOUNDARYUTILS_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Operation;
}

namespace atir {

class MatMulOp;
class AddOp;
class ReluOp;

llvm::SmallVector<mlir::Value> collectEscapingResults(
    llvm::ArrayRef<mlir::Operation *> fusedOps);

llvm::SmallVector<mlir::Value> collectBoundaryInputs(
    llvm::ArrayRef<mlir::Operation *> fusedOps);

// Returns the bias operand of `add` when it feeds from `matmul` with a
// rank-1 tensor bias (the builtin kernels take AnncMemRef1DF32), otherwise
// nullptr.
mlir::Value rank1BiasOperandOf(MatMulOp matmul, AddOp add);

// True when the matmul/add/relu chain matches the annc_execution_v2 builtin
// kernels: plain matmul, rank-1 bias, unbounded relu.
bool isExecutionV2BuiltinCompatible(MatMulOp matmul, AddOp add, ReluOp relu);

}  // namespace atir

#endif  // ANNC_DIALECT_ATIR_PASSES_PATTERNS_FUSIONBOUNDARYUTILS_H
