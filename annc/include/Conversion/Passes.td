#ifndef ATIR_MLIR_CONVERSION_PASSES_TD
#define ATIR_MLIR_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerContextKernelEntries : Pass<"lower-context-kernel-entries", "ModuleOp"> {
  let summary = "Lower closed single-builtin context-v3 entries to pointer/status calls";
  let description = [{
    Run after fusion pruning, shape specialization, FastCodegen and any input
    constant folding, before ATIR-to-Linalg conversion. Tensor arguments remain
    visible to shape inference until this pass replaces the exported entry ABI.
    Legacy entries are left unchanged.
  }];
  let constructor = "createLowerContextKernelEntriesPass()";
  let dependentDialects = [
    "AtirDialect",
    "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

/* Linalg */
def ConvertAtirToLinalg : Pass<"convert-atir-to-linalg", "ModuleOp"> {
  let summary = "convert atir ops to linalg ops";
  let constructor = "createConvertAtirToLinalg()";
  let dependentDialects = [
    "AtirDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

#endif // ATIR_MLIR_CONVERSION_PASSES_TD
