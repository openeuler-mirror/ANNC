#ifndef ATIR_MLIR_CONVERSION_PASSES_TD
#define ATIR_MLIR_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertContextKernelABI : Pass<"convert-context-kernel-abi", "ModuleOp"> {
  let summary = "Convert context-v3 kernel entry inputs to context slots";
  let description = [{
    Run after fusion pruning and shape specialization, before ATIR-to-Linalg
    conversion. The pass accepts only builtin ranked tensor arguments, unpacks
    their context memref descriptors in function argument order, and preserves
    function results and returns. Legacy ATIR tensor arguments are rejected.
  }];
  let constructor = "createConvertContextKernelABIPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def LowerContextKernelEntries : Pass<"lower-context-kernel-entries", "ModuleOp"> {
  let summary = "Lower closed single-builtin context-v3 entries to pointer/status calls";
  let description = [{
    Lower a closed tensor-signature entry containing either a shared ATIR
    builtin or a schema-checked standard call extracted from StableHLO.
    Tensor arguments remain visible until this pass replaces the exported
    entry ABI. Legacy entries are left unchanged.
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

def ConvertAtirV2ToLinalg : Pass<"convert-atir-v2-to-linalg", "ModuleOp"> {
  let summary = "convert atir v2 ops to linalg ops";
  let description = [{
    Lower ATIR V2 operations to the Linalg and builtin dialects. The pass is
    currently a conversion entry point; V2 operation patterns will be added
    incrementally without changing the legacy ATIR lowering pass.
  }];
  let constructor = "createConvertAtirV2ToLinalg()";
  let dependentDialects = [
    "AtirDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif // ATIR_MLIR_CONVERSION_PASSES_TD
