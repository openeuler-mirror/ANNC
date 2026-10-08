#ifndef ATIR_MLIR_CONVERSION_PASSES_TD
#define ATIR_MLIR_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

/* Linalg */
def ConvertAtirToLinalg : Pass<"convert-atir-to-linalg", "ModuleOp"> {
  let summary = "convert atir ops to linalg ops";
  let constructor = "createConvertAtirToLinalg()";
  let dependentDialects = [
    "AtirDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::bufferization::BufferizationDialect"
  ];
}

#endif // ATIR_MLIR_CONVERSION_PASSES_TD
