#ifndef ATIR_MLIR_CONVERSION_PASSES_TD
#define ATIR_MLIR_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

/* Affine */
def ConvertAtirToAffine : Pass<"convert-atir-to-affine", "ModuleOp"> {
  let summary = "convert atir ops to affine ops";
  let constructor = "createConvertAtirToAffine()";
  let dependentDialects = [
    "AtirDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
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
    "mlir::bufferization::BufferizationDialect"
  ];
}

#endif // ATIR_MLIR_CONVERSION_PASSES_TD