#ifndef PIMP_MLIR_CONVERSION_PASSES_TD
#define PIMP_MLIR_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

/* Affine */
def ConvertPimpToAffine : Pass<"convert-pimp-to-affine", "ModuleOp"> {
  let summary = "convert pimp ops to affine ops";
  let constructor = "createConvertPimpToAffine()";
  let dependentDialects = [
    "PimpDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

/* Linalg */
def ConvertPimpToLinalg : Pass<"convert-pimp-to-linalg", "ModuleOp"> {
  let summary = "convert pimp ops to linalg ops";
  let constructor = "createConvertPimpToLinalg()";
  let dependentDialects = [
    "PimpDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif // PIMP_MLIR_CONVERSION_PASSES_TD