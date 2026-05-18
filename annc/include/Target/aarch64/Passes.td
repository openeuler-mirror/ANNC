#ifndef PIMP_MLIR_AARCH64_CODEGEN_PASSES_TD
#define PIMP_MLIR_AARCH64_CODEGEN_PASSES_TD

include "mlir/Pass/PassBase.td"

/* wrap one shot bufferize */
def KPGemmOneShotBufferize : Pass<"annc-one-shot-bufferize", "ModuleOp"> {
  let summary = "bufferize tensor to memref";
  let constructor = "createKPGemmOneShotBufferize()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif // PIMP_MLIR_AARCH64_CODEGEN_PASSES_TD