#ifndef ATIR_MLIR_AARCH64_CODEGEN_PASSES_TD
#define ATIR_MLIR_AARCH64_CODEGEN_PASSES_TD

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


/*  */
def CacheParallel : Pass<"cache-parallel", "ModuleOp"> {
  let summary = "cache parallel on linalg op";
  let constructor = "createCacheParallel()";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::memref::MemRefDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::vector::VectorDialect"
  ];
}

/* cache reduction */
def CacheReduction : Pass<"cache-reduction", "ModuleOp"> {
  let summary = "cache reduction on linalg op";
  let constructor = "createCacheReduction()";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::memref::MemRefDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::vector::VectorDialect"
  ];
}
/* vector common parallel */
def VectorCommonParallel : Pass<"vector-common-parallel", "ModuleOp"> {
  let summary = "vector common parallel on linalg op";
  let constructor = "createVectorCommonParallel()";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::memref::MemRefDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::vector::VectorDialect"
  ];
}

/* vector reduction */
def VectorReduction : Pass<"vector-reduction", "ModuleOp"> {
  let summary = "vector reduction on linalg op";
  let constructor = "createVectorReduction()";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::memref::MemRefDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::vector::VectorDialect"
  ];
}
#endif // ATIR_MLIR_AARCH64_CODEGEN_PASSES_TD