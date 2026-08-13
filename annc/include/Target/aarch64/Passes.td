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

/* Normalize an eligible AArch64 GEMM problem before target tuning. */
def AArch64ResolveGemmPlan
    : Pass<"aarch64-resolve-gemm-plan", "ModuleOp"> {
  let summary = "resolve an eligible AArch64 GEMM problem";
  let constructor = "createAArch64ResolveGemmPlan()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect"
  ];
}

/* Select the baseline target-private strategy before optional autotuning. */
def AArch64SelectGemmStrategy
    : Pass<"aarch64-select-gemm-strategy", "ModuleOp"> {
  let summary = "select a baseline AArch64 GEMM strategy";
  let constructor = "createAArch64SelectGemmStrategy()";
  let options = [
    Option<"configPath", "config-path", "std::string", "",
           "Path to the external AArch64 GEMM tuning configuration">
  ];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect"
  ];
}

/* Optionally refine a target-private candidate before the plan is frozen. */
def AArch64AutotuneGemmPlan
    : Pass<"aarch64-autotune-gemm-plan", "ModuleOp"> {
  let summary = "autotune an AArch64 GEMM candidate";
  let constructor = "createAArch64AutotuneGemmPlan()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect"
  ];
}

/* Freeze a validated target-private candidate as an executable GEMM plan. */
def AArch64FinalizeGemmPlan
    : Pass<"aarch64-finalize-gemm-plan", "ModuleOp"> {
  let summary = "finalize the selected AArch64 GEMM plan";
  let constructor = "createAArch64FinalizeGemmPlan()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect"
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

/* packed affine matmul kernel */
def MatmulPackAffine : Pass<"matmul-pack-affine", "ModuleOp"> {
  let summary = "packed affine matmul kernel optimization";
  let constructor = "createMatmulPackAffine()";
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