// RUN: not annc-asm %s -aarch64-gemm-leaf-materialization 2>&1 | FileCheck %s

// CHECK: prepacked RHS requires a valid data binding

func.func @missing_prepacked_rhs_binding(
    %c: memref<3x8xf32>, %a: memref<3x8xf32>,
    %b: memref<8x8xf32>) attributes {
  annc.aarch64.gemm_strategies = {s0 = {
    version = 4 : i64, target_arch = "hip12", isa = "neon",
    data_type = "f32", kernel_family = "annc-neon-f32-v1",
    cache_strategy = "fixed", kernel_strategy = "fixed",
    cache_parameters = {mc = 3 : i64, nc = 8 : i64, kc = 8 : i64},
    kernel_parameters = {mr = 3 : i64, panel_lanes = 2 : i64},
    execution_kind = "gemm", rhs_packing = "prepacked"
  }}
} {
  linalg.matmul {
    annc.aarch64.gemm_stage = "kernel_tiled",
    annc.aarch64.kc_mode = "overwrite",
    annc.aarch64.gemm_problem = {
      version = 1 : i64, m = 3 : i64, n = 8 : i64, k = 8 : i64,
      lda = 8 : i64, ldb = 8 : i64, ldc = 8 : i64
    },
    annc.aarch64.gemm_strategy = "s0"
  } ins(%a, %b : memref<3x8xf32>, memref<8x8xf32>)
    outs(%c : memref<3x8xf32>)
  return
}
