// RUN: annc-asm %s -aarch64-gemm-leaf-materialization -aarch64-gemm-microkernel-lowering -aarch64-gemm-abi-lowering -aarch64-verify-gemm-schedule | FileCheck %s

// The row-major leaf carries ldb and lowers to an external symbol supplied by
// libannc_gemm_microkernels.a; no assembly source is part of this repository.
// CHECK: llvm.func @annc_aarch64_neon_kernel_rm_mr2_n3_kg_r1_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32)
// CHECK-LABEL: func.func @row_major_neon(
// CHECK: llvm.call @annc_aarch64_neon_kernel_rm_mr2_n3_kg_r1_f32
// CHECK-NOT: @__annc_aarch64_gemm_microkernel_rm_leaf
func.func @row_major_neon(%c: memref<2x3xf32>, %a: memref<2x5xf32>,
                           %b: memref<5x3xf32>) attributes {annc.aarch64.gemm_strategies = {s0 = {version = 4 : i64, target_arch = "hip12", isa = "neon", data_type = "f32", kernel_family = "annc-neon-f32-v1", cache_strategy = "fixed", kernel_strategy = "fixed", cache_parameters = {mc = 3 : i64, nc = 16 : i64, kc = 8 : i64}, kernel_parameters = {mr = 3 : i64, panel_lanes = 2 : i64}, execution_kind = "gemm", rhs_packing = "direct"}}} {
  linalg.matmul {
    annc.aarch64.gemm_stage = "kernel_tiled",
    annc.aarch64.kc_mode = "overwrite",
    annc.aarch64.gemm_problem = {version = 1 : i64, m = 2 : i64, n = 3 : i64, k = 5 : i64, lda = 5 : i64, ldb = 3 : i64, ldc = 3 : i64}, annc.aarch64.gemm_strategy = "s0", annc.aarch64.gemm_candidate = {version = 4 : i64, planning_state = "planned", strategy = "s0", max_thread_count = 1 : i64, thread_count = 1 : i64, tasks_m = 1 : i64, tasks_n = 1 : i64, shard_direction = "rows", thread_partition = "static-2d"}
  } ins(%a, %b : memref<2x5xf32>, memref<5x3xf32>)
    outs(%c : memref<2x3xf32>)
  return
}
