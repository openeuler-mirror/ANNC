// RUN: annc-asm %s -aarch64-gemm-thread-tiling | FileCheck %s
// RUN: annc-asm %s -aarch64-gemm-thread-tiling -aarch64-gemm-cache-blocking | FileCheck %s --check-prefix=CACHE

// CHECK-LABEL: func.func @threaded_gemm(
// CHECK: arith.constant 4 : i64
// CHECK: call @annc_threadpool_parallel_for_gemm
// CHECK-LABEL: func.func private @__annc_gemm_thread_task_0(
// CHECK-SAME: i64,
// CHECK: arith.divsi
// CHECK: arith.remsi
// CHECK: scf.if
// CHECK: linalg.matmul
// CHECK-SAME: annc.aarch64.gemm_stage = "thread_tiled"
// CHECK: return
// CACHE: annc.aarch64.gemm_candidate = {
// CACHE-SAME: thread_count = 4 : i64
// CACHE-SAME: thread_partition = "static-2d"
// CACHE: annc.aarch64.gemm_stage = "cache_blocked"
func.func @threaded_gemm(
    %c: memref<13x40xf32>, %a: memref<13x8xf32>, %b: memref<8x40xf32>) attributes {annc.aarch64.gemm_strategies = {s0 = {version = 4 : i64, target_arch = "hip12", isa = "sve", data_type = "f32", kernel_family = "annc-sve-f32-v1", cache_strategy = "fixed", kernel_strategy = "fixed", cache_parameters = {mc = 12 : i64, nc = 40 : i64, kc = 8 : i64}, kernel_parameters = {mr = 3 : i64, panel_lanes = 2 : i64}, execution_kind = "gemm", rhs_packing = "packed"}}} {
  linalg.matmul {
    annc.aarch64.gemm_problem = {version = 1 : i64, m = 13 : i64, n = 40 : i64, k = 8 : i64, lda = 8 : i64, ldb = 40 : i64, ldc = 40 : i64}, annc.aarch64.gemm_strategy = "s0", annc.aarch64.gemm_candidate = {version = 4 : i64, planning_state = "planned", strategy = "s0", max_thread_count = 4 : i64, thread_count = 4 : i64, tasks_m = 2 : i64, tasks_n = 2 : i64, shard_direction = "rows", thread_partition = "static-2d"}
  } ins(%a, %b : memref<13x8xf32>, memref<8x40xf32>)
    outs(%c : memref<13x40xf32>)
  return
}
