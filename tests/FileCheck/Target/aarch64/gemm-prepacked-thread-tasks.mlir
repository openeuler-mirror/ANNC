// RUN: annc-asm %s -aarch64-gemm-thread-tiling -aarch64-gemm-cache-blocking -aarch64-gemm-kernel-tiling | FileCheck %s
// RUN: annc-asm %s -aarch64-gemm-thread-tiling -aarch64-gemm-cache-blocking -aarch64-gemm-kernel-tiling -aarch64-gemm-leaf-materialization | FileCheck %s --check-prefix=LEAF
// RUN: annc-asm %s -aarch64-gemm-thread-tiling -aarch64-gemm-cache-blocking -aarch64-gemm-kernel-tiling -aarch64-gemm-leaf-materialization -aarch64-gemm-microkernel-lowering -aarch64-gemm-abi-lowering -aarch64-verify-gemm-schedule | FileCheck %s --check-prefix=SCHED

// A prepacked-RHS GEMM sharded across a 2-D task grid: tasks starting
// mid-NC-block must rebase their panel column onto the full-matrix layout.

// CHECK-LABEL: func.func @threaded_prepacked_gemm(
// CHECK: arith.constant 8 : i64
// CHECK: call @annc_threadpool_parallel_for_gemm
// CHECK-LABEL: func.func private @__annc_gemm_thread_task_0(
// CHECK-SAME: i64
// CHECK: arith.divsi
// CHECK: arith.remsi
// CHECK: linalg.matmul
// CHECK-SAME: annc.aarch64.gemm_problem = {
// CHECK-SAME: n = 64 : i64
// CHECK-SAME: annc.aarch64.gemm_stage = "kernel_tiled"
// CHECK: return

// LEAF-LABEL: func.func @threaded_prepacked_gemm(
// LEAF: call @annc_threadpool_parallel_for_gemm
// LEAF-LABEL: func.func private @__annc_gemm_thread_task_0(
// LEAF: arith.divui
// LEAF: arith.select
// LEAF: arith.subi
// LEAF: arith.andi
// LEAF: memref.get_global @packed_thread_task_data
// LEAF-NOT: __annc_aarch64_gemm_pack_b_leaf
// LEAF: call @__annc_aarch64_gemm_microkernel_leaf
// LEAF: return

// SCHED: call @annc_threadpool_parallel_for_gemm
// SCHED-LABEL: func.func private @__annc_gemm_thread_task_0(
// SCHED: llvm.call @annc_aarch64_neon_kernel_
func.func @threaded_prepacked_gemm(
    %c: memref<32x64xf32>, %a: memref<32x16xf32>, %b: memref<16x64xf32>) attributes {annc.aarch64.gemm_strategies = {s0 = {version = 4 : i64, target_arch = "hip09", isa = "neon", data_type = "f32", kernel_family = "annc-neon-f32-v1", cache_strategy = "fixed", kernel_strategy = "fixed", cache_parameters = {mc = 32 : i64, nc = 32 : i64, kc = 16 : i64}, kernel_parameters = {mr = 4 : i64, panel_lanes = 1 : i64}, execution_kind = "gemm", rhs_packing = "prepacked"}}} {
  linalg.matmul {
    annc.aarch64.gemm_problem = {version = 1 : i64, m = 32 : i64, n = 64 : i64, k = 16 : i64, lda = 16 : i64, ldb = 64 : i64, ldc = 64 : i64}, annc.aarch64.gemm_strategy = "s0", annc.aarch64.gemm_candidate = {version = 4 : i64, planning_state = "planned", strategy = "s0", max_thread_count = 8 : i64, thread_count = 8 : i64, tasks_m = 2 : i64, tasks_n = 4 : i64, shard_direction = "rows", thread_partition = "static-2d"}, annc.aarch64.prepacked_rhs = {version = 2 : i64, k = 16 : i64, n = 64 : i64, kc = 16 : i64, nc = 32 : i64, data_symbol = "packed_thread_task_data", data_elements = 1024 : i64}
  } ins(%a, %b : memref<32x16xf32>, memref<16x64xf32>)
    outs(%c : memref<32x64xf32>)
  return
}
