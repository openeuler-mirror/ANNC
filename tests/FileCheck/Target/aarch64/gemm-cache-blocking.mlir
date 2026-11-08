// RUN: annc-asm %s -aarch64-gemm-cache-blocking | FileCheck %s

// CHECK-DAG: #[[UNROLL:.*]] = #llvm.loop_unroll<disable = true>
// CHECK-DAG: #[[NO_UNROLL:.*]] = #llvm.loop_annotation<unroll = #[[UNROLL]]>
// CHECK-LABEL: func.func @blocked_gemm(
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[CM:.*]] = arith.constant 12 : index
// CHECK-DAG: %[[CMC:.*]] = arith.constant 6 : index
// CHECK: scf.for %[[IC:.*]] = %[[C0]] to %[[CM]] step %[[CMC]] {
// CHECK: linalg.matmul
// CHECK-SAME: annc.aarch64.gemm_stage = "cache_blocked"
// CHECK-SAME: annc.aarch64.kc_mode = "overwrite"
// CHECK: linalg.matmul
// CHECK-SAME: annc.aarch64.gemm_stage = "cache_blocked"
// CHECK-SAME: annc.aarch64.kc_mode = "overwrite"
// CHECK: linalg.matmul
// CHECK-SAME: annc.aarch64.gemm_stage = "cache_blocked"
// CHECK-SAME: annc.aarch64.kc_mode = "accumulate"
// CHECK: linalg.matmul
// CHECK-SAME: annc.aarch64.gemm_stage = "cache_blocked"
// CHECK-SAME: annc.aarch64.kc_mode = "accumulate"
// CHECK: } {loop_annotation = #[[NO_UNROLL]]}
// CHECK: return
func.func @blocked_gemm(
    %c: memref<12x20xf32>,
    %a: memref<12x8xf32>,
    %b: memref<8x20xf32>) attributes {annc.aarch64.gemm_strategies = {s0 = {version = 4 : i64, target_arch = "hip12", isa = "neon", data_type = "f32", kernel_family = "annc-neon-f32-v1", cache_strategy = "fixed", kernel_strategy = "fixed", cache_parameters = {mc = 6 : i64, nc = 16 : i64, kc = 4 : i64}, kernel_parameters = {mr = 3 : i64, panel_lanes = 2 : i64}, execution_kind = "gemm", rhs_packing = "packed"}}} {
  linalg.matmul {
    annc.aarch64.gemm_problem = {version = 1 : i64, m = 12 : i64, n = 20 : i64, k = 8 : i64, lda = 8 : i64, ldb = 20 : i64, ldc = 20 : i64}, annc.aarch64.gemm_strategy = "s0", annc.aarch64.gemm_candidate = {version = 4 : i64, planning_state = "planned", strategy = "s0", max_thread_count = 1 : i64, thread_count = 1 : i64, tasks_m = 1 : i64, tasks_n = 1 : i64, shard_direction = "rows", thread_partition = "static-2d"}
  } ins(%a, %b : memref<12x8xf32>, memref<8x20xf32>)
    outs(%c : memref<12x20xf32>)
  return
}
