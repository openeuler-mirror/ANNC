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
    %b: memref<8x20xf32>) {
  linalg.matmul {
    annc.aarch64.gemm_plan = {
      first_kc_mode = "overwrite",
      data_type = "f32",
      k = 8 : i64,
      kc = 4 : i64,
      lda = 8 : i64,
      ldb = 20 : i64,
      ldc = 20 : i64,
      m = 12 : i64,
      macro_order = "mkn",
      mc = 6 : i64,
      mr = 3 : i64,
      n = 20 : i64,
      nc = 16 : i64,
      next_kc_mode = "accumulate",
      panel_lanes = 2 : i64,
      rhs_packing = "packed",
      thread_count = 1 : i64,
      thread_partition = "serial",
      vector_length_bytes = 32 : i64,
      version = 1 : i64
    }
  } ins(%a, %b : memref<12x8xf32>, memref<8x20xf32>)
    outs(%c : memref<12x20xf32>)
  return
}
