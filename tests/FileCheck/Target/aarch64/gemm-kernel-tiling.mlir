// RUN: annc-asm %s -aarch64-gemm-kernel-tiling | FileCheck %s

// CHECK-DAG: #[[UNROLL:.*]] = #llvm.loop_unroll<disable = true>
// CHECK-DAG: #[[NO_UNROLL:.*]] = #llvm.loop_annotation<unroll = #[[UNROLL]]>
// CHECK-LABEL: func.func @kernel_tiling(
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK: scf.for %[[IR:.*]] = %[[C0]] to {{.*}} step %[[C3]] {
// CHECK: %[[JR0:.*]] = arith.constant 0 : index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: scf.for %[[JR:.*]] = %[[JR0]] to {{.*}} step %[[C16]] {
// CHECK: linalg.matmul
// CHECK-SAME: annc.aarch64.gemm_stage = "kernel_tiled"
// CHECK: } {loop_annotation = #[[NO_UNROLL]]}
// CHECK: linalg.matmul
// CHECK-SAME: annc.aarch64.gemm_stage = "kernel_tiled"
// CHECK: } {loop_annotation = #[[NO_UNROLL]]}
// CHECK: return
func.func @kernel_tiling(%c: memref<12x72xf32>, %a: memref<12x8xf32>,
                         %b: memref<8x72xf32>) {
  linalg.matmul {
    annc.aarch64.gemm_stage = "cache_blocked",
    annc.aarch64.kc_mode = "overwrite",
    annc.aarch64.gemm_plan = {
      first_kc_mode = "overwrite",
      data_type = "f32", k = 8 : i64,
      kc = 8 : i64, lda = 8 : i64, ldb = 72 : i64, ldc = 72 : i64,
      m = 12 : i64, macro_order = "mkn", mc = 12 : i64,
      mr = 3 : i64, n = 72 : i64, nc = 24 : i64,
      next_kc_mode = "accumulate", panel_lanes = 2 : i64,
      rhs_packing = "packed", thread_count = 1 : i64,
      thread_partition = "serial", vector_length_bytes = 32 : i64,
      version = 1 : i64
    }
  } ins(%a, %b : memref<12x8xf32>, memref<8x72xf32>)
    outs(%c : memref<12x72xf32>)
  return
}
