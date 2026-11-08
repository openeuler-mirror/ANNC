// RUN: annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-hip09-neon-test.json intra-thread-count=16" -aarch64-gemm-thread-planning | FileCheck %s

// VecMat vectorizes N, not K. Reusing the MatVec reduction model undercounts
// each 16-column cache block's RHS and compute work and keeps this case serial.
// The output-column model instead selects four column tasks with this fixture.
// CHECK-LABEL: func.func @vecmat_columns
// CHECK-SAME: annc.aarch64.gemm_strategies = {s0 = {
// CHECK-SAME: execution_kind = "vector-matrix"
// CHECK-SAME: rhs_packing = "direct"
// CHECK: linalg.matmul
// CHECK-SAME: annc.aarch64.gemm_candidate = {
// CHECK-SAME: shard_direction = "columns"
// CHECK-SAME: tasks_m = 1 : i64
// CHECK-SAME: tasks_n = 4 : i64
// CHECK-SAME: thread_count = 4 : i64
func.func @vecmat_columns(%c: memref<1x128xf32>, %a: memref<1x400xf32>,
                          %b: memref<400x128xf32>) {
  linalg.matmul ins(%a, %b : memref<1x400xf32>, memref<400x128xf32>)
    outs(%c : memref<1x128xf32>)
  return
}
