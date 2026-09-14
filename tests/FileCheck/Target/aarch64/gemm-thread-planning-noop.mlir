// RUN: annc-asm %s -aarch64-gemm-thread-planning | FileCheck %s

// CHECK: planning_state = "planned"
// CHECK-SAME: tasks_m = 1 : i64
// CHECK-SAME: tasks_n = 2 : i64
// CHECK-SAME: thread_count = 2 : i64
func.func @already_planned(
    %c: memref<4x8xf32>, %a: memref<4x4xf32>, %b: memref<4x8xf32>) {
  linalg.matmul {
    annc.aarch64.gemm_candidate = {
      planning_state = "planned",
      thread_count = 2 : i64,
      tasks_m = 1 : i64,
      tasks_n = 2 : i64
    }
  } ins(%a, %b : memref<4x4xf32>, memref<4x8xf32>)
    outs(%c : memref<4x8xf32>)
  return
}
