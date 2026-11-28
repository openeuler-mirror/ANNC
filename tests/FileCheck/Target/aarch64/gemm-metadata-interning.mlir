// RUN: annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-test.json" -aarch64-gemm-thread-planning | FileCheck %s
// RUN: annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-test.json" | sed 's/strategy = "s0"/strategy = "missing"/g' | not annc-asm - -aarch64-gemm-thread-planning 2>&1 | FileCheck %s --check-prefix=BAD-STRATEGY
// RUN: annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-test.json" | sed 's/strategy = "s0", //g' | not annc-asm - -aarch64-gemm-thread-planning 2>&1 | FileCheck %s --check-prefix=BAD-STRATEGY
// RUN: annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-test.json" | sed 's/cost_model = "c0"/cost_model = "missing"/g' | not annc-asm - -aarch64-gemm-thread-planning 2>&1 | FileCheck %s --check-prefix=BAD-COST
// RUN: annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-test.json" | sed 's/cost_model = "c0", //g' | not annc-asm - -aarch64-gemm-thread-planning 2>&1 | FileCheck %s --check-prefix=BAD-COST
// RUN: annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-test.json" -aarch64-gemm-thread-planning | sed 's/strategy = "s0"/strategy = "missing"/g' | not annc-asm - -aarch64-gemm-thread-tiling 2>&1 | FileCheck %s --check-prefix=BAD-STRATEGY

// CHECK-LABEL: func.func @shared_metadata(
// CHECK-SAME: annc.aarch64.gemm_strategies = {s0 = {
// CHECK-NOT: s1 =
// CHECK: linalg.matmul
// CHECK-SAME: annc.aarch64.gemm_candidate = {
// CHECK-SAME: strategy = "s0"
// CHECK: linalg.matmul
// CHECK-SAME: annc.aarch64.gemm_candidate = {
// CHECK-SAME: strategy = "s0"
// BAD-STRATEGY: invalid GEMM reference
// BAD-STRATEGY-SAME: annc.aarch64.gemm_strategies
// BAD-STRATEGY-NOT: Stack dump
// BAD-COST: invalid GEMM reference
// BAD-COST-SAME: annc.aarch64.gemm_cost_models
// BAD-COST-NOT: Stack dump
func.func @shared_metadata(
    %c0: memref<12x20xf32>, %c1: memref<12x20xf32>,
    %a0: memref<12x8xf32>, %b0: memref<8x20xf32>,
    %a1: memref<12x8xf32>, %b1: memref<8x20xf32>) {
  linalg.matmul ins(%a0, %b0 : memref<12x8xf32>, memref<8x20xf32>)
    outs(%c0 : memref<12x20xf32>)
  linalg.matmul ins(%a1, %b1 : memref<12x8xf32>, memref<8x20xf32>)
    outs(%c1 : memref<12x20xf32>)
  return
}
