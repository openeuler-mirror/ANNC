// RUN: not annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-v2-test.json" 2>&1 | FileCheck %s

// CHECK: version 2 tuning config is no longer supported
// CHECK-SAME: replacing cache_tile with cache.strategy
func.func @v2_config(
    %c: memref<4x8xf32>, %a: memref<4x4xf32>, %b: memref<4x8xf32>) {
  linalg.matmul ins(%a, %b : memref<4x4xf32>, memref<4x8xf32>)
    outs(%c : memref<4x8xf32>)
  return
}
