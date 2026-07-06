// RUN: annc-asm %s -atir-select-lowering-strategy -convert-atir-to-linalg -annc-one-shot-bufferize -cache-parallel | FileCheck %s
// CacheParallel: 选策略、转换为 linalg、bufferize、M/N 维 affine.parallel 切块

// CHECK-LABEL: func @pipeline
// CHECK: affine.parallel
// CHECK: memref.subview
// CHECK: linalg.matmul
func.func @pipeline(
    %c: !atir.tensor<4x4xf32>,
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32> {
  %0 = "atir.MatMul"(%c, %a, %b) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.0 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>, !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  return %0 : !atir.tensor<4x4xf32>
}

// -----

// 空函数
// CHECK-LABEL: func @empty
func.func @empty() {
  return
}