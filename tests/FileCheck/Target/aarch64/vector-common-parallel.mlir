// RUN: annc-asm %s -atir-select-lowering-strategy -convert-atir-to-linalg -annc-one-shot-bufferize -vector-common-parallel | FileCheck %s
// VectorCommonParallel: 完整 pipeline —— 选策略 -> 转 linalg -> bufferize -> micro kernel vector 化

// CHECK-LABEL: func @pipeline
// CHECK: linalg.matmul
func.func @pipeline(
    %c: !atir.tensor<4x4xf32>,
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32> {
  %0 = "atir.MatMul"(%c, %a, %b) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.0 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>, !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  return %0 : !atir.tensor<4x4xf32>
}

// -----

// 空函数：pass 不修改
// CHECK-LABEL: func @empty
func.func @empty() {
  return
}