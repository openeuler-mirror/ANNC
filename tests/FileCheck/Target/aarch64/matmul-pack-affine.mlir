// RUN: annc-asm %s -atir-select-lowering-strategy -convert-atir-to-linalg -annc-one-shot-bufferize -matmul-pack-affine | FileCheck %s
// MatmulPackAffine: 完整 pipeline —— 选策略 -> 转 linalg -> bufferize -> 用 block 大小做 packing tiling

// CHECK-LABEL: func @pipeline
// CHECK: affine.for
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
}// RUN: annc-asm %s -matmul-pack-affine | FileCheck %s
// MatmulPackAffine: 有 lowering_config 时用 block 大小做 packing tiling

// CHECK-LABEL: func @smoke
func.func @smoke(
    %c: tensor<4x4xf32>,
    %a: tensor<4x8xf32>,
    %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
  // CHECK: linalg.matmul
  // CHECK-NOT: affine.for
  %0 = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x4xf32>) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}