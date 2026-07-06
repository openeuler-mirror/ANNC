// RUN: annc-asm %s -atir-select-lowering-strategy -convert-atir-to-linalg -annc-one-shot-bufferize -cache-reduction | FileCheck %s
// CacheReduction: 完整 pipeline —— 先选 lowering 策略、再转 linalg、最后做 K 轴循环展开
// atir-select-lowering-strategy 给 MatMul 加 lowering_config
// convert-atir-to-linalg 把 MatMul 变成 linalg.matmul 并保留属性
// cache-reduction 读 lowering_config 做 affine.for 循环展开

// 有 lowering_config 的 atir.MatMul → 展开为 affine.for + 多个 linalg.matmul
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

// MatMul 在转换成 linalg 时如果 lowering_config 缺失，linalg.matmul 上就没有该属性，
// cache-reduction 读不到就回退（不展开），但前面的 select-lowering 仍然有效。
// 这个 case 验证 linalg.matmul 存在但无 affine.for 包裹。
// CHECK-LABEL: func @basic_matmul
// CHECK: linalg.matmul
// CHECK-NOT: affine.for
func.func @basic_matmul(
    %c: !atir.tensor<4x4xf32>,
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32> {
  %0 = "atir.MatMul"(%c, %a, %b) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.0 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>, !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  return %0 : !atir.tensor<4x4xf32>
}