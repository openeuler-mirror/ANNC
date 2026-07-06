// RUN: annc-opt --split-input-file %s -atir-unroll | FileCheck %s
// Unroll: 对 lhs/rhs 都带 onchip 属性的 MatMul 展开为多个小 MatMul

// 展开后：原 MatMul 被替换成 内层 MatMul (带 k_size) + atir.Concat
// CHECK-LABEL: func @unroll_matmul
// CHECK: "atir.MatMul"
// CHECK-SAME: k_size = 4
// CHECK-SAME: m_size = 2
// CHECK: "atir.Concat"
func.func @unroll_matmul(
    %c: !atir.tensor<4x4xf32, onchip = #atir.Tiling<axes = [0, 1], start = [[0], [0]], size = [[2], [2]]>>,
    %a: !atir.tensor<4x8xf32, onchip = #atir.Tiling<axes = [0, 1], start = [[0], [0]], size = [[2], [4]]>>,
    %b: !atir.tensor<8x4xf32, onchip = #atir.Tiling<axes = [0, 1], start = [[0], [0]], size = [[4], [2]]>>) -> !atir.tensor<4x4xf32> {
  %0 = "atir.MatMul"(%c, %a, %b) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.0 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<4x4xf32, onchip = #atir.Tiling<axes = [0, 1], start = [[0], [0]], size = [[2], [2]]>>, !atir.tensor<4x8xf32, onchip = #atir.Tiling<axes = [0, 1], start = [[0], [0]], size = [[2], [4]]>>, !atir.tensor<8x4xf32, onchip = #atir.Tiling<axes = [0, 1], start = [[0], [0]], size = [[4], [2]]>>) -> !atir.tensor<4x4xf32>
  return %0 : !atir.tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @no_tiling
// CHECK-NOT: k_size =
// CHECK-NOT: atir.Concat
func.func @no_tiling(
    %c: !atir.tensor<4x4xf32>,
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32> {
  %0 = "atir.MatMul"(%c, %a, %b) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.0 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>, !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  return %0 : !atir.tensor<4x4xf32>
}