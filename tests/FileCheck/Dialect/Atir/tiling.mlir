// RUN: annc-opt --split-input-file %s -atir-tiling | FileCheck %s
// Tiling: 给 MatMul 的 lhs/rhs tensor 类型加上 onchip_parallel tiling 属性

// CHECK-LABEL: func @matmul_tiled
// CHECK-SAME: onchip = #atir.Tiling
func.func @matmul_tiled(
    %c: !atir.tensor<64x64xf32>,
    %a: !atir.tensor<64x128xf32>,
    %b: !atir.tensor<128x64xf32>) -> !atir.tensor<64x64xf32> {
  // CHECK: "atir.MatMul"
  // CHECK: onchip = #atir.Tiling
  %0 = "atir.MatMul"(%c, %a, %b) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.0 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<64x64xf32>, !atir.tensor<64x128xf32>, !atir.tensor<128x64xf32>) -> !atir.tensor<64x64xf32>
  return %0 : !atir.tensor<64x64xf32>
}

// -----

// 空函数
// CHECK-LABEL: func @no_matmul
func.func @no_matmul() {
  // CHECK-NOT: onchip
  return
}