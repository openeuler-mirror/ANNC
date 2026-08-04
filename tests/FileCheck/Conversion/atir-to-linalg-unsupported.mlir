// RUN: not annc-asm %s -convert-atir-to-linalg 2>&1 | FileCheck %s

// CHECK: error: 'atir.MatMul' op Linalg MatMul lowering does not support transpose
func.func @reject_transpose(
    %c: !atir.tensor<4x2xf32>,
    %a: !atir.tensor<2x3xf32>,
    %b: !atir.tensor<3x4xf32>) {
  %0 = "atir.MatMul"(%c, %a, %b) <{
    left_transpose = false,
    output_transpose = true,
    right_transpose = false,
    withBias = false
  }> : (!atir.tensor<4x2xf32>, !atir.tensor<2x3xf32>,
        !atir.tensor<3x4xf32>) -> !atir.tensor<4x2xf32>
  return
}
