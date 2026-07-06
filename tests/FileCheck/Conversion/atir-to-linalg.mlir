// RUN: annc-asm --split-input-file %s -convert-atir-to-linalg | FileCheck %s
// ConvertAtirToLinalg: atir op -> linalg

// CHECK-LABEL: func @matmul_to_linalg
// CHECK: memref
// CHECK: linalg.matmul
// CHECK-NOT: "atir.MatMul"
func.func @matmul_to_linalg(
    %c: !atir.tensor<4x4xf32>,
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32> {
  %0 = "atir.MatMul"(%c, %a, %b) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.0 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>, !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  return %0 : !atir.tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @empty
func.func @empty() {
  return
}