// RUN: annc-asm --split-input-file %s -atir-gemm-epilogue-fusion -convert-atir-to-linalg | FileCheck %s
// ConvertAtirToLinalg: atir op -> linalg

// CHECK-LABEL: func @matmul_to_linalg
// CHECK: %[[BUFFER:.*]] = memref.alloca() : memref<4x4xf32>
// CHECK: bufferization.to_tensor %[[BUFFER]] restrict writable
// CHECK: linalg.matmul
// CHECK-NOT: "atir.MatMul"
func.func @matmul_to_linalg(
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32> {
  %c = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %0 = "atir.MatMul"(%c, %a, %b) <{left_transpose = false, output_transpose = false, right_transpose = false, withBias = false}> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>, !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  return %0 : !atir.tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @empty
func.func @empty() {
  return
}

// -----

// CHECK-LABEL: func @buffer_to_alloca
// CHECK: %[[BUFFER:.*]] = memref.alloca() : memref<4x4xf32>
// CHECK: bufferization.to_tensor %[[BUFFER]] restrict writable
// CHECK-NOT: "atir.buffer"
func.func @buffer_to_alloca() {
  %0 = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  return
}

// -----

// CHECK-LABEL: func @matmul_bias_relu_to_linalg
// CHECK-NOT: linalg.matmul
// CHECK: linalg.generic
// CHECK: annc.gemm
// CHECK: arith.maxnumf
func.func @matmul_bias_relu_to_linalg(
    %c: !atir.tensor<4x4xf32>, %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>, %bias: !atir.tensor<4xf32>)
    -> !atir.tensor<4x4xf32> {
  %matmul_buffer = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %matmul = "atir.MatMul"(%matmul_buffer, %a, %b) <{
    left_transpose = false, output_transpose = false, right_transpose = false,
    withBias = false
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  %add_buffer = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %add = "atir.Add"(%add_buffer, %matmul, %bias) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>, !atir.tensor<4xf32>)
      -> !atir.tensor<4x4xf32>
  %relu = "atir.Relu"(%c, %add) <{relu_limit = -1.0 : f32}> :
      (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32>
  return %relu : !atir.tensor<4x4xf32>
}
