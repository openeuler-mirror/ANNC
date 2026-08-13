// RUN: annc-asm --split-input-file %s -atir-gemm-epilogue-fusion | FileCheck %s

// CHECK-LABEL: func.func @matmul_bias_relu(
// CHECK-SAME: %[[A:.*]]: !atir.tensor<4x8xf32>, %[[B:.*]]: !atir.tensor<8x4xf32>, %[[C:.*]]: !atir.tensor<4x4xf32>, %[[BIAS:.*]]: !atir.tensor<4xf32>)
// CHECK-NOT: "atir.buffer"
// CHECK-NOT: "atir.Add"
// CHECK-NOT: "atir.Relu"
// CHECK: "atir.MatMul"(%[[C]], %[[A]], %[[B]], %[[BIAS]])
// CHECK-SAME: annc.gemm.epilogue = [{broadcast = "n", input = 0 : i64, kind = "bias_add"}, {kind = "relu", limit = -1.000000e+00 : f32}]
// CHECK: return
func.func @matmul_bias_relu(
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>,
    %c: !atir.tensor<4x4xf32>,
    %bias: !atir.tensor<4xf32>) -> !atir.tensor<4x4xf32> {
  %matmul_buffer = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %matmul = "atir.MatMul"(%matmul_buffer, %a, %b) <{left_transpose = false,
    output_transpose = false,right_transpose = false, withBias = false
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  %add_buffer = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %add = "atir.Add"(%add_buffer, %matmul, %bias) <{
    do_relu = false,
    relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>,
        !atir.tensor<4xf32>) -> !atir.tensor<4x4xf32>
  %relu = "atir.Relu"(%c, %add) <{relu_limit = -1.0 : f32}> :
      (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>) ->
      !atir.tensor<4x4xf32>
  return %relu : !atir.tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @matmul_bias(
// CHECK-NOT: "atir.buffer"
// CHECK-NOT: "atir.Add"
// CHECK: "atir.MatMul"
// CHECK-SAME: annc.gemm.epilogue = [{broadcast = "n", input = 0 : i64, kind = "bias_add"}]
// CHECK: return
func.func @matmul_bias(
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>,
    %c: !atir.tensor<4x4xf32>,
    %bias: !atir.tensor<4xf32>) -> !atir.tensor<4x4xf32> {
  %tmp = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{left_transpose = false,
    output_transpose = false,right_transpose = false, withBias = false
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  %add = "atir.Add"(%c, %matmul, %bias) <{
    do_relu = false,
    relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>,
        !atir.tensor<4xf32>) -> !atir.tensor<4x4xf32>
  return %add : !atir.tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @late_bias_definition(
// CHECK: "atir.buffer"
// CHECK: "atir.MatMul"
// CHECK-NOT: "atir.Add"
func.func @late_bias_definition(
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>,
    %c: !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32> {
  %tmp = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false,
    output_transpose = false,
    right_transpose = false,
    withBias = false
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  %bias = "atir.buffer"() : () -> !atir.tensor<4xf32>
  %add = "atir.Add"(%c, %matmul, %bias) <{
    do_relu = false,
    relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>,
        !atir.tensor<4xf32>) -> !atir.tensor<4x4xf32>
  return %add : !atir.tensor<4x4xf32>
}

// -----

// Legacy MatMul post-op attributes are not consumed by this pass.
// CHECK-LABEL: func.func @reject_legacy_matmul_postop(
// CHECK: "atir.MatMul"
// CHECK: do_relu = true
// CHECK: "atir.Add"
func.func @reject_legacy_matmul_postop(
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>,
    %c: !atir.tensor<4x4xf32>,
    %bias: !atir.tensor<4xf32>) -> !atir.tensor<4x4xf32> {
  %tmp = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    do_relu = true,
    left_transpose = false,
    output_transpose = false,
    relu_limit = -1.0 : f32,
    right_transpose = false,
    withBias = false
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  %add = "atir.Add"(%c, %matmul, %bias) <{
    do_relu = false,
    relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>,
        !atir.tensor<4xf32>) -> !atir.tensor<4x4xf32>
  return %add : !atir.tensor<4x4xf32>
}

// -----

// A rank-2 addend is not the supported N-axis bias epilogue.
// CHECK-LABEL: func.func @reject_non_bias_addend(
// CHECK: "atir.MatMul"
// CHECK: "atir.Add"
func.func @reject_non_bias_addend(
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>,
    %c: !atir.tensor<4x4xf32>,
    %addend: !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32> {
  %tmp = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{left_transpose = false,
    output_transpose = false,right_transpose = false, withBias = false
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  %add = "atir.Add"(%c, %matmul, %addend) <{
    do_relu = false,
    relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>,
        !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32>
  return %add : !atir.tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @reject_transposed_matmul(
// CHECK: "atir.MatMul"
// CHECK: "atir.Add"
func.func @reject_transposed_matmul(
    %a: !atir.tensor<2x3xf32>,
    %b: !atir.tensor<3x4xf32>,
    %c: !atir.tensor<4x2xf32>,
    %bias: !atir.tensor<2xf32>) -> !atir.tensor<4x2xf32> {
  %tmp = "atir.buffer"() : () -> !atir.tensor<4x2xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false,
    output_transpose = true,
    right_transpose = false,
    withBias = false
  }> : (!atir.tensor<4x2xf32>, !atir.tensor<2x3xf32>,
        !atir.tensor<3x4xf32>) -> !atir.tensor<4x2xf32>
  %add = "atir.Add"(%c, %matmul, %bias) <{
    do_relu = false,
    relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x2xf32>, !atir.tensor<4x2xf32>,
        !atir.tensor<2xf32>) -> !atir.tensor<4x2xf32>
  return %add : !atir.tensor<4x2xf32>
}
