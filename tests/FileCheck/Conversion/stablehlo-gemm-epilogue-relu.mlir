// RUN: annc-stablehlo-sink --help >/dev/null && annc-asm --split-input-file %s -annc-gemm-epilogue-commit --verify-each | FileCheck %s

// -----

func.func @matmul_bias_relu(
    %lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>, %bias: tensor<4xf32>)
    -> tensor<2x4xf32> {
  %dot = stablehlo.dot %lhs, %rhs, precision = [DEFAULT, DEFAULT]
      : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  %bias_view = stablehlo.broadcast_in_dim %bias, dims = [1]
      : (tensor<4xf32>) -> tensor<2x4xf32>
  %add = stablehlo.add %dot, %bias_view : tensor<2x4xf32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %zero_view = stablehlo.broadcast_in_dim %zero, dims = []
      : (tensor<f32>) -> tensor<2x4xf32>
  %relu = stablehlo.maximum %add, %zero_view : tensor<2x4xf32>
  return %relu : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @matmul_bias_relu
// CHECK: atir.fused_matmul
// CHECK-SAME: steps = [{broadcast = "n", input = 1 : i64, opcode = "add"}, {broadcast = "scalar", input = 0 : i64, opcode = "relu"}]
// CHECK-NOT: stablehlo.maximum

// -----

func.func @nonzero_maximum_is_not_captured(
    %lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
  %dot = stablehlo.dot %lhs, %rhs, precision = [DEFAULT, DEFAULT]
      : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  %limit = stablehlo.constant dense<1.0> : tensor<f32>
  %limit_view = stablehlo.broadcast_in_dim %limit, dims = []
      : (tensor<f32>) -> tensor<2x4xf32>
  %max = stablehlo.maximum %dot, %limit_view : tensor<2x4xf32>
  return %max : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @nonzero_maximum_is_not_captured
// CHECK: %[[MAX_DOT:[^ ]+]] = atir.fused_matmul
// CHECK-NOT: annc.gemm.epilogue
// CHECK: stablehlo.maximum %[[MAX_DOT]],

// -----

func.func @m_axis_vector_is_not_captured(
    %lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>, %bias: tensor<2xf32>)
    -> tensor<2x4xf32> {
  %dot = stablehlo.dot %lhs, %rhs, precision = [DEFAULT, DEFAULT]
      : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  %bias_view = stablehlo.broadcast_in_dim %bias, dims = [0]
      : (tensor<2xf32>) -> tensor<2x4xf32>
  %add = stablehlo.add %dot, %bias_view : tensor<2x4xf32>
  return %add : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @m_axis_vector_is_not_captured
// CHECK: %[[ROW_DOT:[^ ]+]] = atir.fused_matmul
// CHECK-NOT: annc.gemm.epilogue
// CHECK: stablehlo.add %[[ROW_DOT]],
