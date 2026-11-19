// RUN: annc-stablehlo-sink --help >/dev/null && annc-asm %s -annc-gemm-epilogue-commit --verify-each | FileCheck %s

module {
  func.func @bias_defined_after_dot(%lhs: tensor<2x3xf32>,
                                    %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %dot = stablehlo.dot %lhs, %rhs, precision = [DEFAULT, DEFAULT]
        : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %bias = stablehlo.constant dense<1.0> : tensor<4xf32>
    %bias_broadcast = stablehlo.broadcast_in_dim %bias, dims = [1]
        : (tensor<4xf32>) -> tensor<2x4xf32>
    %result = stablehlo.add %dot, %bias_broadcast : tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// CHECK-LABEL: func.func @bias_defined_after_dot(
// CHECK: %[[BIAS:.*]] = stablehlo.constant
// CHECK: %[[FUSED:.*]] = atir.fused_matmul %{{.*}}, %{{.*}}, %[[BIAS]]
// CHECK: return %[[FUSED]]
