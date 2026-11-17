// RUN: not annc-asm %s --convert-atir-v2-to-linalg 2>&1 | FileCheck %s

module {
  func.func @unsupported_m_vector(
      %lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>,
      %bias: tensor<2xf32>) -> tensor<2x4xf32> {
    %result = "atir.fused_matmul"(%lhs, %rhs, %bias) {
      annc.gemm.epilogue = {
        output_role = "main_d",
        steps = [{broadcast = "m", input = 0 : i64, opcode = "add"}],
        terminal_view = "identity", version = 1 : i64
      }
    } : (tensor<2x3xf32>, tensor<3x4xf32>, tensor<2xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// CHECK: error: 'atir.fused_matmul' op rank-1 M-axis epilogue vectors are unsupported; use a rank-2 Mx1 tensor
