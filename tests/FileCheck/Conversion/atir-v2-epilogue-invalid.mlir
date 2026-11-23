// RUN: not annc-asm %s --convert-atir-v2-to-linalg 2>&1 | FileCheck %s

func.func @invalid_description(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
  %0 = atir.fused_matmul %lhs, %rhs {
    annc.gemm.epilogue = {
      version = 2 : i64, output_role = "main_d", terminal_view = "identity",
      steps = [{opcode = "relu", input = 0 : i64, broadcast = "scalar"}]
    }
  } : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK: error: 'atir.fused_matmul' op invalid GEMM epilogue attribute
