// RUN: annc-asm %s --convert-atir-v2-to-linalg | FileCheck %s

module {
  func.func @fused(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %result = "atir.fused_matmul"(%lhs, %rhs) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }

  func.func @fused_epilogue(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>,
                            %bias: tensor<4xf32>) -> tensor<2x4xf32> {
    %result = "atir.fused_matmul"(%lhs, %rhs, %bias) {
      annc.gemm.epilogue = {
        output_role = "main_d",
        steps = [{broadcast = "n", input = 1 : i64, opcode = "add"},
                 {broadcast = "scalar", input = 0 : i64,
                  limit = -1.000000e+00 : f32, opcode = "relu"}],
        terminal_view = "identity", version = 1 : i64
      }
    } : (tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// CHECK-LABEL: func.func @fused(
// CHECK-SAME: -> memref<2x4xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2x4xf32>
// CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>) outs(%[[EMPTY]] : tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK: bufferization.to_memref
// CHECK: return {{.*}} : memref<2x4xf32>
// CHECK-LABEL: func.func @fused_epilogue(
// CHECK-SAME: -> memref<2x4xf32>
// CHECK: linalg.generic
// CHECK-SAME: ins(%arg0, %arg1, %arg2 : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>)
// CHECK: bufferization.to_memref
// CHECK: return {{.*}} : memref<2x4xf32>
