// RUN: annc-tf2atir-v2 --help >/dev/null && annc-asm %s -annc-gemm-epilogue-commit --verify-each | FileCheck %s

module {
  func.func @main(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>,
                  %bias: tensor<4xf32>, %scale: tensor<4xf32>)
      -> tensor<2x4xf32> {
    %dot = stablehlo.dot %lhs, %rhs, precision = [DEFAULT, DEFAULT]
        : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %dot_view = stablehlo.broadcast_in_dim %dot, dims = [0, 1]
        : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %bias_view = stablehlo.broadcast_in_dim %bias, dims = [1]
        : (tensor<4xf32>) -> tensor<2x4xf32>
    %add = stablehlo.add %dot_view, %bias_view : tensor<2x4xf32>
    %scale_view = stablehlo.broadcast_in_dim %scale, dims = [1]
        : (tensor<4xf32>) -> tensor<2x4xf32>
    %add_view = stablehlo.broadcast_in_dim %add, dims = [0, 1]
        : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %mul = stablehlo.multiply %scale_view, %add_view : tensor<2x4xf32>
    return %mul : tensor<2x4xf32>
  }
}

// CHECK-LABEL: func.func @main
// CHECK: atir.fused_matmul %{{.*}}, %{{.*}}, %arg2, %arg3
// CHECK-SAME: annc.gemm.epilogue = {output_role = "main_d", steps = [{broadcast = "n", input = 1 : i64, opcode = "add"}, {broadcast = "n", input = 0 : i64, opcode = "mul"}]
// CHECK: return %{{.*}} : tensor<2x4xf32>
// CHECK-NOT: stablehlo.dot
// CHECK-NOT: stablehlo.add
// CHECK-NOT: stablehlo.multiply
