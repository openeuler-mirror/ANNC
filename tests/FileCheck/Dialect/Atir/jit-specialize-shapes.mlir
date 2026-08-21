// RUN: annc-opt %s --atir-select-kernel='kernel-name=fused_matmul_a' --atir-specialize-shapes='shape-spec=%S/Inputs/jit-runtime-shapes.json' | FileCheck %s

module {
  func.func private @fused_matmul_a(
      %lhs: !atir.tensor<?x8xf16>,
      %rhs: !atir.tensor<8x4xf16>,
      %out: !atir.tensor<?x4xf16>) attributes {
        annc.kernel,
        fusion.pattern = "matmul",
        llvm.emit_c_interface
      } {
    %buffer = "atir.buffer"() : () -> !atir.tensor<?x4xf16>
    %0 = "atir.MatMul"(%buffer, %lhs, %rhs) <{
      do_relu = false,
      left_transpose = false,
      output_transpose = false,
      relu_limit = -1.0 : f32,
      right_transpose = false,
      withBias = false
    }> : (!atir.tensor<?x4xf16>, !atir.tensor<?x8xf16>,
          !atir.tensor<8x4xf16>) -> !atir.tensor<?x4xf16>
    return
  }

  func.func private @fused_matmul_b(
      %lhs: !atir.tensor<?x16xf32>,
      %rhs: !atir.tensor<16x2xf32>,
      %out: !atir.tensor<?x2xf32>) attributes {
        annc.kernel,
        fusion.pattern = "matmul",
        llvm.emit_c_interface
      } {
    return
  }
}

// CHECK-LABEL: func.func private @fused_matmul_a(
// CHECK-SAME: %[[LHS:.*]]: !atir.tensor<3x8xf16>
// CHECK-SAME: %[[RHS:.*]]: !atir.tensor<8x4xf16>
// CHECK-SAME: %[[OUT:.*]]: !atir.tensor<3x4xf16>
// CHECK: %[[BUFFER:.*]] = "atir.buffer"() : () -> !atir.tensor<3x4xf16>
// CHECK: "atir.MatMul"(%[[BUFFER]], %[[LHS]], %[[RHS]])
// CHECK-SAME: (!atir.tensor<3x4xf16>, !atir.tensor<3x8xf16>, !atir.tensor<8x4xf16>) -> !atir.tensor<3x4xf16>
// CHECK-NOT: func.func private @fused_matmul_b
