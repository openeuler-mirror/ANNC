// RUN: annc-asm --split-input-file %s -atir-fast-codegen -lower-context-kernel-entries -convert-atir-to-linalg | FileCheck %s
// RUN: annc-asm --split-input-file %s -atir-fast-codegen | FileCheck %s --check-prefix=FAST

// CHECK-LABEL: func.func private @fused_matmul_add_relu(
// CHECK-SAME: %[[CTX:[A-Za-z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: %[[STATUS:[A-Za-z0-9_]+]] = call @[[CALLEE:ANNCKernel_ContextV3_[A-Za-z0-9_]+]](%[[CTX]]) : (!llvm.ptr) -> i32
// CHECK: return %[[STATUS]] : i32
// CHECK-NOT: memref.alloc
// FAST-LABEL: func.func private @fused_matmul_add_relu
// FAST: atir.Customize
// FAST-SAME: custom.op_name = "MatMulAddReluWithAddOutput"
// FAST-NOT: "atir.MatMul"
// FAST-NOT: "atir.Add"
// FAST-NOT: "atir.Relu"
module attributes {module.state = "atir"} {
  func.func private @fused_matmul_add_relu(
      %lhs: !atir.tensor<2x2xf32, name = "lhs:0">,
      %rhs: !atir.tensor<2x2xf32, name = "rhs:0">,
      %bias: !atir.tensor<2xf32, name = "bias:0">)
      -> (!atir.tensor<2x2xf32, name = "bias_add:0">,
          !atir.tensor<2x2xf32, name = "relu:0">)
      attributes {
        annc.kernel,
        fusion.metadata = {abi = "annc_context_v3"}
      } {
    %matmul_out = "atir.buffer"() : () -> !atir.tensor<2x2xf32>
    %add_out = "atir.buffer"() : () -> !atir.tensor<2x2xf32, name = "bias_add:0">
    %relu_out = "atir.buffer"() : () -> !atir.tensor<2x2xf32, name = "relu:0">
    %matmul = "atir.MatMul"(%matmul_out, %lhs, %rhs) <{
      do_relu = false,
      left_transpose = false,
      output_transpose = false,
      relu_limit = -1.0 : f32,
      right_transpose = false,
      withBias = false
    }> : (!atir.tensor<2x2xf32>, !atir.tensor<2x2xf32, name = "lhs:0">,
          !atir.tensor<2x2xf32, name = "rhs:0">) -> !atir.tensor<2x2xf32>
    %add = "atir.Add"(%add_out, %matmul, %bias) <{
      do_relu = false,
      relu_limit = -1.0 : f32
    }> : (!atir.tensor<2x2xf32, name = "bias_add:0">,
          !atir.tensor<2x2xf32>, !atir.tensor<2xf32, name = "bias:0">)
        -> !atir.tensor<2x2xf32, name = "bias_add:0">
    %relu = "atir.Relu"(%relu_out, %add) <{relu_limit = -1.0 : f32}> :
        (!atir.tensor<2x2xf32, name = "relu:0">,
         !atir.tensor<2x2xf32, name = "bias_add:0">)
        -> !atir.tensor<2x2xf32, name = "relu:0">
    return %add, %relu : !atir.tensor<2x2xf32, name = "bias_add:0">,
                         !atir.tensor<2x2xf32, name = "relu:0">
  }
}
