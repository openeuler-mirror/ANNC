// RUN: annc-opt --split-input-file %s -atir-op-fusion | FileCheck %s
// OpFusion: 将 MatMulOp 替换为 call @fused_* + private kernel 函数
// MatMul operand 顺序: (C=输出buffer, lhs, rhs)

// main 函数内部有 matmul，触发 fusion
// CHECK-LABEL: func @main
func.func @main(
    %a: !atir.tensor<64x128xf32>,
    %b: !atir.tensor<128x64xf32>,
    %c: !atir.tensor<64x64xf32>) -> !atir.tensor<64x64xf32> {
  %0 = "atir.MatMul"(%c, %a, %b) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.0 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<64x64xf32>, !atir.tensor<64x128xf32>, !atir.tensor<128x64xf32>) -> !atir.tensor<64x64xf32>
  // CHECK: call @fused_matmul
  // CHECK: func.func private @fused_matmul
  // CHECK: annc.execution_mode = "jit"
  // CHECK: fusion.metadata
  // CHECK: args = [
  // CHECK: execution_mode = "jit"
  // CHECK: outputs = [
  // CHECK: template_fingerprint =
  // CHECK-NOT: tf.inputs
  // CHECK-NOT: tf.input_shapes
  // CHECK-NOT: tf.output
  // CHECK-NOT: Nconstants
  // CHECK-NOT: Nfixed
  // CHECK-NOT: Ndynamic
  // CHECK-NOT: num_outputs
  // CHECK-NOT: input_ranks
  // CHECK-NOT: output_ranks
  // CHECK-NOT: output_shapes
  return %0 : !atir.tensor<64x64xf32>
}

// -----

// 函数不叫 main，不触发
// CHECK-LABEL: func @other
func.func @other(
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>,
    %c: !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32> {
  %0 = "atir.MatMul"(%c, %a, %b) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.0 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>, !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  // CHECK: "atir.MatMul"
  // CHECK-NOT: call @fused_matmul
  return %0 : !atir.tensor<4x4xf32>
}

// -----

// main 函数无 MatMul，不触发
// CHECK-LABEL: func @main
func.func @main(
    %c: !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32> {
  // CHECK: return
  // CHECK-NOT: call @fused_
  return %c : !atir.tensor<4x4xf32>
}

// -----

// Add escapes through a non-return consumer and Relu escapes through the
// return. Materialize both boundary values in the v2 private kernel.
// CHECK-LABEL: func.func @main
// CHECK: "atir.MatMul"
// CHECK-SAME: annc.fusion_materialized
// CHECK: "atir.Add"
// CHECK: "atir.Relu"
// CHECK-NOT: call @fused_matmul_add_relu
// CHECK: return %{{.*}}, %{{.*}}
// CHECK-LABEL: func.func private @fused_matmul_add_relu
// CHECK-SAME: %[[CTX:.*]]: !llvm.ptr,
// CHECK-SAME: !atir.tensor<2x3xf32, name = "lhs:0">,
// CHECK-SAME: !atir.tensor<3x4xf32, name = "rhs:0">,
// CHECK-SAME: !atir.tensor<4xf32, name = "bias:0">)
// CHECK-SAME: -> (!atir.tensor<2x4xf32, name = "bias_add:0">, !atir.tensor<2x4xf32, name = "relu:0">)
// CHECK: fusion.metadata = {
// CHECK-SAME: abi = "annc_execution_v2"
// CHECK-SAME: kernel_arg_order = []
// CHECK-SAME: tf_name = "bias_add:0"
// CHECK-SAME: tf_name = "relu:0"
// CHECK: return %{{.*}}, %{{.*}} : !atir.tensor<2x4xf32, name = "bias_add:0">, !atir.tensor<2x4xf32, name = "relu:0">
func.func @main(
    %lhs: !atir.tensor<2x3xf32, name = "lhs:0">,
    %rhs: !atir.tensor<3x4xf32, name = "rhs:0">,
    %bias: !atir.tensor<4xf32, name = "bias:0">,
    %external_bias: !atir.tensor<4xf32, name = "external_bias:0">)
    -> (!atir.tensor<2x4xf32, name = "external_add:0">,
        !atir.tensor<2x4xf32, name = "relu:0">) {
  %matmul_out = "atir.buffer"() : () -> !atir.tensor<2x4xf32>
  %add_out = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "bias_add:0">
  %relu_out = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "relu:0">
  %external_out = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "external_add:0">
  %matmul = "atir.MatMul"(%matmul_out, %lhs, %rhs) <{
    do_relu = false,
    left_transpose = false,
    output_transpose = false,
    relu_limit = -1.0 : f32,
    right_transpose = false,
    withBias = false
  }> : (!atir.tensor<2x4xf32>, !atir.tensor<2x3xf32, name = "lhs:0">,
        !atir.tensor<3x4xf32, name = "rhs:0">)
      -> !atir.tensor<2x4xf32>
  %add = "atir.Add"(%add_out, %matmul, %bias) <{
    do_relu = false,
    relu_limit = -1.0 : f32
  }> : (!atir.tensor<2x4xf32, name = "bias_add:0">,
        !atir.tensor<2x4xf32>, !atir.tensor<4xf32, name = "bias:0">)
      -> !atir.tensor<2x4xf32, name = "bias_add:0">
  %relu = "atir.Relu"(%relu_out, %add) <{relu_limit = -1.0 : f32}> :
      (!atir.tensor<2x4xf32, name = "relu:0">,
       !atir.tensor<2x4xf32, name = "bias_add:0">)
      -> !atir.tensor<2x4xf32, name = "relu:0">
  %external = "atir.Add"(%external_out, %add, %external_bias) <{
    do_relu = false,
    relu_limit = -1.0 : f32
  }> : (!atir.tensor<2x4xf32, name = "external_add:0">,
        !atir.tensor<2x4xf32, name = "bias_add:0">,
        !atir.tensor<4xf32, name = "external_bias:0">)
      -> !atir.tensor<2x4xf32, name = "external_add:0">
  return %external, %relu : !atir.tensor<2x4xf32, name = "external_add:0">,
                            !atir.tensor<2x4xf32, name = "relu:0">
}

// -----

// CHECK-LABEL: func.func @ambiguous_add
// CHECK-NOT: annc.fusion_materialized
// CHECK-NOT: annc_execution_v2
func.func @ambiguous_add(
    %lhs: !atir.tensor<2x2xf32>, %rhs: !atir.tensor<2x2xf32>,
    %bias0: !atir.tensor<2xf32>, %bias1: !atir.tensor<2xf32>)
    -> (!atir.tensor<2x2xf32>, !atir.tensor<2x2xf32>) {
  %matmul_out = "atir.buffer"() : () -> !atir.tensor<2x2xf32>
  %add0_out = "atir.buffer"() : () -> !atir.tensor<2x2xf32>
  %add1_out = "atir.buffer"() : () -> !atir.tensor<2x2xf32>
  %matmul = "atir.MatMul"(%matmul_out, %lhs, %rhs) <{
    do_relu = false, left_transpose = false, output_transpose = false,
    relu_limit = -1.0 : f32, right_transpose = false, withBias = false
  }> : (!atir.tensor<2x2xf32>, !atir.tensor<2x2xf32>,
        !atir.tensor<2x2xf32>) -> !atir.tensor<2x2xf32>
  %add0 = "atir.Add"(%add0_out, %matmul, %bias0) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<2x2xf32>, !atir.tensor<2x2xf32>,
        !atir.tensor<2xf32>) -> !atir.tensor<2x2xf32>
  %add1 = "atir.Add"(%add1_out, %matmul, %bias1) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<2x2xf32>, !atir.tensor<2x2xf32>,
        !atir.tensor<2xf32>) -> !atir.tensor<2x2xf32>
  return %add0, %add1 : !atir.tensor<2x2xf32>, !atir.tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func.func @ambiguous_relu
// CHECK-NOT: annc.fusion_materialized
// CHECK-NOT: annc_execution_v2
func.func @ambiguous_relu(
    %lhs: !atir.tensor<2x2xf32>, %rhs: !atir.tensor<2x2xf32>,
    %bias: !atir.tensor<2xf32>)
    -> (!atir.tensor<2x2xf32>, !atir.tensor<2x2xf32>) {
  %matmul_out = "atir.buffer"() : () -> !atir.tensor<2x2xf32>
  %add_out = "atir.buffer"() : () -> !atir.tensor<2x2xf32>
  %relu0_out = "atir.buffer"() : () -> !atir.tensor<2x2xf32>
  %relu1_out = "atir.buffer"() : () -> !atir.tensor<2x2xf32>
  %matmul = "atir.MatMul"(%matmul_out, %lhs, %rhs) <{
    do_relu = false, left_transpose = false, output_transpose = false,
    relu_limit = -1.0 : f32, right_transpose = false, withBias = false
  }> : (!atir.tensor<2x2xf32>, !atir.tensor<2x2xf32>,
        !atir.tensor<2x2xf32>) -> !atir.tensor<2x2xf32>
  %add = "atir.Add"(%add_out, %matmul, %bias) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<2x2xf32>, !atir.tensor<2x2xf32>,
        !atir.tensor<2xf32>) -> !atir.tensor<2x2xf32>
  %relu0 = "atir.Relu"(%relu0_out, %add) <{relu_limit = -1.0 : f32}> :
      (!atir.tensor<2x2xf32>, !atir.tensor<2x2xf32>)
      -> !atir.tensor<2x2xf32>
  %relu1 = "atir.Relu"(%relu1_out, %add) <{relu_limit = -1.0 : f32}> :
      (!atir.tensor<2x2xf32>, !atir.tensor<2x2xf32>)
      -> !atir.tensor<2x2xf32>
  return %relu0, %relu1 : !atir.tensor<2x2xf32>, !atir.tensor<2x2xf32>
}
