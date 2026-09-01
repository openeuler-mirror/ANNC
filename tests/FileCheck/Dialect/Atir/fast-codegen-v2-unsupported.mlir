// RUN: annc-asm %s -atir-fast-codegen | FileCheck %s

// CHECK-LABEL: func.func private @unsupported_f64
// CHECK: "atir.MatMul"
// CHECK-NOT: atir.Customize
module attributes {module.state = "atir"} {
  func.func private @unsupported_f64(
      %execution: !llvm.ptr,
      %lhs: !atir.tensor<2x2xf64>,
      %rhs: !atir.tensor<2x2xf64>,
      %bias: !atir.tensor<2xf64>)
      -> (!atir.tensor<2x2xf64>, !atir.tensor<2x2xf64>)
      attributes {
        annc.kernel,
        fusion.metadata = {abi = "annc_execution_v2"}
      } {
    %matmul_out = "atir.buffer"() : () -> !atir.tensor<2x2xf64>
    %add_out = "atir.buffer"() : () -> !atir.tensor<2x2xf64>
    %relu_out = "atir.buffer"() : () -> !atir.tensor<2x2xf64>
    %matmul = "atir.MatMul"(%matmul_out, %lhs, %rhs) <{
      do_relu = false, left_transpose = false, output_transpose = false,
      relu_limit = -1.0 : f32, right_transpose = false, withBias = false
    }> : (!atir.tensor<2x2xf64>, !atir.tensor<2x2xf64>,
          !atir.tensor<2x2xf64>) -> !atir.tensor<2x2xf64>
    %add = "atir.Add"(%add_out, %matmul, %bias) <{
      do_relu = false, relu_limit = -1.0 : f32
    }> : (!atir.tensor<2x2xf64>, !atir.tensor<2x2xf64>,
          !atir.tensor<2xf64>) -> !atir.tensor<2x2xf64>
    %relu = "atir.Relu"(%relu_out, %add) <{relu_limit = -1.0 : f32}> :
        (!atir.tensor<2x2xf64>, !atir.tensor<2x2xf64>)
        -> !atir.tensor<2x2xf64>
    return %add, %relu : !atir.tensor<2x2xf64>, !atir.tensor<2x2xf64>
  }
}

// -----

// CHECK-LABEL: func.func @materialized_fallback
// CHECK: "atir.MatMul"
// CHECK-SAME: annc.fusion_materialized
// CHECK-NOT: atir.Customize
module attributes {module.state = "atir"} {
  func.func @materialized_fallback(
      %output: !atir.tensor<2x2xf32>,
      %lhs: !atir.tensor<2x2xf32>,
      %rhs: !atir.tensor<2x2xf32>) -> !atir.tensor<2x2xf32> {
    %result = "atir.MatMul"(%output, %lhs, %rhs) {
      annc.fusion_materialized,
      do_relu = false,
      left_transpose = false,
      output_transpose = false,
      relu_limit = -1.0 : f32,
      right_transpose = false,
      withBias = false
    } : (!atir.tensor<2x2xf32>, !atir.tensor<2x2xf32>,
          !atir.tensor<2x2xf32>) -> !atir.tensor<2x2xf32>
    return %result : !atir.tensor<2x2xf32>
  }
}

// -----

// The v2 builtin only supports plain GEMM, unbounded Relu, and an
// unmodified Add. Unsupported attributes must preserve the original graph.
// CHECK-LABEL: func.func @unsupported_attributes
// CHECK: "atir.MatMul"
// CHECK: "atir.Add"
// CHECK: "atir.Relu"
// CHECK-NOT: atir.Customize
module attributes {module.state = "atir"} {
  func.func @unsupported_attributes(
      %output: !atir.tensor<2x2xf32>, %lhs: !atir.tensor<2x2xf32>,
      %rhs: !atir.tensor<2x2xf32>, %bias: !atir.tensor<2xf32>)
      -> (!atir.tensor<2x2xf32>, !atir.tensor<2x2xf32>) {
    %matmul = "atir.MatMul"(%output, %lhs, %rhs) {
      do_relu = false, left_transpose = false, output_transpose = false,
      relu_limit = -1.0 : f32, right_transpose = true, withBias = false
    } : (!atir.tensor<2x2xf32>, !atir.tensor<2x2xf32>,
          !atir.tensor<2x2xf32>) -> !atir.tensor<2x2xf32>
    %add_output = "atir.buffer"() : () -> !atir.tensor<2x2xf32>
    %add = "atir.Add"(%add_output, %matmul, %bias) {
      do_relu = true, relu_limit = -1.0 : f32
    } : (!atir.tensor<2x2xf32>, !atir.tensor<2x2xf32>,
          !atir.tensor<2xf32>) -> !atir.tensor<2x2xf32>
    %relu_output = "atir.buffer"() : () -> !atir.tensor<2x2xf32>
    %relu = "atir.Relu"(%relu_output, %add) {
      relu_limit = 6.0 : f32
    } : (!atir.tensor<2x2xf32>, !atir.tensor<2x2xf32>)
        -> !atir.tensor<2x2xf32>
    return %add, %relu : !atir.tensor<2x2xf32>, !atir.tensor<2x2xf32>
  }
}
