// RUN: annc-opt %s --atir-identity-canonicalize --atir-op-fusion | FileCheck %s

module @main attributes {module.state = "atir"} {
  func.func @main(
      %lhs: !atir.tensor<2x2xf32, name = "lhs">,
      %rhs: !atir.tensor<2x2xf32, name = "rhs">)
      -> !atir.tensor<2x2xf32, name = "__annc_output_0"> {
    %matmul_buffer = "atir.buffer"() : () ->
        !atir.tensor<2x2xf32, name = "matmul">
    %matmul = "atir.MatMul"(%matmul_buffer, %lhs, %rhs)
        <{do_relu = false, left_transpose = false, output_transpose = false,
          relu_limit = -1.000000e+00 : f32, right_transpose = false,
          withBias = false}>
        : (!atir.tensor<2x2xf32, name = "matmul">,
           !atir.tensor<2x2xf32, name = "lhs">,
           !atir.tensor<2x2xf32, name = "rhs">)
        -> !atir.tensor<2x2xf32, name = "matmul">
    %output_buffer = "atir.buffer"() : () ->
        !atir.tensor<2x2xf32, name = "__annc_output_0">
    %output = atir.Identity %output_buffer, %matmul
        : <2x2xf32, name = "__annc_output_0">,
          <2x2xf32, name = "matmul">
        -> <2x2xf32, name = "__annc_output_0">
    return %output : !atir.tensor<2x2xf32, name = "__annc_output_0">
  }
}

// CHECK-LABEL: func.func @main
// CHECK: %[[OUTPUT:.*]] = "atir.buffer"() : () -> !atir.tensor<2x2xf32, name = "__annc_output_0">
// CHECK: call @fused_matmul_{{.*}}({{.*}}, %[[OUTPUT]])
// CHECK: return %[[OUTPUT]] : !atir.tensor<2x2xf32, name = "__annc_output_0">
