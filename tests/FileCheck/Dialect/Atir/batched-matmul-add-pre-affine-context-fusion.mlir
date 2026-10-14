// RUN: annc-opt --atir-op-fusion %s | FileCheck %s

// Pre-affine form: MatMul(X * scale + offset, W) + bias.  The fold must expose
// X as the dynamic kernel input; keeping the affine Add as the input is not
// equivalent to the ORT patch even if a generic BMM site is materialized.
// CHECK-LABEL: func.func @main
// CHECK-NOT: tf_name = "affined"
// CHECK: func.func private @fused_batched_matmul_add_
// CHECK-SAME: args = [{dtype = "f32", rank = 3 : i64, role = "dynamic", shape = [1, 2, 3], tf_name = "x"}
// CHECK: atir.Customize
// CHECK-SAME: custom.arg_names = ["data", "weight", "bias"]
// CHECK-SAME: custom.op_name = "BatchedMatMulAdd"
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<1x2x3xf32, name = "x">) -> !atir.tensor<1x2x4xf32, name = "y"> {
    %scale = atir.constant "public" @scale -> <3xf32, name = "scale", data = dense<[1.250000e+00, -5.000000e-01, 2.000000e+00]> : tensor<3xf32>>
    %offset = atir.constant "public" @offset -> <3xf32, name = "offset", data = dense<[1.000000e-01, -2.000000e-01, 3.000000e-01]> : tensor<3xf32>>
    %w = atir.constant "public" @w -> <1x3x4xf32, name = "w", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]]]> : tensor<1x3x4xf32>>
    %b = atir.constant "public" @b -> <1x1x4xf32, name = "b", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]]> : tensor<1x1x4xf32>>
    %scaled_buf = "atir.buffer"() : () -> !atir.tensor<1x2x3xf32, name = "scaled">
    %scaled = atir.Mul %scaled_buf, %x, %scale {metadata = {tf.input.0 = "x", tf.input.1 = "scale", tf.name = "n_mul", tf.op = "Mul"}} : (!atir.tensor<1x2x3xf32, name = "scaled">, !atir.tensor<1x2x3xf32, name = "x">, !atir.tensor<3xf32, name = "scale", data = dense<[1.250000e+00, -5.000000e-01, 2.000000e+00]> : tensor<3xf32>>) -> !atir.tensor<1x2x3xf32, name = "scaled">
    %affined_buf = "atir.buffer"() : () -> !atir.tensor<1x2x3xf32, name = "affined">
    %affined = "atir.Add"(%affined_buf, %scaled, %offset) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.input.0 = "scaled", tf.input.1 = "offset", tf.name = "n_affine", tf.op = "Add"}} : (!atir.tensor<1x2x3xf32, name = "affined">, !atir.tensor<1x2x3xf32, name = "scaled">, !atir.tensor<3xf32, name = "offset", data = dense<[1.000000e-01, -2.000000e-01, 3.000000e-01]> : tensor<3xf32>>) -> !atir.tensor<1x2x3xf32, name = "affined">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<1x2x4xf32, name = "mm">
    %mm = "atir.MatMul"(%mm_buf, %affined, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> {metadata = {tf.name = "n_mm", tf.op = "MatMul"}} : (!atir.tensor<1x2x4xf32, name = "mm">, !atir.tensor<1x2x3xf32, name = "affined">, !atir.tensor<1x3x4xf32, name = "w", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]]]> : tensor<1x3x4xf32>>) -> !atir.tensor<1x2x4xf32, name = "mm">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<1x2x4xf32, name = "y">
    %y = "atir.Add"(%y_buf, %mm, %b) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add", tf.op = "Add", tf.output_tensors = ["y"]}} : (!atir.tensor<1x2x4xf32, name = "y">, !atir.tensor<1x2x4xf32, name = "mm">, !atir.tensor<1x1x4xf32, name = "b", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]]> : tensor<1x1x4xf32>>) -> !atir.tensor<1x2x4xf32, name = "y">
    return %y : !atir.tensor<1x2x4xf32, name = "y">
  }
}
