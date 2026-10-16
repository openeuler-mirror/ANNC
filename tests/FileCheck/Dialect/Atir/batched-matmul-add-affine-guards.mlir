// RUN: annc-opt --atir-op-fusion --split-input-file %s | FileCheck %s

// Commuted Mul retains the dynamic operand and folds the vector constant.
// CHECK-LABEL: func.func @main
// CHECK: func.func private @fused_batched_matmul_add_
// CHECK-SAME: tf_name = "x"
// CHECK-SAME: member_node_names = ["n_add", "n_mm", "n_affine", "n_mul"]
// CHECK: atir.constant "private" @pre_affine_weight
// CHECK: atir.constant "private" @pre_affine_bias
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<1x2x3xf32, name = "x">) -> !atir.tensor<1x2x4xf32, name = "y"> {
    %scale = atir.constant "public" @scale -> <3xf32, name = "scale", data = dense<[1.250000e+00, -5.000000e-01, 2.000000e+00]> : tensor<3xf32>>
    %offset = atir.constant "public" @offset -> <3xf32, name = "offset", data = dense<[1.000000e-01, -2.000000e-01, 3.000000e-01]> : tensor<3xf32>>
    %w = atir.constant "public" @w -> <1x3x4xf32, name = "w", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]]]> : tensor<1x3x4xf32>>
    %b = atir.constant "public" @b -> <1x1x4xf32, name = "b", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]]> : tensor<1x1x4xf32>>
    %scaled_buf = "atir.buffer"() : () -> !atir.tensor<1x2x3xf32, name = "scaled">
    %scaled = atir.Mul %scaled_buf, %scale, %x {metadata = {tf.input.0 = "x", tf.input.1 = "scale", tf.name = "n_mul", tf.op = "Mul"}} : (!atir.tensor<1x2x3xf32, name = "scaled">, !atir.tensor<3xf32, name = "scale", data = dense<[1.250000e+00, -5.000000e-01, 2.000000e+00]> : tensor<3xf32>>, !atir.tensor<1x2x3xf32, name = "x">) -> !atir.tensor<1x2x3xf32, name = "scaled">
    %affined_buf = "atir.buffer"() : () -> !atir.tensor<1x2x3xf32, name = "affined">
    %affined = "atir.Add"(%affined_buf, %scaled, %offset) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.input.0 = "scaled", tf.input.1 = "offset", tf.name = "n_affine", tf.op = "Add"}} : (!atir.tensor<1x2x3xf32, name = "affined">, !atir.tensor<1x2x3xf32, name = "scaled">, !atir.tensor<3xf32, name = "offset", data = dense<[1.000000e-01, -2.000000e-01, 3.000000e-01]> : tensor<3xf32>>) -> !atir.tensor<1x2x3xf32, name = "affined">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<1x2x4xf32, name = "mm">
    %mm = "atir.MatMul"(%mm_buf, %affined, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> {metadata = {tf.name = "n_mm", tf.op = "MatMul"}} : (!atir.tensor<1x2x4xf32, name = "mm">, !atir.tensor<1x2x3xf32, name = "affined">, !atir.tensor<1x3x4xf32, name = "w", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]]]> : tensor<1x3x4xf32>>) -> !atir.tensor<1x2x4xf32, name = "mm">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<1x2x4xf32, name = "y">
    %y = "atir.Add"(%y_buf, %mm, %b) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add", tf.op = "Add", tf.output_tensors = ["y"]}} : (!atir.tensor<1x2x4xf32, name = "y">, !atir.tensor<1x2x4xf32, name = "mm">, !atir.tensor<1x1x4xf32, name = "b", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]]> : tensor<1x1x4xf32>>) -> !atir.tensor<1x2x4xf32, name = "y">
    return %y : !atir.tensor<1x2x4xf32, name = "y">
  }
}

// -----

// A dynamic offset must leave the full affine result as input: no partial match.
// CHECK-LABEL: func.func @main
// CHECK: func.func private @fused_batched_matmul_add_
// CHECK-SAME: tf_name = "affined"
// CHECK-SAME: member_node_names = ["n_add", "n_mm"]
// CHECK-NOT: @pre_affine_weight
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<1x2x3xf32, name = "x">, %offset: !atir.tensor<3xf32, name = "offset">) -> !atir.tensor<1x2x4xf32, name = "y"> {
    %scale = atir.constant "public" @scale -> <3xf32, name = "scale", data = dense<[1.250000e+00, -5.000000e-01, 2.000000e+00]> : tensor<3xf32>>
    %w = atir.constant "public" @w -> <1x3x4xf32, name = "w", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]]]> : tensor<1x3x4xf32>>
    %b = atir.constant "public" @b -> <1x1x4xf32, name = "b", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]]> : tensor<1x1x4xf32>>
    %scaled_buf = "atir.buffer"() : () -> !atir.tensor<1x2x3xf32, name = "scaled">
    %scaled = atir.Mul %scaled_buf, %x, %scale {metadata = {tf.input.0 = "x", tf.input.1 = "scale", tf.name = "n_mul", tf.op = "Mul"}} : (!atir.tensor<1x2x3xf32, name = "scaled">, !atir.tensor<1x2x3xf32, name = "x">, !atir.tensor<3xf32, name = "scale", data = dense<[1.250000e+00, -5.000000e-01, 2.000000e+00]> : tensor<3xf32>>) -> !atir.tensor<1x2x3xf32, name = "scaled">
    %affined_buf = "atir.buffer"() : () -> !atir.tensor<1x2x3xf32, name = "affined">
    %affined = "atir.Add"(%affined_buf, %scaled, %offset) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.input.0 = "scaled", tf.input.1 = "offset", tf.name = "n_affine", tf.op = "Add"}} : (!atir.tensor<1x2x3xf32, name = "affined">, !atir.tensor<1x2x3xf32, name = "scaled">, !atir.tensor<3xf32, name = "offset">) -> !atir.tensor<1x2x3xf32, name = "affined">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<1x2x4xf32, name = "mm">
    %mm = "atir.MatMul"(%mm_buf, %affined, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> {metadata = {tf.name = "n_mm", tf.op = "MatMul"}} : (!atir.tensor<1x2x4xf32, name = "mm">, !atir.tensor<1x2x3xf32, name = "affined">, !atir.tensor<1x3x4xf32, name = "w", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]]]> : tensor<1x3x4xf32>>) -> !atir.tensor<1x2x4xf32, name = "mm">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<1x2x4xf32, name = "y">
    %y = "atir.Add"(%y_buf, %mm, %b) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add", tf.op = "Add", tf.output_tensors = ["y"]}} : (!atir.tensor<1x2x4xf32, name = "y">, !atir.tensor<1x2x4xf32, name = "mm">, !atir.tensor<1x1x4xf32, name = "b", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]]> : tensor<1x1x4xf32>>) -> !atir.tensor<1x2x4xf32, name = "y">
    return %y : !atir.tensor<1x2x4xf32, name = "y">
  }
}

// -----

// An externally visible affine result cannot be swallowed by folding.
// CHECK-LABEL: func.func @main
// CHECK: func.func private @fused_batched_matmul_add_
// CHECK-SAME: tf_name = "affined"
// CHECK-SAME: member_node_names = ["n_add", "n_mm"]
// CHECK-NOT: @pre_affine_weight
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<1x2x3xf32, name = "x">) -> (!atir.tensor<1x2x4xf32, name = "y">, !atir.tensor<1x2x3xf32, name = "affined">) {
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
    return %y, %affined : !atir.tensor<1x2x4xf32, name = "y">, !atir.tensor<1x2x3xf32, name = "affined">
  }
}

// -----

// Equal element count is insufficient: per-row scales are not per-K affine.
// CHECK-LABEL: func.func @main
// CHECK: func.func private @fused_batched_matmul_add_
// CHECK-SAME: tf_name = "affined"
// CHECK-SAME: member_node_names = ["n_add", "n_mm"]
// CHECK-NOT: @pre_affine_weight
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<1x3x3xf32, name = "x">) -> !atir.tensor<1x3x4xf32, name = "y"> {
    %scale = atir.constant "public" @scale -> <3x1xf32, name = "scale", data = dense<1.250000e+00> : tensor<3x1xf32>>
    %offset = atir.constant "public" @offset -> <3xf32, name = "offset", data = dense<[1.000000e-01, -2.000000e-01, 3.000000e-01]> : tensor<3xf32>>
    %w = atir.constant "public" @w -> <1x3x4xf32, name = "w", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]]]> : tensor<1x3x4xf32>>
    %b = atir.constant "public" @b -> <1x1x4xf32, name = "b", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]]> : tensor<1x1x4xf32>>
    %scaled_buf = "atir.buffer"() : () -> !atir.tensor<1x3x3xf32, name = "scaled">
    %scaled = atir.Mul %scaled_buf, %x, %scale {metadata = {tf.input.0 = "x", tf.input.1 = "scale", tf.name = "n_mul", tf.op = "Mul"}} : (!atir.tensor<1x3x3xf32, name = "scaled">, !atir.tensor<1x3x3xf32, name = "x">, !atir.tensor<3x1xf32, name = "scale", data = dense<1.250000e+00> : tensor<3x1xf32>>) -> !atir.tensor<1x3x3xf32, name = "scaled">
    %affined_buf = "atir.buffer"() : () -> !atir.tensor<1x3x3xf32, name = "affined">
    %affined = "atir.Add"(%affined_buf, %scaled, %offset) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.input.0 = "scaled", tf.input.1 = "offset", tf.name = "n_affine", tf.op = "Add"}} : (!atir.tensor<1x3x3xf32, name = "affined">, !atir.tensor<1x3x3xf32, name = "scaled">, !atir.tensor<3xf32, name = "offset", data = dense<[1.000000e-01, -2.000000e-01, 3.000000e-01]> : tensor<3xf32>>) -> !atir.tensor<1x3x3xf32, name = "affined">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<1x3x4xf32, name = "mm">
    %mm = "atir.MatMul"(%mm_buf, %affined, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> {metadata = {tf.name = "n_mm", tf.op = "MatMul"}} : (!atir.tensor<1x3x4xf32, name = "mm">, !atir.tensor<1x3x3xf32, name = "affined">, !atir.tensor<1x3x4xf32, name = "w", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]]]> : tensor<1x3x4xf32>>) -> !atir.tensor<1x3x4xf32, name = "mm">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<1x3x4xf32, name = "y">
    %y = "atir.Add"(%y_buf, %mm, %b) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add", tf.op = "Add", tf.output_tensors = ["y"]}} : (!atir.tensor<1x3x4xf32, name = "y">, !atir.tensor<1x3x4xf32, name = "mm">, !atir.tensor<1x1x4xf32, name = "b", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]]> : tensor<1x1x4xf32>>) -> !atir.tensor<1x3x4xf32, name = "y">
    return %y : !atir.tensor<1x3x4xf32, name = "y">
  }
}

// -----

// Missing ONNX contract must not insert constants before returning failure.
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_batched_matmul_add
// CHECK-NOT: @pre_affine_weight
module attributes {module.state = "atir"} {
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

// -----

// Rank-4 site: fused as BatchedMatMulAdd4 with rank-4 descriptors (the rank-3
// kernel cannot express the extra batch dim). This case has no pre-affine
// chain; rank-4 affine folding is covered by the BMM context end-to-end suite.
// CHECK-LABEL: func.func @main
// CHECK: func.func private @fused_batched_matmul_add_
// CHECK: atir.Customize
// CHECK-SAME: custom.arg_ranks = [4, 4, 4]
// CHECK-SAME: custom.op_name = "BatchedMatMulAdd4"
// CHECK-SAME: custom.result_ranks = [4]
// CHECK-NOT: @pre_affine_weight
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<1x1x2x3xf32, name = "x">) -> !atir.tensor<1x1x2x4xf32, name = "y"> {
    %w = atir.constant "public" @"w" -> <1x1x3x4xf32, name = "w", data = dense<1.000000> : tensor<1x1x3x4xf32>>
    %b = atir.constant "public" @"b" -> <1x1x1x4xf32, name = "b", data = dense<0.000000> : tensor<1x1x1x4xf32>>
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<1x1x2x4xf32, name = "mm">
    %mm = "atir.MatMul"(%mm_buf, %x, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> {metadata = {tf.name = "n_mm", tf.op = "MatMul"}} : (!atir.tensor<1x1x2x4xf32, name = "mm">, !atir.tensor<1x1x2x3xf32, name = "x">, !atir.tensor<1x1x3x4xf32, name = "w", data = dense<1.000000> : tensor<1x1x3x4xf32>>) -> !atir.tensor<1x1x2x4xf32, name = "mm">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<1x1x2x4xf32, name = "y">
    %y = "atir.Add"(%y_buf, %mm, %b) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add", tf.op = "Add", tf.output_tensors = ["y"]}} : (!atir.tensor<1x1x2x4xf32, name = "y">, !atir.tensor<1x1x2x4xf32, name = "mm">, !atir.tensor<1x1x1x4xf32, name = "b", data = dense<0.000000> : tensor<1x1x1x4xf32>>) -> !atir.tensor<1x1x2x4xf32, name = "y">
    return %y : !atir.tensor<1x1x2x4xf32, name = "y">
  }
}


// -----

// A broadcast weight with per-output-batch bias still admits a base BMM site,
// even though the reference pre-affine fold requires equal weight/bias batches.
// CHECK-LABEL: func.func @main
// CHECK: func.func private @fused_batched_matmul_add_
// CHECK-SAME: tf_name = "affined"
// CHECK-SAME: member_node_names = ["n_add", "n_mm"]
// CHECK-NOT: @pre_affine_weight
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<2x2x3xf32, name = "x">) -> !atir.tensor<2x2x4xf32, name = "y"> {
    %scale = atir.constant "public" @scale -> <3xf32, name = "scale", data = dense<[1.250000e+00, -5.000000e-01, 2.000000e+00]> : tensor<3xf32>>
    %offset = atir.constant "public" @offset -> <3xf32, name = "offset", data = dense<[1.000000e-01, -2.000000e-01, 3.000000e-01]> : tensor<3xf32>>
    %w = atir.constant "public" @w -> <1x3x4xf32, name = "w", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]]]> : tensor<1x3x4xf32>>
    %b = atir.constant "public" @b -> <2x1x4xf32, name = "b", data = dense<0.000000e+00> : tensor<2x1x4xf32>>
    %scaled_buf = "atir.buffer"() : () -> !atir.tensor<2x2x3xf32, name = "scaled">
    %scaled = atir.Mul %scaled_buf, %x, %scale {metadata = {tf.input.0 = "x", tf.input.1 = "scale", tf.name = "n_mul", tf.op = "Mul"}} : (!atir.tensor<2x2x3xf32, name = "scaled">, !atir.tensor<2x2x3xf32, name = "x">, !atir.tensor<3xf32, name = "scale", data = dense<[1.250000e+00, -5.000000e-01, 2.000000e+00]> : tensor<3xf32>>) -> !atir.tensor<2x2x3xf32, name = "scaled">
    %affined_buf = "atir.buffer"() : () -> !atir.tensor<2x2x3xf32, name = "affined">
    %affined = "atir.Add"(%affined_buf, %scaled, %offset) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.input.0 = "scaled", tf.input.1 = "offset", tf.name = "n_affine", tf.op = "Add"}} : (!atir.tensor<2x2x3xf32, name = "affined">, !atir.tensor<2x2x3xf32, name = "scaled">, !atir.tensor<3xf32, name = "offset", data = dense<[1.000000e-01, -2.000000e-01, 3.000000e-01]> : tensor<3xf32>>) -> !atir.tensor<2x2x3xf32, name = "affined">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<2x2x4xf32, name = "mm">
    %mm = "atir.MatMul"(%mm_buf, %affined, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> {metadata = {tf.name = "n_mm", tf.op = "MatMul"}} : (!atir.tensor<2x2x4xf32, name = "mm">, !atir.tensor<2x2x3xf32, name = "affined">, !atir.tensor<1x3x4xf32, name = "w", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]]]> : tensor<1x3x4xf32>>) -> !atir.tensor<2x2x4xf32, name = "mm">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<2x2x4xf32, name = "y">
    %y = "atir.Add"(%y_buf, %mm, %b) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add", tf.op = "Add", tf.output_tensors = ["y"]}} : (!atir.tensor<2x2x4xf32, name = "y">, !atir.tensor<2x2x4xf32, name = "mm">, !atir.tensor<2x1x4xf32, name = "b", data = dense<0.000000e+00> : tensor<2x1x4xf32>>) -> !atir.tensor<2x2x4xf32, name = "y">
    return %y : !atir.tensor<2x2x4xf32, name = "y">
  }
}
