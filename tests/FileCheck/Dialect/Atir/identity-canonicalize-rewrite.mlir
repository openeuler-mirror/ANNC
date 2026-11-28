// RUN: annc-opt %s --atir-identity-canonicalize -o %t.once
// RUN: FileCheck %s --input-file=%t.once
// RUN: annc-opt %t.once --atir-identity-canonicalize -o %t.twice
// RUN: diff %t.once %t.twice

// Case 1: Eliminate Identity with multiple users.
// Expect: redirect both users, rename the producer and its buffer, and remove
// the unused Identity buffer without adding output metadata.
// CHECK-LABEL: func.func @eliminate_identity(
// CHECK-NEXT: %[[BUFFER:.*]] = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "output">
// CHECK-NEXT: %[[VALUE:.*]] = atir.Gather %[[BUFFER]], %{{[^,]+}}, %{{[^,]+}}, %{{[^ ]+}} : <1xf32, name = "output">
// CHECK-SAME: -> <1xf32, name = "output">
// CHECK-NEXT: %[[SHAPE:.*]] = atir.Shape %[[VALUE]] : (!atir.tensor<1xf32, name = "output">) -> !atir.tensor<1xi32>
// CHECK-NEXT: %[[RANK:.*]] = atir.Rank %[[VALUE]] : (!atir.tensor<1xf32, name = "output">) -> !atir.tensor<i32>
// CHECK-NEXT: return %[[SHAPE]], %[[RANK]] : !atir.tensor<1xi32>, !atir.tensor<i32>
func.func @eliminate_identity(
    %input: !atir.tensor<4xf32>,
    %indices: !atir.tensor<1xi32>,
    %axis: !atir.tensor<i32>)
    -> (!atir.tensor<1xi32>, !atir.tensor<i32>) {
  %gather_buffer = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "source">
  %gather = atir.Gather %gather_buffer, %input, %indices, %axis : <1xf32, name = "source">, <4xf32>, <1xi32>, <i32> -> <1xf32, name = "source">
  %identity_buffer = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "output">
  %identity = atir.Identity %identity_buffer, %gather : <1xf32, name = "output">, <1xf32, name = "source"> -> <1xf32, name = "output">
  %shape = atir.Shape %identity : (!atir.tensor<1xf32, name = "output">) -> !atir.tensor<1xi32>
  %rank = atir.Rank %identity : (!atir.tensor<1xf32, name = "output">) -> !atir.tensor<i32>
  return %shape, %rank : !atir.tensor<1xi32>, !atir.tensor<i32>
}

// Case 2: Preserve Identity when its input is a function argument.
// Expect: keep Identity and leave the function signature unchanged.
// CHECK-LABEL: func.func @preserve_argument(
// CHECK-SAME: %[[ARG:[^:]+]]: !atir.tensor<2xf32, name = "input">
// CHECK-SAME: -> !atir.tensor<2xf32, name = "output">
// CHECK-NEXT: %[[BUFFER:.*]] = "atir.buffer"() : () -> !atir.tensor<2xf32, name = "output">
// CHECK-NEXT: %[[RESULT:.*]] = atir.Identity %[[BUFFER]], %[[ARG]]
// CHECK-SAME: : <2xf32, name = "output">, <2xf32, name = "input"> -> <2xf32, name = "output">
// CHECK-NEXT: return %[[RESULT]] : !atir.tensor<2xf32, name = "output">
func.func @preserve_argument(
    %input: !atir.tensor<2xf32, name = "input">)
    -> !atir.tensor<2xf32, name = "output"> {
  %buffer = "atir.buffer"() : () -> !atir.tensor<2xf32, name = "output">
  %result = atir.Identity %buffer, %input : <2xf32, name = "output">, <2xf32, name = "input"> -> <2xf32, name = "output">
  return %result : !atir.tensor<2xf32, name = "output">
}

// Case 3: Preserve Identity when its input and result layouts differ.
// Expect: keep both buffers and leave endpoint metadata on Identity.
// CHECK-LABEL: func.func @preserve_layout_mismatch(
// CHECK-NEXT: %[[SOURCE_BUFFER:.*]] = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "gather", layout = "source_layout">
// CHECK-NEXT: %[[SOURCE:.*]] = atir.Gather %[[SOURCE_BUFFER]], {{.*}} {metadata = {tag = "keep"}}
// CHECK-SAME: : <1xf32, name = "gather", layout = "source_layout">, <4xf32>, <1xi32>, <i32> -> <1xf32, name = "gather", layout = "source_layout">
// CHECK-NEXT: %[[OUTPUT_BUFFER:.*]] = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "gather", layout = "output_layout">
// CHECK-NEXT: %[[RESULT:.*]] = atir.Identity %[[OUTPUT_BUFFER]], %[[SOURCE]] {metadata = {tf.output_tensor = "output:0"}}
// CHECK-SAME: : <1xf32, name = "gather", layout = "output_layout">, <1xf32, name = "gather", layout = "source_layout"> -> <1xf32, name = "gather", layout = "output_layout">
// CHECK-NEXT: return %[[RESULT]] : !atir.tensor<1xf32, name = "gather", layout = "output_layout">
func.func @preserve_layout_mismatch(
    %input: !atir.tensor<4xf32>,
    %indices: !atir.tensor<1xi32>,
    %axis: !atir.tensor<i32>)
    -> !atir.tensor<1xf32, name = "gather", layout = "output_layout"> {
  %gather_buffer = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "gather", layout = "source_layout">
  %gather = atir.Gather %gather_buffer, %input, %indices, %axis {metadata = {tag = "keep"}} : <1xf32, name = "gather", layout = "source_layout">, <4xf32>, <1xi32>, <i32> -> <1xf32, name = "gather", layout = "source_layout">
  %identity_buffer = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "gather", layout = "output_layout">
  %identity = atir.Identity %identity_buffer, %gather {metadata = {tf.output_tensor = "output:0"}} : <1xf32, name = "gather", layout = "output_layout">, <1xf32, name = "gather", layout = "source_layout"> -> <1xf32, name = "gather", layout = "output_layout">
  return %identity : !atir.tensor<1xf32, name = "gather", layout = "output_layout">
}

// Case 4: Eliminate a chain of Identity operations.
// Expect: preserve the final result type and remove both unused Identity buffers.
// CHECK-LABEL: func.func @eliminate_identity_chain(
// CHECK-SAME: -> !atir.tensor<1xf32, name = "output">
// CHECK-NEXT: %[[BUFFER:.*]] = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "output">
// CHECK-NEXT: %[[VALUE:.*]] = atir.Gather %[[BUFFER]],
// CHECK-SAME: : <1xf32, name = "output">, <4xf32>, <1xi32>, <i32> -> <1xf32, name = "output">
// CHECK-NEXT: return %[[VALUE]] : !atir.tensor<1xf32, name = "output">
func.func @eliminate_identity_chain(
    %input: !atir.tensor<4xf32>,
    %indices: !atir.tensor<1xi32>,
    %axis: !atir.tensor<i32>)
    -> !atir.tensor<1xf32, name = "output"> {
  %gather_buffer = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "source">
  %gather = atir.Gather %gather_buffer, %input, %indices, %axis : <1xf32, name = "source">, <4xf32>, <1xi32>, <i32> -> <1xf32, name = "source">
  %middle_buffer = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "middle">
  %middle = atir.Identity %middle_buffer, %gather : <1xf32, name = "middle">, <1xf32, name = "source"> -> <1xf32, name = "middle">
  %output_buffer = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "output">
  %output = atir.Identity %output_buffer, %middle : <1xf32, name = "output">, <1xf32, name = "middle"> -> <1xf32, name = "output">
  return %output : !atir.tensor<1xf32, name = "output">
}

// Case 5: Eliminate Identity after a producer without an output buffer.
// Expect: rename the producer's result, keeping its input type unchanged.
// CHECK-LABEL: func.func @preserve_shape_input(
// CHECK-SAME: %[[INPUT:[^:]+]]: !atir.tensor<2x3xf32, name = "input">
// CHECK-SAME: -> !atir.tensor<2xi32, name = "output">
// CHECK-NEXT: %[[SHAPE:.*]] = atir.Shape %[[INPUT]] : (!atir.tensor<2x3xf32, name = "input">) -> !atir.tensor<2xi32, name = "output">
// CHECK-NEXT: return %[[SHAPE]] : !atir.tensor<2xi32, name = "output">
func.func @preserve_shape_input(
    %input: !atir.tensor<2x3xf32, name = "input">)
    -> !atir.tensor<2xi32, name = "output"> {
  %shape = atir.Shape %input : (!atir.tensor<2x3xf32, name = "input">) -> !atir.tensor<2xi32, name = "shape">
  %buffer = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "output">
  %result = atir.Identity %buffer, %shape : <2xi32, name = "output">, <2xi32, name = "shape"> -> <2xi32, name = "output">
  return %result : !atir.tensor<2xi32, name = "output">
}

// Case 6: Eliminate Identity whose output buffer has another user.
// Expect: keep the shared buffer and preserve its remaining use.
// CHECK-LABEL: func.func @preserve_shared_buffer(
// CHECK-NEXT: %[[SOURCE_BUFFER:.*]] = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "output">
// CHECK-NEXT: %[[VALUE:.*]] = atir.Gather %[[SOURCE_BUFFER]],
// CHECK-SAME: -> <1xf32, name = "output">
// CHECK-NEXT: %[[SHARED_BUFFER:.*]] = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "output">
// CHECK-NEXT: %[[SHAPE:.*]] = atir.Shape %[[SHARED_BUFFER]] : (!atir.tensor<1xf32, name = "output">) -> !atir.tensor<1xi32>
// CHECK-NEXT: return %[[VALUE]], %[[SHAPE]] : !atir.tensor<1xf32, name = "output">, !atir.tensor<1xi32>
func.func @preserve_shared_buffer(
    %input: !atir.tensor<4xf32>,
    %indices: !atir.tensor<1xi32>,
    %axis: !atir.tensor<i32>)
    -> (!atir.tensor<1xf32, name = "output">, !atir.tensor<1xi32>) {
  %gather_buffer = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "source">
  %gather = atir.Gather %gather_buffer, %input, %indices, %axis : <1xf32, name = "source">, <4xf32>, <1xi32>, <i32> -> <1xf32, name = "source">
  %shared_buffer = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "output">
  %identity = atir.Identity %shared_buffer, %gather : <1xf32, name = "output">, <1xf32, name = "source"> -> <1xf32, name = "output">
  %shape = atir.Shape %shared_buffer : (!atir.tensor<1xf32, name = "output">) -> !atir.tensor<1xi32>
  return %identity, %shape : !atir.tensor<1xf32, name = "output">, !atir.tensor<1xi32>
}
