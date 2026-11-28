// RUN: annc-opt %s --atir-identity-canonicalize -o %t
// RUN: FileCheck %s --input-file=%t

// Case 1: Transfer Identity's endpoint to a producer without an endpoint array.
// Expect: create the output endpoint array and preserve unrelated metadata.
// CHECK-LABEL: func.func @transfer_output_endpoint(
// CHECK-NEXT: %[[BUFFER:.*]] = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "output">
// CHECK-NEXT: %[[VALUE:.*]] = atir.Gather %[[BUFFER]], {{.*}} {metadata = {tag = "keep", tf.output_tensors = ["output:0"]}}
// CHECK-SAME: : <1xf32, name = "output">, <4xf32>, <1xi32>, <i32> -> <1xf32, name = "output">
// CHECK-NEXT: return %[[VALUE]] : !atir.tensor<1xf32, name = "output">
func.func @transfer_output_endpoint(
    %input: !atir.tensor<4xf32>,
    %indices: !atir.tensor<1xi32>,
    %axis: !atir.tensor<i32>)
    -> !atir.tensor<1xf32, name = "output"> {
  %gather_buffer = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "source">
  %gather = atir.Gather %gather_buffer, %input, %indices, %axis {metadata = {tag = "keep"}} : <1xf32, name = "source">, <4xf32>, <1xi32>, <i32> -> <1xf32, name = "source">
  %identity_buffer = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "output">
  %identity = atir.Identity %identity_buffer, %gather {metadata = {tf.output_tensor = "output:0"}} : <1xf32, name = "output">, <1xf32, name = "source"> -> <1xf32, name = "output">
  return %identity : !atir.tensor<1xf32, name = "output">
}

// Case 2: Transfer Identity's endpoint for the second result of a producer.
// Expect: update only that result's type and endpoint, preserving the first
// result's type and endpoint and unrelated metadata.
// CHECK-LABEL: func.func @update_second_endpoint(
// CHECK-SAME: -> (!atir.tensor<?xi64, name = "values">, !atir.tensor<4xi32, name = "output">)
// CHECK-NEXT: %[[VALUES:[^,]+]], %[[INDICES:[^ ]+]] = atir.Unique %{{[^ ]+}} {metadata = {tag = "keep", tf.output_tensors = ["values:0", "new_indices:0"]}}
// CHECK-SAME: : (!atir.tensor<4xi64>) -> (!atir.tensor<?xi64, name = "values">, !atir.tensor<4xi32, name = "output">)
// CHECK-NEXT: return %[[VALUES]], %[[INDICES]] : !atir.tensor<?xi64, name = "values">, !atir.tensor<4xi32, name = "output">
func.func @update_second_endpoint(
    %input: !atir.tensor<4xi64>)
    -> (!atir.tensor<?xi64, name = "values">, !atir.tensor<4xi32, name = "output">) {
  %values, %indices = atir.Unique %input {metadata = {tag = "keep", tf.output_tensors = ["values:0", "old_indices:0"]}} : (!atir.tensor<4xi64>) -> (!atir.tensor<?xi64, name = "values">, !atir.tensor<4xi32, name = "indices">)
  %buffer = "atir.buffer"() : () -> !atir.tensor<4xi32, name = "output">
  %output = atir.Identity %buffer, %indices {metadata = {tf.output_tensor = "new_indices:0"}} : <4xi32, name = "output">, <4xi32, name = "indices"> -> <4xi32, name = "output">
  return %values, %output : !atir.tensor<?xi64, name = "values">, !atir.tensor<4xi32, name = "output">
}
