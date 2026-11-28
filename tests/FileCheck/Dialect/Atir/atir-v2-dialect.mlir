// RUN: annc-opt --split-input-file %s | FileCheck %s
// V2 dialect roundtrips. Value domain: builtin tensors (`AnyTensor`);
// `!atir.tensor` is the V1 domain and is rejected by design.

// UnsupportedOp roundtrip (custom format: attr-dict (operands) : type).
// CHECK-LABEL: func @unsupported_roundtrip
// CHECK: atir.unsupported {{.*}}opType = "FooOp"{{.*}} (%{{.*}}) : (tensor<4xf32>) -> tensor<4xf32>
func.func @unsupported_roundtrip(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "atir.unsupported"(%arg0) {detail = "", metadata = {device = "/device:CPU:0"}, opType = "FooOp", reason = "unknown_op"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// Sparse capability op on builtin tensors.
// CHECK-LABEL: func @v2_sparse_to_dense
// CHECK: "atir.sparse_to_dense_v2"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{validateIndices = false}>
func.func @v2_sparse_to_dense(%i: tensor<2xi64>, %s: tensor<2xi64>, %v: tensor<2xf32>, %d: tensor<f32>) -> tensor<2x2xf32> {
  %0 = "atir.sparse_to_dense_v2"(%i, %s, %v, %d) <{validateIndices = false}> : (tensor<2xi64>, tensor<2xi64>, tensor<2xf32>, tensor<f32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// String tensor = builtin tensor with `!atir.string` elements (no encoding).
// CHECK-LABEL: func @v2_string_to_number
// CHECK: "atir.string_to_number_v2"(%{{.*}}) : (tensor<2x!atir.string>) -> tensor<2xf32>
func.func @v2_string_to_number(%s: tensor<2x!atir.string>) -> tensor<2xf32> {
  %0 = "atir.string_to_number_v2"(%s) : (tensor<2x!atir.string>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @v2_top_k
// CHECK: "atir.top_k_v2"(%{{.*}}, %{{.*}}) : (tensor<4xf32>, tensor<i32>) -> (tensor<4x2xf32>, tensor<4x2xi32>)
func.func @v2_top_k(%x: tensor<4xf32>, %k: tensor<i32>) -> (tensor<4x2xf32>, tensor<4x2xi32>) {
  %0:2 = "atir.top_k_v2"(%x, %k) : (tensor<4xf32>, tensor<i32>) -> (tensor<4x2xf32>, tensor<4x2xi32>)
  return %0#0, %0#1 : tensor<4x2xf32>, tensor<4x2xi32>
}

// -----

// CustomizeOp keeps its V1 `!atir.tensor` contract (unchanged by this PR).
// CHECK-LABEL: func @customize_v1_domain
// CHECK: atir.Customize
func.func @customize_v1_domain(%arg0: !atir.tensor<4xf32, name = "x">) -> !atir.tensor<4xf32, name = "y"> {
  %0 = "atir.Customize"(%arg0) {opType = "Foo"} : (!atir.tensor<4xf32, name = "x">) -> !atir.tensor<4xf32, name = "y">
  return %0 : !atir.tensor<4xf32, name = "y">
}
