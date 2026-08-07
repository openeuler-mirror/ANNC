// RUN: annc-opt %s --atir-identity-canonicalize | FileCheck %s

// CHECK-LABEL: func.func @main
func.func @main(%input: !atir.tensor<4xf32, name = "input">,
                %indices: !atir.tensor<1xi32, name = "indices">,
                %axis: !atir.tensor<i32, name = "axis">)
    -> (!atir.tensor<1xi64, name = "shape1">,
        !atir.tensor<1xi64, name = "shape2">) {
  %gather_buffer = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "gather">
  %gather = atir.Gather %gather_buffer, %input, %indices, %axis
      : <1xf32, name = "gather">, <4xf32, name = "input">,
        <1xi32, name = "indices">, <i32, name = "axis">
        -> <1xf32, name = "gather">
  %identity_buffer = "atir.buffer"() : () -> !atir.tensor<1xf32, name = "identity">
  %identity = atir.Identity %identity_buffer, %gather
      : <1xf32, name = "identity">, <1xf32, name = "gather">
        -> <1xf32, name = "identity">
  %shape1 = atir.Shape %identity
      : (!atir.tensor<1xf32, name = "identity">)
        -> !atir.tensor<1xi64, name = "shape1">
  %shape2 = atir.Shape %identity
      : (!atir.tensor<1xf32, name = "identity">)
        -> !atir.tensor<1xi64, name = "shape2">
  // CHECK: atir.Gather {{.*}}name = "identity"
  // CHECK-NOT: atir.Identity
  // CHECK: atir.Shape {{.*}}name = "shape1"
  // CHECK: atir.Shape {{.*}}name = "shape2"
  return %shape1, %shape2 : !atir.tensor<1xi64, name = "shape1">,
      !atir.tensor<1xi64, name = "shape2">
}
