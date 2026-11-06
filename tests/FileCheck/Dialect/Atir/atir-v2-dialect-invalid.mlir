// RUN: not annc-opt %s 2>&1 | FileCheck %s
// UnsupportedOp verify: `opType` must be non-empty.
// CHECK: requires a non-empty opType attribute
func.func @bad_unsupported(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "atir.unsupported"(%arg0) {opType = "", reason = "unknown_op"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
