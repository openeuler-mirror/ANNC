// RUN: annc-fusion-metadata %s -o %t && FileCheck %s --input-file=%t

// CHECK: "abi": "annc_execution_v2"
// CHECK: "outputs": [
// CHECK: "dtype": "i64"
// CHECK: "tf_name": "unique:0"
// CHECK: "dtype": "i32"
// CHECK: "tf_name": "unique:1"
// CHECK: "dtype": "f32"
// CHECK: "tf_name": "gather:0"

module {
  func.func private @fused_kp_gather(
      %execution: !llvm.ptr, %data: memref<2x4xf32>, %keys: memref<4x3xi64>,
      %begin: memref<2xi32>) -> i32 attributes {
    annc.kernel,
    fusion.metadata = {
      abi = "annc_execution_v2",
      args = [
        {dtype = "f32", rank = 2 : i64, role = "dynamic",
         shape = [2 : i64, 4 : i64], tf_name = "data:0"},
        {dtype = "i64", rank = 2 : i64, role = "dynamic",
         shape = [4 : i64, 3 : i64], tf_name = "keys:0"},
        {dtype = "i32", rank = 1 : i64, role = "dynamic",
         shape = [2 : i64], tf_name = "begin:0"}
      ],
      dynamic_dims = [],
      fallback_function = "original_subgraph",
      fusion.pattern = "kp_fused_gather",
      kernel_arg_order = [],
      kernel_name = "fused_kp_gather",
      outputs = [
        {dtype = "i64", rank = 1 : i64, role = "output",
         shape = [-1 : i64], tf_name = "unique:0"},
        {dtype = "i32", rank = 1 : i64, role = "output",
         shape = [-1 : i64], tf_name = "unique:1"},
        {dtype = "f32", rank = 2 : i64, role = "output",
         shape = [-1 : i64, 4 : i64], tf_name = "gather:0"}
      ],
      symbolic_signature = "",
      tf.name = "kp_fused_gather"
    }
  }
}
