// RUN: annc-fusion-metadata %s -o %t && FileCheck %s --input-file=%t
// RUN: sed s/annc_context_v3/annc_execution_v2/g %s > %t.v2.mlir
// Legacy V2 metadata remains readable for explicit compatibility consumers.
// RUN: annc-fusion-metadata %t.v2.mlir -o %t.json
// RUN: sed s/annc_context_v3/annc_context_v2/g %s > %t.unsupported.mlir
// RUN: not annc-fusion-metadata %t.unsupported.mlir -o %t.json 2>&1 | FileCheck %s --check-prefix=REJECT

// RUN: not annc-asm %t.unsupported.mlir --lower-context-kernel-entries 2>&1 | FileCheck %s --check-prefix=REJECT

// REJECT: unsupported fusion ABI: annc_context_v2

// CHECK: "abi": "annc_context_v3"
// CHECK: "dynamic_dims": []
// CHECK: "outputs": [
// CHECK: "dtype": "i64"
// CHECK: "shape": [
// CHECK-NEXT: -1
// CHECK: "tf_name": "unique:0"
// CHECK: "dtype": "i32"
// CHECK: "tf_name": "unique:1"
// CHECK: "dtype": "f32"
// CHECK: "shape": [
// CHECK-NEXT: -1,
// CHECK-NEXT: 4
// CHECK: "tf_name": "gather:0"

module {
  func.func private @fused_kp_gather(
      %context: !llvm.ptr) -> i32 attributes {
    annc.kernel,
    fusion.metadata = {
      abi = "annc_context_v3",
      args = [
        {dtype = "f32", rank = 2 : i64, role = "dynamic",
         shape = [2 : i64, 4 : i64], tf_name = "data:0"},
        {dtype = "i64", rank = 2 : i64, role = "dynamic",
         shape = [4 : i64, 3 : i64], tf_name = "keys:0"},
        {dtype = "i32", rank = 1 : i64, role = "dynamic",
         shape = [2 : i64], tf_name = "begin:0"}
      ],
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
