// RUN: rm -rf %t.dir && mkdir -p %t.dir && split-file %s %t.dir
// RUN: annc-asm %t.dir/positive.mlir -atir-op-fusion-v2 | FileCheck %t.dir/positive.mlir
// RUN: annc-asm %t.dir/positive.mlir -atir-op-fusion-v2 --annc-fast-codegen | FileCheck %t.dir/positive.mlir --check-prefix=CODEGEN
// RUN: annc-asm %t.dir/negative.mlir -atir-op-fusion-v2 | FileCheck %t.dir/negative.mlir
//
// kp_sparse_segment_reduce Form B (the real KEEP model): keys come from
// the SimpleHash64ToSparse custom op as an atir.unsupported carrier, and
// the tail chain folds into a static shape_of->index_cast->slice->reshape.
//
// Expected: carrier boundary + static tail + constant promotion hit
// KPFusedSparseSegmentReduceI32Mean with the V1-style 5-arg ABI;
// negative: shrink=1 (not a column pick) stays unmatched.

//--- positive.mlir
module {
  func.func @main(%data: tensor<?x5xf32>, %indices: tensor<?xi32>) -> (tensor<?x5xf32>, tensor<i32>) attributes {tf.arg_names = ["data", "indices"], tf.entry_function = {control_outputs = "", inputs = "data:0,indices:0", outputs = "mean_out:0,slice_out:0"}} {
    %in = stablehlo.constant dense<0> : tensor<4xi32>
    %keys:2 = atir.unsupported {detail = "", metadata = {call_target_name = "tf.SimpleHash64ToSparse"}, opType = "SimpleHash64ToSparse", reason = "unknown_op"} (%in) : (tensor<4xi32>) -> (tensor<?x?xi64>, tensor<?xi64>) loc("serve_prep/plain_features/SimpleHash64ToSparse")
    %begin = stablehlo.constant dense<[0, 1]> : tensor<2xi32> loc("StridedSlice:")
    %end = stablehlo.constant dense<0> : tensor<2xi32> loc("StridedSlice:")
    %strides = stablehlo.constant dense<[1, 1]> : tensor<2xi32> loc("StridedSlice:")
    %sl = atir.unsupported {detail = "", metadata = {begin_mask = 1 : i64, annc.begin_source = "model/strided_slice/begin", call_target_name = "tf.StridedSlice", device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, residue_reason = "no_legalize_pattern", shrink_axis_mask = 2 : i64}, opType = "StridedSlice", reason = "unknown_op"} (%keys#0, %begin, %end, %strides) : (tensor<?x?xi64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?xi64> loc("StridedSlice:")
    %seg = stablehlo.convert %sl : (tensor<?xi64>) -> tensor<?xi32> loc("Cast:")
    %cst = arith.constant dense<0> : tensor<i32> loc("SparseSegmentMean:")
    %mean = "atir.sparse_segment_mean_v2"(%data, %indices, %seg, %cst) {call_target_name = "tf.SparseSegmentMean", sparse_gradient = false} : (tensor<?x5xf32>, tensor<?xi32>, tensor<?xi32>, tensor<i32>) -> tensor<?x5xf32> loc("SparseSegmentMean:")
    %sh = shape.shape_of %mean : tensor<?x5xf32> -> tensor<2xindex> loc("Shape_1")
    %ic = arith.index_cast %sh : tensor<2xindex> to tensor<2xi32> loc("Shape_1")
    %s = stablehlo.slice %ic [0:1] : (tensor<2xi32>) -> tensor<1xi32> loc("StridedSlice_1:")
    %out1 = stablehlo.reshape %s : (tensor<1xi32>) -> tensor<i32> loc("StridedSlice_1:")
    return %mean, %out1 : tensor<?x5xf32>, tensor<i32>
  }
// Bare residue form (cvr_4 measured): no Cast node, the StridedSlice
// residue feeds ss directly without a convert wrapper.
  func.func @bare(%data: tensor<?x4xf32>, %indices: tensor<?xi32>) -> (tensor<?x4xf32>, tensor<i32>) attributes {tf.arg_names = ["data", "indices"]} {
    %in2 = stablehlo.constant dense<0> : tensor<4xi32>
    %keys2:2 = atir.unsupported {detail = "", metadata = {call_target_name = "tf.SimpleHash64ToSparse"}, opType = "SimpleHash64ToSparse", reason = "unknown_op"} (%in2) : (tensor<4xi32>) -> (tensor<?x?xi64>, tensor<?xi64>) loc("serve_prep/plain_features/SimpleHash64ToSparse")
    %begin2 = stablehlo.constant dense<[0, 1]> : tensor<2xi32> loc("strided_slice_1/begin")
    %end2 = stablehlo.constant dense<0> : tensor<2xi32> loc("strided_slice_1/begin")
    %strides2 = stablehlo.constant dense<[1, 1]> : tensor<2xi32> loc("strided_slice_1/begin")
    %sl2 = atir.unsupported {detail = "", metadata = {begin_mask = 1 : i64, annc.begin_source = "model/strided_slice/begin", call_target_name = "tf.StridedSlice", device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, residue_reason = "no_legalize_pattern", shrink_axis_mask = 2 : i64}, opType = "StridedSlice", reason = "unknown_op"} (%keys2#0, %begin2, %end2, %strides2) : (tensor<?x?xi64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?xi64> loc("strided_slice_1")
    %cst2 = arith.constant dense<0> : tensor<i64> loc("SparseSegmentSum:")
    %sum = "atir.sparse_segment_sum_v2"(%data, %indices, %sl2, %cst2) {call_target_name = "tf.SparseSegmentSum", sparse_gradient = false} : (tensor<?x4xf32>, tensor<?xi32>, tensor<?xi64>, tensor<i64>) -> tensor<?x4xf32> loc("SparseSegmentSum:")
    %sh2 = shape.shape_of %sum : tensor<?x4xf32> -> tensor<2xindex> loc("Shape_1")
    %ic2 = arith.index_cast %sh2 : tensor<2xindex> to tensor<2xi32> loc("Shape_1")
    %s2 = stablehlo.slice %ic2 [0:1] : (tensor<2xi32>) -> tensor<1xi32> loc("strided_slice_2:")
    %out2 = stablehlo.reshape %s2 : (tensor<1xi32>) -> tensor<i32> loc("strided_slice_2:")
    return %sum, %out2 : tensor<?x4xf32>, tensor<i32>
  }
}

// Level 1: 5-arg contract = keys (carrier i64 2D), begin (promoted
// constant), data, indices, begin_1 (synthesized); dual outputs.
// CODEGEN: atir.Customize
// CODEGEN-SAME: custom.op_name = "KPFusedSparseSegmentReduceI32Mean"
// CHECK: call @KPFusedSparseSegmentReduceI32Mean
// The bare form hits too (I32Sum; the site counter carries across
// functions, hence the _1 suffix).
// CHECK: call @KPFusedSparseSegmentReduceI32Sum
// CHECK: func.func private @KPFusedSparseSegmentReduceI32Mean
// CHECK-SAME: annc.kernel
// CHECK-SAME: abi = "annc_context_v3"
// CHECK-SAME: args = [{dtype = "i64", rank = 2 : i64, role = "dynamic", shape = [-9223372036854775808, -9223372036854775808], tf_name = "serve_prep/plain_features/SimpleHash64ToSparse"}, {dtype = "i32", rank = 1 : i64, role = "constant", shape = [2], tf_name = "StridedSlice_1_begin"}, {dtype = "f32", rank = 2 : i64, role = "dynamic", shape = [-9223372036854775808, 5], tf_name = "data"}, {dtype = "i32", rank = 1 : i64, role = "dynamic", shape = [-9223372036854775808], tf_name = "indices"}, {dtype = "i32", rank = 1 : i64, role = "dynamic", shape = [1], tf_name = "StridedSlice_1_stack"}]
// CHECK-SAME: fallback_function = "original_subgraph"
// CHECK-SAME: fusion.pattern = "kp_sparse_segment_reduce"
// CHECK-SAME: kernel_name = "KPFusedSparseSegmentReduceI32Mean"
// CHECK-SAME: outputs = [{dtype = "f32", rank = 2 : i64, role = "output", shape = [-9223372036854775808, 5], tf_name = "SparseSegmentMean"}, {dtype = "i32", rank = 0 : i64, role = "output", shape = [], tf_name = "StridedSlice_1"}]

//--- negative.mlir
module {
  func.func @main(%data: tensor<?x5xf32>, %indices: tensor<?xi32>) -> (tensor<?x5xf32>, tensor<i32>) attributes {tf.arg_names = ["data", "indices"], tf.entry_function = {control_outputs = "", inputs = "data:0,indices:0", outputs = "mean_out:0,slice_out:0"}} {
    %in = stablehlo.constant dense<0> : tensor<4xi32>
    %keys:2 = atir.unsupported {detail = "", metadata = {call_target_name = "tf.SimpleHash64ToSparse"}, opType = "SimpleHash64ToSparse", reason = "unknown_op"} (%in) : (tensor<4xi32>) -> (tensor<?x?xi64>, tensor<?xi64>) loc("serve_prep/plain_features/SimpleHash64ToSparse")
    %begin = stablehlo.constant dense<[0]> : tensor<1xi32> loc("StridedSlice:")
    %end = stablehlo.constant dense<0> : tensor<1xi32> loc("StridedSlice:")
    %strides = stablehlo.constant dense<[1]> : tensor<1xi32> loc("StridedSlice:")
    %sl = atir.unsupported {detail = "", metadata = {begin_mask = 0 : i64, annc.begin_source = "model/strided_slice/begin", call_target_name = "tf.StridedSlice", device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, residue_reason = "no_legalize_pattern", shrink_axis_mask = 1 : i64}, opType = "StridedSlice", reason = "unknown_op"} (%keys#0, %begin, %end, %strides) : (tensor<?x?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64> loc("StridedSlice:")
    %seg = stablehlo.convert %sl : (tensor<?xi64>) -> tensor<?xi32> loc("Cast:")
    %cst = arith.constant dense<0> : tensor<i32> loc("SparseSegmentMean:")
    %mean = "atir.sparse_segment_mean_v2"(%data, %indices, %seg, %cst) {call_target_name = "tf.SparseSegmentMean", sparse_gradient = false} : (tensor<?x5xf32>, tensor<?xi32>, tensor<?xi32>, tensor<i32>) -> tensor<?x5xf32> loc("SparseSegmentMean:")
    %sh = shape.shape_of %mean : tensor<?x5xf32> -> tensor<2xindex> loc("Shape_1")
    %ic = arith.index_cast %sh : tensor<2xindex> to tensor<2xi32> loc("Shape_1")
    %s = stablehlo.slice %ic [0:1] : (tensor<2xi32>) -> tensor<1xi32> loc("StridedSlice_1:")
    %out1 = stablehlo.reshape %s : (tensor<1xi32>) -> tensor<i32> loc("StridedSlice_1:")
    return %mean, %out1 : tensor<?x5xf32>, tensor<i32>
  }
}

// shrink=1 is not a column pick; the keys chain mismatches.
// CHECK-NOT: KPFusedSparseSegmentReduce
