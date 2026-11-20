// RUN: annc-asm %s -atir-op-fusion-v2 | FileCheck %s
//
// kp_sparse_segment_reduce over the mixed-carrier route: the dynamic
// StridedSlice is FE-expanded into stablehlo arithmetic; the slice-scalar
// anchor folds two allowlist closures into the 5-arg schema ABI
// (KPFusedSparseSegmentReduceI64Mean, dual outputs).
//
// Level 2 (tail CODEGEN assertions below) turns the kernel body into an
// atir.Customize via --annc-fast-codegen. Closure constants clone into
// the kernel, not the contract.
module {
  func.func @main(%keys: tensor<4x2xi64>, %begin: tensor<2xi32>,
                  %begin_1: tensor<1xi32>, %data: tensor<6x3xf32>,
                  %indices: tensor<4xi64>) -> (tensor<?x3xf32>, tensor<i32>)
      attributes {tf.arg_names = ["keys", "begin", "begin_1", "data", "indices"], tf.entry_function = {control_outputs = "", inputs = "keys:0,begin:0,begin_1:0,data:0,indices:0", outputs = "mean_out:0,slice_out:0"}} {
    %c = stablehlo.constant dense<2> : tensor<i32> loc("Const:")
    %c_0 = stablehlo.constant dense<0> : tensor<i32> loc("Const:_1")
    %0 = stablehlo.slice %begin [1:2] : (tensor<2xi32>) -> tensor<1xi32> loc("StridedSlice:")
    %1 = stablehlo.reshape %0 : (tensor<1xi32>) -> tensor<i32> loc("StridedSlice:")
    %2 = stablehlo.compare LT, %1, %c_0 : (tensor<i32>, tensor<i32>) -> tensor<i1> loc("StridedSlice:")
    %3 = stablehlo.add %1, %c : tensor<i32> loc("StridedSlice:")
    %4 = stablehlo.select %2, %3, %1 : tensor<i1>, tensor<i32> loc("StridedSlice:")
    %5 = stablehlo.dynamic_slice %keys, %c_0, %4, sizes = [4, 1] : (tensor<4x2xi64>, tensor<i32>, tensor<i32>) -> tensor<4x1xi64> loc(fused["StridedSlice:", "seg_ids"])
    %6 = stablehlo.reshape %5 : (tensor<4x1xi64>) -> tensor<4xi64> loc(fused["StridedSlice:", "seg_ids"])
    %cst = arith.constant dense<0> : tensor<i64> loc(fused["SparseSegmentMean:", "mean_out"])
    %7 = "atir.sparse_segment_mean_v2"(%data, %indices, %6, %cst) {call_target_name = "tf.SparseSegmentMean", sparse_gradient = false} : (tensor<6x3xf32>, tensor<4xi64>, tensor<4xi64>, tensor<i64>) -> tensor<?x3xf32> loc(fused["SparseSegmentMean:", "mean_out"])
    %8 = shape.shape_of %7 : tensor<?x3xf32> -> tensor<2xindex> loc(fused["Shape:", "shape_out"])
    %9 = arith.index_cast %8 : tensor<2xindex> to tensor<2xi32> loc(fused["Shape:", "shape_out"])
    %10 = stablehlo.reshape %begin_1 : (tensor<1xi32>) -> tensor<i32> loc(fused["StridedSlice:", "slice_out"])
    %11 = stablehlo.compare LT, %10, %c_0 : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(fused["StridedSlice:", "slice_out"])
    %12 = stablehlo.add %10, %c : tensor<i32> loc(fused["StridedSlice:", "slice_out"])
    %13 = stablehlo.select %11, %12, %10 : tensor<i1>, tensor<i32> loc(fused["StridedSlice:", "slice_out"])
    %14 = stablehlo.dynamic_slice %9, %13, sizes = [1] : (tensor<2xi32>, tensor<i32>) -> tensor<1xi32> loc(fused["StridedSlice:", "slice_out"])
    %15 = stablehlo.reshape %14 : (tensor<1xi32>) -> tensor<i32> loc(fused["StridedSlice:", "slice_out"])
    return %7, %15 : tensor<?x3xf32>, tensor<i32> loc("output")
  }
}

// Closure constants clone into the kernel, not the contract; the call
// sits in main before the kernel func, so CHECKs follow that order.
// CHECK: call @KPFusedSparseSegmentReduceI64Mean
// CHECK: func.func private @KPFusedSparseSegmentReduceI64Mean
// CHECK-SAME: annc.kernel
// CHECK-SAME: abi = "annc_context_v3"
// CHECK-SAME: args = [{dtype = "i64", rank = 2 : i64, role = "dynamic", shape = [4, 2], tf_name = "keys"}, {dtype = "i32", rank = 1 : i64, role = "dynamic", shape = [2], tf_name = "begin"}, {dtype = "f32", rank = 2 : i64, role = "dynamic", shape = [6, 3], tf_name = "data"}, {dtype = "i64", rank = 1 : i64, role = "dynamic", shape = [4], tf_name = "indices"}, {dtype = "i32", rank = 1 : i64, role = "dynamic", shape = [1], tf_name = "begin_1"}]
// CHECK-SAME: fallback_function = "original_subgraph"
// CHECK-SAME: fusion.pattern = "kp_sparse_segment_reduce"
// CHECK-SAME: kernel_name = "KPFusedSparseSegmentReduceI64Mean"
// CHECK-SAME: outputs = [{dtype = "f32", rank = 2 : i64, role = "output", shape = [-9223372036854775808, 3], tf_name = "mean_out"}, {dtype = "i32", rank = 0 : i64, role = "output", shape = [], tf_name = "slice_out"}]

// RUN: annc-asm %s -atir-op-fusion-v2 --annc-fast-codegen | FileCheck %s --check-prefix=CODEGEN
// CODEGEN: atir.Customize
// CODEGEN-SAME: custom.op_name = "KPFusedSparseSegmentReduceI64Mean"
// CODEGEN: return
