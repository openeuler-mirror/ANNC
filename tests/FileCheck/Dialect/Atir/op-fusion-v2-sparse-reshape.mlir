// RUN: annc-asm %s -atir-op-fusion-v2 | FileCheck %s
// RUN: annc-asm %s -atir-op-fusion-v2 --annc-fast-codegen | FileCheck %s --check-prefix=CODEGEN
//
module {
  func.func @main(%keys: tensor<?x2xi64>, %begin: tensor<2xi32>,
                  %new_shape: tensor<2xi64>) -> (tensor<2x2xi64>, tensor<2xi64>)
      attributes {tf.arg_names = ["keys", "begin", "new_shape"], tf.entry_function = {control_outputs = "", inputs = "keys:0,begin:0,new_shape:0", outputs = "sr_out:0,sr_out:1"}} {
    %c = stablehlo.constant dense<[[0], [1]]> : tensor<2x1xi64> loc("Const:")
    %c_pack = stablehlo.constant {annc.const_source = "Const_7"} dense<1> : tensor<i64> loc("Const_7")
    %c_0 = stablehlo.constant dense<1> : tensor<2xi32> loc("Const:_1")
    %c_1 = stablehlo.constant dense<[0, 2]> : tensor<2xi32> loc("Const:_2")
    %0 = shape.shape_of %keys : tensor<?x2xi64> -> tensor<2xindex> loc(fused["Shape:", "shape_k"])
    %1 = arith.index_cast %0 : tensor<2xindex> to tensor<2xi32> loc(fused["Shape:", "shape_k"])
    %2 = stablehlo.slice %1 [0:1] : (tensor<2xi32>) -> tensor<1xi32> loc(fused["StridedSlice:", "num_rows"])
    %3 = stablehlo.reshape %c_pack : (tensor<i64>) -> tensor<1xi64> loc(fused["Pack:", "pack"])
    %cvt = stablehlo.convert %3 : (tensor<1xi64>) -> tensor<1xi32>
    %4 = stablehlo.concatenate %2, %cvt, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(fused["Pack:", "pack"])
    %5 = stablehlo.convert %4 : (tensor<2xi32>) -> tensor<2xi64> loc(fused["Cast:", "in_shape"])
    %6 = atir.unsupported {metadata = {begin_mask = 1 : i64, ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 2 : i64}, opType = "StridedSlice"} (%keys, %begin, %c_1, %c_0) : (tensor<?x2xi64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi64> loc(fused["StridedSlice:", "seg_ids"])
    %7 = stablehlo.reshape %6 : (tensor<2xi64>) -> tensor<2x1xi64> loc(fused["Reshape:", "seg_r"])
    %8 = stablehlo.concatenate %c, %7, dim = 1 : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<2x2xi64> loc(fused["ConcatV2:", "indices"])
    %outputIndices, %outputShape = "atir.sparse_reshape_v2"(%8, %5, %new_shape) {T = i64, call_target_name = "tf.SparseReshape"} : (tensor<2x2xi64>, tensor<2xi64>, tensor<2xi64>) -> (tensor<2x2xi64>, tensor<2xi64>) loc(fused["SparseReshape:", "sr_out"])
    return %outputIndices, %outputShape : tensor<2x2xi64>, tensor<2xi64> loc("output")
  }
}

// CHECK: call @KPFusedSparseReshapeI64
// CHECK: func.func private @KPFusedSparseReshapeI64
// CHECK-SAME: annc.kernel
// CHECK-SAME: abi = "annc_context_v3"
// CHECK-SAME: args = [{dtype = "i64", rank = 2 : i64, role = "dynamic", shape = [-9223372036854775808, 2], tf_name = "keys"}, {dtype = "i32", rank = 1 : i64, role = "dynamic", shape = [2], tf_name = "begin"}, {dtype = "i64", rank = 0 : i64, role = "constant", shape = [], tf_name = "Const_7"}, {dtype = "i64", rank = 1 : i64, role = "dynamic", shape = [2], tf_name = "new_shape"}]
// CHECK-SAME: fallback_function = "original_subgraph"
// CHECK-SAME: fusion.pattern = "kp_sparse_reshape"
// CHECK-SAME: kernel_name = "KPFusedSparseReshapeI64"
// CHECK-SAME: outputs = [{dtype = "i64", rank = 2 : i64, role = "output", shape = [2, 2], tf_name = "sr_out:0"}, {dtype = "i64", rank = 1 : i64, role = "output", shape = [2], tf_name = "sr_out:1"}]

// CODEGEN: atir.Customize
// CODEGEN-SAME: custom.op_name = "KPFusedSparseReshapeI64"
// CODEGEN: return
