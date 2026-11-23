// RUN: rm -rf %t.dir && mkdir -p %t.dir && split-file %s %t.dir
// RUN: annc-asm %t.dir/positive.mlir -atir-op-fusion-v2 | FileCheck %t.dir/positive.mlir
// RUN: annc-asm %t.dir/positive.mlir -atir-op-fusion-v2 --annc-fast-codegen | FileCheck %t.dir/positive.mlir --check-prefix=CODEGEN
// RUN: annc-asm %t.dir/negative.mlir -atir-op-fusion-v2 | FileCheck %t.dir/negative.mlir
//
//

//--- positive.mlir
module {
  func.func @main(%keys: tensor<?x2xi64>) -> (tensor<?x2xi64>, tensor<2xi64>) attributes {tf.arg_names = ["keys"], tf.entry_function = {control_outputs = "", inputs = "keys:0", outputs = "sr_out:0,sr_shape:0"}} {
    %c49 = stablehlo.constant dense<[-1, 1]> : tensor<2xi32> loc("Reshape:")
    %c31 = stablehlo.constant dense<[0, 0]> : tensor<2xi32> loc("StridedSlice:")
    %c22 = stablehlo.constant dense<[1, 1]> : tensor<2xi32> loc("StridedSlice:")
    %c30 = stablehlo.constant dense<[1, 1]> : tensor<2xi32> loc("StridedSlice:")
    %seg = atir.unsupported {detail = "", metadata = {begin_mask = 1 : i64, call_target_name = "tf.StridedSlice", device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, residue_reason = "no_legalize_pattern", shrink_axis_mask = 2 : i64}, opType = "StridedSlice", reason = "unknown_op"} (%keys, %c31, %c22, %c30) : (tensor<?x2xi64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?xi64> loc("StridedSlice:")
    %segCol = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", device = "", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%seg, %c49) : (tensor<?xi64>, tensor<2xi32>) -> tensor<?x1xi64> loc("Reshape_1:")
    %rowScalar = atir.unsupported {detail = "", metadata = {begin_mask = 0 : i64, call_target_name = "tf.StridedSlice", device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, residue_reason = "no_legalize_pattern", shrink_axis_mask = 1 : i64}, opType = "StridedSlice", reason = "unknown_op"} (%keys, %c31, %c22, %c30) : (tensor<?x2xi64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<i64> loc("StridedSlice_1:")
    %idx1 = stablehlo.reshape %rowScalar : (tensor<i64>) -> tensor<1xi64> loc("Shape_1")
    %iota = stablehlo.dynamic_iota %idx1, dim = 0 : (tensor<1xi64>) -> tensor<?xi32> loc("Range:")
    %rowCol32 = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", device = "", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%iota, %c49) : (tensor<?xi32>, tensor<2xi32>) -> tensor<?x1xi32> loc("Reshape_2:")
    %rowCol = stablehlo.convert %rowCol32 : (tensor<?x1xi32>) -> tensor<?x1xi64> loc("Cast:")
    %indices = stablehlo.concatenate %rowCol, %segCol, dim = 1 : (tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x2xi64> loc("concat:")
    %r1c = arith.trunci %rowScalar : tensor<i64> to tensor<i32> loc("Shape_2")
    %r1 = stablehlo.reshape %r1c : (tensor<i32>) -> tensor<1xi32> loc("Shape_2")
    %c2 = stablehlo.constant {annc.const_source = "Pack_c"} dense<2> : tensor<i64> loc("Pack_c")
    %p2 = stablehlo.reshape %c2 : (tensor<i64>) -> tensor<1xi64> loc("Pack_c")
    %cvt = stablehlo.convert %p2 : (tensor<1xi64>) -> tensor<1xi32> loc("Pack_c")
    %cat2 = stablehlo.concatenate %r1, %cvt, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc("concat_1:")
    %inShape = stablehlo.convert %cat2 : (tensor<2xi32>) -> tensor<2xi64> loc("Cast_1:")
    %cnew = stablehlo.constant {annc.const_source = "SparseReshape_new_shape"} dense<[-1, 1]> : tensor<2xi64> loc("SparseReshape:")
    %outI, %outS = "atir.sparse_reshape_v2"(%indices, %inShape, %cnew) {call_target_name = "tf.SparseReshape"} : (tensor<?x2xi64>, tensor<2xi64>, tensor<2xi64>) -> (tensor<?x2xi64>, tensor<2xi64>) loc("SparseReshape:")
    return %outI, %outS : tensor<?x2xi64>, tensor<2xi64>
  }
}

// CHECK: call @KPFusedSparseReshapeI64
// CHECK: func.func private @KPFusedSparseReshapeI64
// CHECK-SAME: annc.kernel
// CHECK-SAME: abi = "annc_context_v3"
// CHECK-SAME: fusion.pattern = "kp_sparse_reshape"
// CHECK-SAME: kernel_name = "KPFusedSparseReshapeI64"
// CODEGEN: atir.Customize
// CODEGEN-SAME: custom.op_name = "KPFusedSparseReshapeI64"
// CODEGEN: return

//--- negative.mlir
module {
  func.func @main(%keys: tensor<?xi64>) -> (tensor<?x2xi64>, tensor<2xi64>) attributes {tf.arg_names = ["keys"], tf.entry_function = {control_outputs = "", inputs = "keys:0", outputs = "sr_out:0,sr_shape:0"}} {
    %c49 = stablehlo.constant dense<[-1, 1]> : tensor<2xi32> loc("Reshape:")
    %c31 = stablehlo.constant dense<[0, 0]> : tensor<2xi32> loc("StridedSlice:")
    %c22 = stablehlo.constant dense<[1, 1]> : tensor<2xi32> loc("StridedSlice:")
    %c30 = stablehlo.constant dense<[1, 1]> : tensor<2xi32> loc("StridedSlice:")
    %seg = atir.unsupported {detail = "", metadata = {begin_mask = 1 : i64, call_target_name = "tf.StridedSlice", device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, residue_reason = "no_legalize_pattern", shrink_axis_mask = 2 : i64}, opType = "StridedSlice", reason = "unknown_op"} (%keys, %c31, %c22, %c30) : (tensor<?xi64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?xi64> loc("StridedSlice:")
    %segCol = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", device = "", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%seg, %c49) : (tensor<?xi64>, tensor<2xi32>) -> tensor<?x1xi64> loc("Reshape_1:")
    %rowCol = stablehlo.constant dense<[[0]]> : tensor<1x1xi64> loc("Cast:")
    %indices = stablehlo.concatenate %rowCol, %segCol, dim = 1 : (tensor<1x1xi64>, tensor<?x1xi64>) -> tensor<?x2xi64> loc("concat:")
    %inShape = stablehlo.constant dense<[4, 2]> : tensor<2xi64> loc("Cast_1:")
    %cnew = stablehlo.constant {annc.const_source = "SparseReshape_new_shape"} dense<[-1, 1]> : tensor<2xi64> loc("SparseReshape:")
    %outI, %outS = "atir.sparse_reshape_v2"(%indices, %inShape, %cnew) {call_target_name = "tf.SparseReshape"} : (tensor<?x2xi64>, tensor<2xi64>, tensor<2xi64>) -> (tensor<?x2xi64>, tensor<2xi64>) loc("SparseReshape:")
    return %outI, %outS : tensor<?x2xi64>, tensor<2xi64>
  }
}

// CHECK-NOT: KPFusedSparseReshape
