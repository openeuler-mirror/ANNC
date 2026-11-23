// RUN: annc-opt %s -atir-op-fusion-v2 | FileCheck %s
//
// op-fusion-v2 pattern: kp_embedding_padding, C form -- the only form the
// real frontend emits on cvr_3-like graphs (1030 batch-6, 12 sites). The
// origin vector is the sparse_reshape_v2 outputShape result itself; its
// dim-0 row count is a plain convert/slice/reshape chain (no carrier, no
// StridedSlice residue); fill is the legalized dynamic_broadcast_in_dim.
// Without C-form acceptance every real padding site silently drops.

module {
  func.func @main(%data: tensor<?x4xf32>) -> tensor<?x4xf32>
      attributes {tf.arg_names = ["data"], tf.entry_function = {control_outputs = "", inputs = "data:0", outputs = "out:0"}} {
    %idx = stablehlo.constant dense<[[0, 0], [1, 3]]> : tensor<2x2xi64> loc("Const:_idx")
    %shp = stablehlo.constant dense<[8, 4]> : tensor<2xi64> loc("Const:_shp")
    %dfl = stablehlo.constant dense<[1, 4]> : tensor<2xi64> loc("Const:_dfl")
    %outI:2 = "atir.sparse_reshape_v2"(%idx, %shp, %dfl) {annc.source_node = "serve_prep/gen/SparseReshape", annc.source_op = "tf.SparseReshape", annc.source_ports = dense<[0, 1]> : tensor<2xi32>, call_target_name = "tf.SparseReshape"} : (tensor<2x2xi64>, tensor<2xi64>, tensor<2xi64>) -> (tensor<2x2xi64>, tensor<2xi64>) loc(fused["SparseReshape:", "serve_prep/gen/SparseReshape"])
    %originCast = stablehlo.convert %outI#1 {annc.source_node = "model/click/Shape/Cast", annc.source_op = "tf.Cast"} : (tensor<2xi64>) -> tensor<2xi32> loc("pad_c")
    %originSlice = stablehlo.slice %originCast [0:1] {annc.begin_source = "model/click/strided_slice/stack", annc.source_node = "model/click/strided_slice", annc.source_op = "tf.StridedSlice"} : (tensor<2xi32>) -> tensor<1xi32> loc("pad_c")
    %originRows = stablehlo.reshape %originSlice {annc.source_node = "model/click/strided_slice", annc.source_op = "tf.StridedSlice"} : (tensor<1xi32>) -> tensor<i32> loc("pad_c")
    %dataShp = shape.shape_of %data : tensor<?x4xf32> -> tensor<2xindex> loc("pad_c")
    %dataCast = arith.index_cast %dataShp : tensor<2xindex> to tensor<2xi32> loc("pad_c")
    %dataSlice = stablehlo.slice %dataCast [0:1] : (tensor<2xi32>) -> tensor<1xi32> loc("pad_c")
    %dataRows = stablehlo.reshape %dataSlice {annc.source_node = "model/click/strided_slice_1", annc.source_op = "tf.StridedSlice"} : (tensor<1xi32>) -> tensor<i32> loc("pad_c")
    %sub = stablehlo.subtract %originRows, %dataRows {annc.prov = ["model/click/strided_slice:0", "model/click/strided_slice_1:0"], annc.source_node = "model/click/sub", annc.source_op = "tf.Sub"} : tensor<i32> loc(fused["Sub:", "model/click/sub"])
    %rowsPack = stablehlo.reshape %sub {annc.source_node = "model/click/zeros/packed", annc.source_op = "tf.Pack"} : (tensor<i32>) -> tensor<1xi32> loc("pad_c")
    %packc = stablehlo.constant dense<4> : tensor<1xi32> loc("Const:_pack")
    %dims = stablehlo.concatenate %rowsPack, %packc, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc("pad_c")
    %c0f = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc("Const:_fill")
    %fill = stablehlo.dynamic_broadcast_in_dim %c0f, %dims, dims = [] {annc.source_node = "model/click/zeros", annc.source_op = "tf.Fill"} : (tensor<f32>, tensor<2xi32>) -> tensor<?x4xf32> loc("pad_c")
    %concat = stablehlo.concatenate %data, %fill, dim = 0 : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32> loc(fused["ConcatV2:", "model/click/concat"])
    %cShape = stablehlo.constant dense<[-1, 4]> : tensor<2xi32> loc("Const:_shape")
    %out = atir.unsupported {detail = "", metadata = {annc.source_node = "model/click/Reshape_1", annc.source_op = "tf.Reshape", call_target_name = "tf.Reshape", residue_reason = "pattern_requires_static_shape"}, opType = "Reshape", reason = "unknown_op"} (%concat, %cShape) : (tensor<?x4xf32>, tensor<2xi32>) -> tensor<?x4xf32> loc(fused["Reshape:", "model/click/Reshape_1"])
    return %out : tensor<?x4xf32> loc("output")
  }
}

// CHECK: call @KPFusedEmbeddingPadding_{{[0-9a-zA-Z_]+}}(
// CHECK-SAME: ) -> (tensor<i32>, tensor<?x4xf32>)
// CHECK: func.func private @KPFusedEmbeddingPadding_{{[0-9a-zA-Z_]+}}
// CHECK-SAME: annc.kernel
// CHECK-SAME: abi = "annc_context_v3"
// C-form origin slot: the sparse_reshape_v2 outputShape vector itself
// (V1 wires the same tensor as the origin_shape edge).
// CHECK-SAME: args = [{dtype = "i64", rank = 1 : i64, role = "dynamic", shape = [2], tf_name = "serve_prep/gen/SparseReshape:1"}
// CHECK-SAME: {dtype = "i32", rank = 0 : i64, role = "dynamic", shape = [], tf_name = "model/click/strided_slice_1:0"}
// CHECK-SAME: {dtype = "i32", rank = 0 : i64, role = "dynamic", shape = [], tf_name = "model/click/Reshape_1/pack"}
// CHECK-SAME: {dtype = "f32", rank = 2 : i64, role = "dynamic", shape = [-9223372036854775808, 4], tf_name = "data"}
// CHECK-SAME: {dtype = "i32", rank = 1 : i64, role = "constant", shape = [2], tf_name = "model/click/Reshape_1/shape"}]
// CHECK-SAME: fusion.pattern = "kp_embedding_padding"
// CHECK-SAME: outputs = [{dtype = "i32", rank = 0 : i64, role = "output", shape = [], tf_name = "model/click/sub:0"}
// CHECK-SAME: {dtype = "f32", rank = 2 : i64, role = "output", shape = [-9223372036854775808, 4], tf_name = "model/click/concat:0"}]
