// RUN: annc-opt %s -atir-op-fusion-v2 | FileCheck %s
//
// op-fusion-v2 kp_embedding_padding robustness: mid-chain scaffold shape
// changes must not affect matching (see EmbeddingPaddingV2.cpp).
//   * @glue_renamed: the annc.getDim / annc.setDim / StridedSlice opTypes
//     are all renamed - they are shape-carrying glue, renaming keeps the
//     semantics, the site still matches;
//   * @hop_removed: one setDim segment removed from the dims chain (sub
//     wired straight to getDim) - a hop count change still matches.
// The Fill / Reshape opTypes are the only exact discriminators (renaming
// them rejects, see the negative cases). The pass probes the whole module
// once @main exists, so only the first case needs that name.
// CHECK: call @KPFusedEmbeddingPadding_{{[0-9a-zA-Z_]+}}(
// CHECK-SAME: ) -> (tensor<i32>, tensor<?x4xf32>)
// CHECK: call @KPFusedEmbeddingPadding_{{[0-9a-zA-Z_]+}}(
// CHECK-SAME: ) -> (tensor<i32>, tensor<?x4xf32>)
// CHECK: func.func private @KPFusedEmbeddingPadding_{{[0-9a-zA-Z_]+}}(
// CHECK-SAME: kernel_name = "KPFusedEmbeddingPadding_{{[0-9a-zA-Z_]+}}"
// CHECK: func.func private @KPFusedEmbeddingPadding_{{[0-9a-zA-Z_]+}}(
// CHECK-SAME: kernel_name = "KPFusedEmbeddingPadding_{{[0-9a-zA-Z_]+}}"
module {
  func.func @main(%data: tensor<?x4xf32>, %hash: tensor<*x!atir.string>) -> tensor<?x240x4xf32>
      attributes {tf.arg_names = ["data", "hash"], tf.entry_function = {control_outputs = "", inputs = "data:0,hash:0", outputs = "out:0"}} {
    %s:3 = atir.unsupported {annc.source_node = "serve_prep/trival_features", detail = "", metadata = {call_target_name = "tf.UnifyPbHashSparse", residue_reason = "unregistered_custom_op"}, opType = "UnifyPbHashSparse", reason = "unknown_op"} (%hash) : (tensor<*x!atir.string>) -> (tensor<?x?xi64>, tensor<?xi64>, tensor<*xi64>) loc(fused["UnifyPbHashSparse:", "serve_prep/trival_features"])
    %originVec:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDimRenamed", reason = "unknown_op"} (%s#2) : (tensor<*xi64>) -> (tensor<?xi64>, tensor<?xi64>) loc("pad_s")
    %conv = stablehlo.convert %originVec#0 : (tensor<?xi64>) -> tensor<?xi32> loc("pad_s")
    %setd1 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.setDim"}, opType = "annc.setDimRenamed", reason = "unknown_op"} (%conv) : (tensor<?xi32>) -> tensor<*xi32> loc("pad_s")
    %c0 = stablehlo.constant dense<0> : tensor<1xi32> loc("Const:_0")
    %c1 = stablehlo.constant dense<1> : tensor<1xi32> loc("Const:_1")
    %ss = atir.unsupported {detail = "", metadata = {begin_mask = 0 : i64, call_target_name = "tf.StridedSlice", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, residue_reason = "unranked", shrink_axis_mask = 1 : i64}, opType = "StridedSliceRenamed", reason = "unknown_op"} (%setd1, %c0, %c1, %c1) : (tensor<*xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<*xi32> loc(fused["StridedSlice:", "origin0"])
    %getd2:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDimRenamed", reason = "unknown_op"} (%ss) : (tensor<*xi32>) -> (tensor<?xi32>, tensor<?xi64>) loc("pad_s")
    %shp = shape.shape_of %data : tensor<?x4xf32> -> tensor<2xindex> loc(fused["Shape:", "shape_d"])
    %cast = arith.index_cast %shp : tensor<2xindex> to tensor<2xi32> loc(fused["Shape:", "shape_d"])
    %sl = stablehlo.slice %cast [0:1] : (tensor<2xi32>) -> tensor<1xi32> loc(fused["StridedSlice:", "num_rows"])
    %rows = stablehlo.reshape %sl {annc.source_node = "num_rows"} : (tensor<1xi32>) -> tensor<i32> loc(fused["StridedSlice:", "num_rows"])
    %getd3:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDimRenamed", reason = "unknown_op"} (%rows) : (tensor<i32>) -> (tensor<1xi32>, tensor<?xi64>) loc("pad_s")
    %shp2 = shape.shape_of %getd2#0 : tensor<?xi32> -> tensor<1xindex> loc("pad_s")
    %cst_shape = shape.const_shape [1] : tensor<1xindex> loc("pad_s")
    %bshape = shape.broadcast %shp2, %cst_shape : tensor<1xindex>, tensor<1xindex> -> tensor<1xindex> loc("pad_s")
    %b1 = stablehlo.dynamic_broadcast_in_dim %getd2#0, %bshape, dims = [0] : (tensor<?xi32>, tensor<1xindex>) -> tensor<?xi32> loc("pad_s")
    %b2 = stablehlo.dynamic_broadcast_in_dim %getd3#0, %bshape, dims = [0] : (tensor<1xi32>, tensor<1xindex>) -> tensor<?xi32> loc("pad_s")
    %sub = stablehlo.subtract %b1, %b2 {annc.prov = ["serve_prep/trival_features:2", "num_rows:0"]} : tensor<?xi32> loc(fused["Sub:", "model/sub"])
    %setd2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.setDim"}, opType = "annc.setDimRenamed", reason = "unknown_op"} (%sub) : (tensor<?xi32>) -> tensor<*xi32> loc("pad_s")
    %getd4:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDimRenamed", reason = "unknown_op"} (%setd2) : (tensor<*xi32>) -> (tensor<?xi32>, tensor<?xi64>) loc("pad_s")
    %c4 = stablehlo.constant dense<4> : tensor<i32> loc("Const:_pack")
    %getd5:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDimRenamed", reason = "unknown_op"} (%c4) : (tensor<i32>) -> (tensor<1xi32>, tensor<?xi64>) loc("pad_s")
    %dims = stablehlo.concatenate %getd4#0, %getd5#0, dim = 0 : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32> loc("pad_s")
    %setd3 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.setDim"}, opType = "annc.setDimRenamed", reason = "unknown_op"} (%dims) : (tensor<?xi32>) -> tensor<*xi32> loc("pad_s")
    %c0f = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc("Const:_fill")
    %fill = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Fill", residue_reason = "unranked"}, opType = "Fill", reason = "unknown_op"} (%setd3, %c0f) : (tensor<*xi32>, tensor<f32>) -> tensor<?x4xf32> loc(fused["Fill:", "model/fill"])
    %concat = stablehlo.concatenate %data, %fill, dim = 0 : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32> loc(fused["ConcatV2:", "model/concat"])
    %c68 = stablehlo.constant dense<[-1, 240, 4]> : tensor<3xi32> loc("Const:_shape68")
    %out = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%concat, %c68) : (tensor<?x4xf32>, tensor<3xi32>) -> tensor<?x240x4xf32> loc(fused["Reshape:", "model/Reshape"])
    return %out : tensor<?x240x4xf32>
  }

  func.func @hop_removed(%data: tensor<?x4xf32>, %hash: tensor<*x!atir.string>) -> tensor<?x240x4xf32>
      attributes {tf.arg_names = ["data", "hash"], tf.entry_function = {control_outputs = "", inputs = "data:0,hash:0", outputs = "out:0"}} {
    %s:3 = atir.unsupported {annc.source_node = "serve_prep/trival_features", detail = "", metadata = {call_target_name = "tf.UnifyPbHashSparse", residue_reason = "unregistered_custom_op"}, opType = "UnifyPbHashSparse", reason = "unknown_op"} (%hash) : (tensor<*x!atir.string>) -> (tensor<?x?xi64>, tensor<?xi64>, tensor<*xi64>) loc(fused["UnifyPbHashSparse:", "serve_prep/trival_features"])
    %originVec:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%s#2) : (tensor<*xi64>) -> (tensor<?xi64>, tensor<?xi64>) loc("pad_s")
    %conv = stablehlo.convert %originVec#0 : (tensor<?xi64>) -> tensor<?xi32> loc("pad_s")
    %setd1 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.setDim"}, opType = "annc.setDim", reason = "unknown_op"} (%conv) : (tensor<?xi32>) -> tensor<*xi32> loc("pad_s")
    %c0 = stablehlo.constant dense<0> : tensor<1xi32> loc("Const:_0")
    %c1 = stablehlo.constant dense<1> : tensor<1xi32> loc("Const:_1")
    %ss = atir.unsupported {detail = "", metadata = {begin_mask = 0 : i64, call_target_name = "tf.StridedSlice", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, residue_reason = "unranked", shrink_axis_mask = 1 : i64}, opType = "StridedSlice", reason = "unknown_op"} (%setd1, %c0, %c1, %c1) : (tensor<*xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<*xi32> loc(fused["StridedSlice:", "origin0"])
    %getd2:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%ss) : (tensor<*xi32>) -> (tensor<?xi32>, tensor<?xi64>) loc("pad_s")
    %shp = shape.shape_of %data : tensor<?x4xf32> -> tensor<2xindex> loc(fused["Shape:", "shape_d"])
    %cast = arith.index_cast %shp : tensor<2xindex> to tensor<2xi32> loc(fused["Shape:", "shape_d"])
    %sl = stablehlo.slice %cast [0:1] : (tensor<2xi32>) -> tensor<1xi32> loc(fused["StridedSlice:", "num_rows"])
    %rows = stablehlo.reshape %sl {annc.source_node = "num_rows"} : (tensor<1xi32>) -> tensor<i32> loc(fused["StridedSlice:", "num_rows"])
    %getd3:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%rows) : (tensor<i32>) -> (tensor<1xi32>, tensor<?xi64>) loc("pad_s")
    %shp2 = shape.shape_of %getd2#0 : tensor<?xi32> -> tensor<1xindex> loc("pad_s")
    %cst_shape = shape.const_shape [1] : tensor<1xindex> loc("pad_s")
    %bshape = shape.broadcast %shp2, %cst_shape : tensor<1xindex>, tensor<1xindex> -> tensor<1xindex> loc("pad_s")
    %b1 = stablehlo.dynamic_broadcast_in_dim %getd2#0, %bshape, dims = [0] : (tensor<?xi32>, tensor<1xindex>) -> tensor<?xi32> loc("pad_s")
    %b2 = stablehlo.dynamic_broadcast_in_dim %getd3#0, %bshape, dims = [0] : (tensor<1xi32>, tensor<1xindex>) -> tensor<?xi32> loc("pad_s")
    %sub = stablehlo.subtract %b1, %b2 {annc.prov = ["serve_prep/trival_features:2", "num_rows:0"]} : tensor<?xi32> loc(fused["Sub:", "model/sub"])
    // setDim segment between sub and getDim removed.
    %getd4:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%sub) : (tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi64>) loc("pad_s")
    %c4 = stablehlo.constant dense<4> : tensor<i32> loc("Const:_pack")
    %getd5:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%c4) : (tensor<i32>) -> (tensor<1xi32>, tensor<?xi64>) loc("pad_s")
    %dims = stablehlo.concatenate %getd4#0, %getd5#0, dim = 0 : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32> loc("pad_s")
    %setd3 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.setDim"}, opType = "annc.setDim", reason = "unknown_op"} (%dims) : (tensor<?xi32>) -> tensor<*xi32> loc("pad_s")
    %c0f = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc("Const:_fill")
    %fill = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Fill", residue_reason = "unranked"}, opType = "Fill", reason = "unknown_op"} (%setd3, %c0f) : (tensor<*xi32>, tensor<f32>) -> tensor<?x4xf32> loc(fused["Fill:", "model/fill"])
    %concat = stablehlo.concatenate %data, %fill, dim = 0 : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32> loc(fused["ConcatV2:", "model/concat"])
    %c68 = stablehlo.constant dense<[-1, 240, 4]> : tensor<3xi32> loc("Const:_shape68")
    %out = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%concat, %c68) : (tensor<?x4xf32>, tensor<3xi32>) -> tensor<?x240x4xf32> loc(fused["Reshape:", "model/Reshape"])
    return %out : tensor<?x240x4xf32>
  }
}
