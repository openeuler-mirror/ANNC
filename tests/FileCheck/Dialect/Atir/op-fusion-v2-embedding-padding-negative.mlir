// RUN: annc-opt %s -atir-op-fusion-v2 | FileCheck %s
// op-fusion-v2 kp_embedding_padding negatives (fail-closed, see
// EmbeddingPaddingV2.cpp):
//   * @neg_fill_nonzero: Fill value != 0 - the kernel assumes zero-fill;
//   * @neg_data_dyn: dynamic data cols - the [-1, C] reshape_sizes cannot
//     be synthesized statically;
//   * @neg_no_bcast: missing broadcast on the subtract lhs (skeleton
//     violation) - the rank1 i64 origin terminal is unreachable;
//   * @neg_no_prov: structurally complete but the sub lacks annc.prov -
//     roles and wiring names are provenance-only, no guessing.
// Scaffold opType renames and hop changes do NOT reject (see the glue
// cases). The pass probes the whole module once @main exists.
// CHECK: func.func @main
// CHECK-NOT: annc.kernel
// CHECK: func.func @neg_data_dyn
// CHECK-NOT: annc.kernel
// CHECK: func.func @neg_no_bcast
// CHECK-NOT: annc.kernel
// CHECK: func.func @neg_no_prov
// CHECK-NOT: annc.kernel
// CHECK: return
module {
  func.func @main(%data: tensor<?x4xf32>, %hash: tensor<*x!atir.string>) -> tensor<?x240x4xf32>
      attributes {tf.arg_names = ["data", "hash"], tf.entry_function = {control_outputs = "", inputs = "data:0,hash:0", outputs = "out:0"}} {
    %s:3 = atir.unsupported {detail = "", metadata = {call_target_name = "tf.UnifyPbHashSparse", residue_reason = "unregistered_custom_op"}, opType = "UnifyPbHashSparse", reason = "unknown_op"} (%hash) : (tensor<*x!atir.string>) -> (tensor<?x?xi64>, tensor<?xi64>, tensor<*xi64>) loc(fused["UnifyPbHashSparse:", "serve_prep/trival_features"])
    %originVec:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%s#2) : (tensor<*xi64>) -> (tensor<?xi64>, tensor<?xi64>) loc("pad_s")
    %conv = stablehlo.convert %originVec#0 : (tensor<?xi64>) -> tensor<?xi32> loc("pad_s")
    %setd1 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.setDim"}, opType = "annc.setDim", reason = "unknown_op"} (%conv) : (tensor<?xi32>) -> tensor<*xi32> loc("pad_s")
    %c0 = stablehlo.constant dense<0> : tensor<1xi32> loc("pad_c0")
    %c1 = stablehlo.constant dense<1> : tensor<1xi32> loc("pad_c1")
    %ss = atir.unsupported {detail = "", metadata = {begin_mask = 0 : i64, call_target_name = "tf.StridedSlice", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, residue_reason = "unranked", shrink_axis_mask = 1 : i64}, opType = "StridedSlice", reason = "unknown_op"} (%setd1, %c0, %c1, %c1) : (tensor<*xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<*xi32> loc(fused["StridedSlice:", "origin0"])
    %getd2:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%ss) : (tensor<*xi32>) -> (tensor<?xi32>, tensor<?xi64>) loc("pad_s")
    %shp = shape.shape_of %data : tensor<?x4xf32> -> tensor<2xindex> loc("pad_shp")
    %cast = arith.index_cast %shp : tensor<2xindex> to tensor<2xi32> loc("pad_shp")
    %sl = stablehlo.slice %cast [0:1] : (tensor<2xi32>) -> tensor<1xi32> loc("pad_rows")
    %rows = stablehlo.reshape %sl : (tensor<1xi32>) -> tensor<i32> loc("pad_rows")
    %getd3:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%rows) : (tensor<i32>) -> (tensor<1xi32>, tensor<?xi64>) loc("pad_s")
    %shp2 = shape.shape_of %getd2#0 : tensor<?xi32> -> tensor<1xindex> loc("pad_s")
    %cst_shape = shape.const_shape [1] : tensor<1xindex> loc("pad_s")
    %bshape = shape.broadcast %shp2, %cst_shape : tensor<1xindex>, tensor<1xindex> -> tensor<1xindex> loc("pad_s")
    %b1 = stablehlo.dynamic_broadcast_in_dim %getd2#0, %bshape, dims = [0] : (tensor<?xi32>, tensor<1xindex>) -> tensor<?xi32> loc("pad_s")
    %b2 = stablehlo.dynamic_broadcast_in_dim %getd3#0, %bshape, dims = [0] : (tensor<1xi32>, tensor<1xindex>) -> tensor<?xi32> loc("pad_s")
    %sub = stablehlo.subtract %b1, %b2 {annc.prov = ["serve_prep/trival_features:2", "num_rows:0"]} : tensor<?xi32> loc(fused["Sub:", "model/sub"])
    %setd2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.setDim"}, opType = "annc.setDim", reason = "unknown_op"} (%sub) : (tensor<?xi32>) -> tensor<*xi32> loc("pad_s")
    %getd4:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%setd2) : (tensor<*xi32>) -> (tensor<?xi32>, tensor<?xi64>) loc("pad_s")
    %c4 = stablehlo.constant dense<4> : tensor<i32> loc("pad_c4")
    %getd5:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%c4) : (tensor<i32>) -> (tensor<1xi32>, tensor<?xi64>) loc("pad_s")
    %dims = stablehlo.concatenate %getd4#0, %getd5#0, dim = 0 : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32> loc("pad_s")
    %setd3 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.setDim"}, opType = "annc.setDim", reason = "unknown_op"} (%dims) : (tensor<?xi32>) -> tensor<*xi32> loc("pad_s")
    %c0f = stablehlo.constant dense<1.000000e+00> : tensor<f32> loc("pad_c0f")
    %fill = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Fill", residue_reason = "unranked"}, opType = "Fill", reason = "unknown_op"} (%setd3, %c0f) : (tensor<*xi32>, tensor<f32>) -> tensor<?x4xf32> loc(fused["Fill:", "model/fill"])
    %concat = stablehlo.concatenate %data, %fill, dim = 0 : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32> loc(fused["ConcatV2:", "model/concat"])
    %c68 = stablehlo.constant dense<[-1, 240, 4]> : tensor<3xi32> loc("pad_c68")
    %out = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%concat, %c68) : (tensor<?x4xf32>, tensor<3xi32>) -> tensor<?x240x4xf32> loc(fused["Reshape:", "model/Reshape"])
    return %out : tensor<?x240x4xf32>
  }

  func.func @neg_data_dyn(%data: tensor<?x?xf32>, %hash: tensor<*x!atir.string>) -> tensor<?x240x4xf32>
      attributes {tf.arg_names = ["data", "hash"], tf.entry_function = {control_outputs = "", inputs = "data:0,hash:0", outputs = "out:0"}} {
    %s:3 = atir.unsupported {detail = "", metadata = {call_target_name = "tf.UnifyPbHashSparse", residue_reason = "unregistered_custom_op"}, opType = "UnifyPbHashSparse", reason = "unknown_op"} (%hash) : (tensor<*x!atir.string>) -> (tensor<?x?xi64>, tensor<?xi64>, tensor<*xi64>) loc(fused["UnifyPbHashSparse:", "serve_prep/trival_features"])
    %originVec:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%s#2) : (tensor<*xi64>) -> (tensor<?xi64>, tensor<?xi64>) loc("pad_s")
    %conv = stablehlo.convert %originVec#0 : (tensor<?xi64>) -> tensor<?xi32> loc("pad_s")
    %setd1 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.setDim"}, opType = "annc.setDim", reason = "unknown_op"} (%conv) : (tensor<?xi32>) -> tensor<*xi32> loc("pad_s")
    %c0 = stablehlo.constant dense<0> : tensor<1xi32> loc("pad_c0")
    %c1 = stablehlo.constant dense<1> : tensor<1xi32> loc("pad_c1")
    %ss = atir.unsupported {detail = "", metadata = {begin_mask = 0 : i64, call_target_name = "tf.StridedSlice", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, residue_reason = "unranked", shrink_axis_mask = 1 : i64}, opType = "StridedSlice", reason = "unknown_op"} (%setd1, %c0, %c1, %c1) : (tensor<*xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<*xi32> loc(fused["StridedSlice:", "origin0"])
    %getd2:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%ss) : (tensor<*xi32>) -> (tensor<?xi32>, tensor<?xi64>) loc("pad_s")
    %shp = shape.shape_of %data : tensor<?x?xf32> -> tensor<2xindex> loc("pad_shp")
    %cast = arith.index_cast %shp : tensor<2xindex> to tensor<2xi32> loc("pad_shp")
    %sl = stablehlo.slice %cast [0:1] : (tensor<2xi32>) -> tensor<1xi32> loc("pad_rows")
    %rows = stablehlo.reshape %sl : (tensor<1xi32>) -> tensor<i32> loc("pad_rows")
    %getd3:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%rows) : (tensor<i32>) -> (tensor<1xi32>, tensor<?xi64>) loc("pad_s")
    %shp2 = shape.shape_of %getd2#0 : tensor<?xi32> -> tensor<1xindex> loc("pad_s")
    %cst_shape = shape.const_shape [1] : tensor<1xindex> loc("pad_s")
    %bshape = shape.broadcast %shp2, %cst_shape : tensor<1xindex>, tensor<1xindex> -> tensor<1xindex> loc("pad_s")
    %b1 = stablehlo.dynamic_broadcast_in_dim %getd2#0, %bshape, dims = [0] : (tensor<?xi32>, tensor<1xindex>) -> tensor<?xi32> loc("pad_s")
    %b2 = stablehlo.dynamic_broadcast_in_dim %getd3#0, %bshape, dims = [0] : (tensor<1xi32>, tensor<1xindex>) -> tensor<?xi32> loc("pad_s")
    %sub = stablehlo.subtract %b1, %b2 {annc.prov = ["serve_prep/trival_features:2", "num_rows:0"]} : tensor<?xi32> loc(fused["Sub:", "model/sub"])
    %setd2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.setDim"}, opType = "annc.setDim", reason = "unknown_op"} (%sub) : (tensor<?xi32>) -> tensor<*xi32> loc("pad_s")
    %getd4:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%setd2) : (tensor<*xi32>) -> (tensor<?xi32>, tensor<?xi64>) loc("pad_s")
    %c4 = stablehlo.constant dense<4> : tensor<i32> loc("pad_c4")
    %getd5:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%c4) : (tensor<i32>) -> (tensor<1xi32>, tensor<?xi64>) loc("pad_s")
    %dims = stablehlo.concatenate %getd4#0, %getd5#0, dim = 0 : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32> loc("pad_s")
    %setd3 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.setDim"}, opType = "annc.setDim", reason = "unknown_op"} (%dims) : (tensor<?xi32>) -> tensor<*xi32> loc("pad_s")
    %c0f = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc("pad_c0f")
    %fill = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Fill", residue_reason = "unranked"}, opType = "Fill", reason = "unknown_op"} (%setd3, %c0f) : (tensor<*xi32>, tensor<f32>) -> tensor<?x?xf32> loc(fused["Fill:", "model/fill"])
    %concat = stablehlo.concatenate %data, %fill, dim = 0 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32> loc(fused["ConcatV2:", "model/concat"])
    %c68 = stablehlo.constant dense<[-1, 240, 4]> : tensor<3xi32> loc("pad_c68")
    %out = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%concat, %c68) : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x240x4xf32> loc(fused["Reshape:", "model/Reshape"])
    return %out : tensor<?x240x4xf32>
  }

  func.func @neg_no_bcast(%data: tensor<?x4xf32>, %hash: tensor<*x!atir.string>) -> tensor<?x240x4xf32>
      attributes {tf.arg_names = ["data", "hash"], tf.entry_function = {control_outputs = "", inputs = "data:0,hash:0", outputs = "out:0"}} {
    %s:3 = atir.unsupported {detail = "", metadata = {call_target_name = "tf.UnifyPbHashSparse", residue_reason = "unregistered_custom_op"}, opType = "UnifyPbHashSparse", reason = "unknown_op"} (%hash) : (tensor<*x!atir.string>) -> (tensor<?x?xi64>, tensor<?xi64>, tensor<*xi64>) loc(fused["UnifyPbHashSparse:", "serve_prep/trival_features"])
    %originVec:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%s#2) : (tensor<*xi64>) -> (tensor<?xi64>, tensor<?xi64>) loc("pad_s")
    %conv = stablehlo.convert %originVec#0 : (tensor<?xi64>) -> tensor<?xi32> loc("pad_s")
    %setd1 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.setDim"}, opType = "annc.setDim", reason = "unknown_op"} (%conv) : (tensor<?xi32>) -> tensor<*xi32> loc("pad_s")
    %c0 = stablehlo.constant dense<0> : tensor<1xi32> loc("pad_c0")
    %c1 = stablehlo.constant dense<1> : tensor<1xi32> loc("pad_c1")
    %ss = atir.unsupported {detail = "", metadata = {begin_mask = 0 : i64, call_target_name = "tf.StridedSlice", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, residue_reason = "unranked", shrink_axis_mask = 1 : i64}, opType = "StridedSlice", reason = "unknown_op"} (%setd1, %c0, %c1, %c1) : (tensor<*xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<*xi32> loc(fused["StridedSlice:", "origin0"])
    %getd2:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%ss) : (tensor<*xi32>) -> (tensor<?xi32>, tensor<?xi64>) loc("pad_s")
    %shp = shape.shape_of %data : tensor<?x4xf32> -> tensor<2xindex> loc("pad_shp")
    %cast = arith.index_cast %shp : tensor<2xindex> to tensor<2xi32> loc("pad_shp")
    %sl = stablehlo.slice %cast [0:1] : (tensor<2xi32>) -> tensor<1xi32> loc("pad_rows")
    %rows = stablehlo.reshape %sl : (tensor<1xi32>) -> tensor<i32> loc("pad_rows")
    %getd3:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%rows) : (tensor<i32>) -> (tensor<1xi32>, tensor<?xi64>) loc("pad_s")
    %shp2 = shape.shape_of %getd2#0 : tensor<?xi32> -> tensor<1xindex> loc("pad_s")
    %cst_shape = shape.const_shape [1] : tensor<1xindex> loc("pad_s")
    %bshape = shape.broadcast %shp2, %cst_shape : tensor<1xindex>, tensor<1xindex> -> tensor<1xindex> loc("pad_s")
    %b2 = stablehlo.dynamic_broadcast_in_dim %getd3#0, %bshape, dims = [0] : (tensor<1xi32>, tensor<1xindex>) -> tensor<?xi32> loc("pad_s")
    // Break: the lhs is not broadcast (rank1 getDim value direct).
    %sub = stablehlo.subtract %getd2#0, %b2 {annc.prov = ["serve_prep/trival_features:2", "num_rows:0"]} : tensor<?xi32> loc(fused["Sub:", "model/sub"])
    %setd2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.setDim"}, opType = "annc.setDim", reason = "unknown_op"} (%sub) : (tensor<?xi32>) -> tensor<*xi32> loc("pad_s")
    %getd4:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%setd2) : (tensor<*xi32>) -> (tensor<?xi32>, tensor<?xi64>) loc("pad_s")
    %c4 = stablehlo.constant dense<4> : tensor<i32> loc("pad_c4")
    %getd5:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%c4) : (tensor<i32>) -> (tensor<1xi32>, tensor<?xi64>) loc("pad_s")
    %dims = stablehlo.concatenate %getd4#0, %getd5#0, dim = 0 : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32> loc("pad_s")
    %setd3 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.setDim"}, opType = "annc.setDim", reason = "unknown_op"} (%dims) : (tensor<?xi32>) -> tensor<*xi32> loc("pad_s")
    %c0f = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc("pad_c0f")
    %fill = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Fill", residue_reason = "unranked"}, opType = "Fill", reason = "unknown_op"} (%setd3, %c0f) : (tensor<*xi32>, tensor<f32>) -> tensor<?x4xf32> loc(fused["Fill:", "model/fill"])
    %concat = stablehlo.concatenate %data, %fill, dim = 0 : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32> loc(fused["ConcatV2:", "model/concat"])
    %c68 = stablehlo.constant dense<[-1, 240, 4]> : tensor<3xi32> loc("pad_c68")
    %out = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%concat, %c68) : (tensor<?x4xf32>, tensor<3xi32>) -> tensor<?x240x4xf32> loc(fused["Reshape:", "model/Reshape"])
    return %out : tensor<?x240x4xf32>
  }

  func.func @neg_no_prov(%data: tensor<?x4xf32>, %hash: tensor<*x!atir.string>) -> tensor<?x240x4xf32>
      attributes {tf.arg_names = ["data", "hash"], tf.entry_function = {control_outputs = "", inputs = "data:0,hash:0", outputs = "out:0"}} {
    %s:3 = atir.unsupported {detail = "", metadata = {call_target_name = "tf.UnifyPbHashSparse", residue_reason = "unregistered_custom_op"}, opType = "UnifyPbHashSparse", reason = "unknown_op"} (%hash) : (tensor<*x!atir.string>) -> (tensor<?x?xi64>, tensor<?xi64>, tensor<*xi64>) loc(fused["UnifyPbHashSparse:", "serve_prep/trival_features"])
    %originVec:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%s#2) : (tensor<*xi64>) -> (tensor<?xi64>, tensor<?xi64>) loc("pad_s")
    %conv = stablehlo.convert %originVec#0 : (tensor<?xi64>) -> tensor<?xi32> loc("pad_s")
    %setd1 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.setDim"}, opType = "annc.setDim", reason = "unknown_op"} (%conv) : (tensor<?xi32>) -> tensor<*xi32> loc("pad_s")
    %c0 = stablehlo.constant dense<0> : tensor<1xi32> loc("pad_c0")
    %c1 = stablehlo.constant dense<1> : tensor<1xi32> loc("pad_c1")
    %ss = atir.unsupported {detail = "", metadata = {begin_mask = 0 : i64, call_target_name = "tf.StridedSlice", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, residue_reason = "unranked", shrink_axis_mask = 1 : i64}, opType = "StridedSlice", reason = "unknown_op"} (%setd1, %c0, %c1, %c1) : (tensor<*xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<*xi32> loc(fused["StridedSlice:", "origin0"])
    %getd2:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%ss) : (tensor<*xi32>) -> (tensor<?xi32>, tensor<?xi64>) loc("pad_s")
    %shp = shape.shape_of %data : tensor<?x4xf32> -> tensor<2xindex> loc("pad_shp")
    %cast = arith.index_cast %shp : tensor<2xindex> to tensor<2xi32> loc("pad_shp")
    %sl = stablehlo.slice %cast [0:1] : (tensor<2xi32>) -> tensor<1xi32> loc("pad_rows")
    %rows = stablehlo.reshape %sl : (tensor<1xi32>) -> tensor<i32> loc("pad_rows")
    %getd3:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%rows) : (tensor<i32>) -> (tensor<1xi32>, tensor<?xi64>) loc("pad_s")
    %shp2 = shape.shape_of %getd2#0 : tensor<?xi32> -> tensor<1xindex> loc("pad_s")
    %cst_shape = shape.const_shape [1] : tensor<1xindex> loc("pad_s")
    %bshape = shape.broadcast %shp2, %cst_shape : tensor<1xindex>, tensor<1xindex> -> tensor<1xindex> loc("pad_s")
    %b1 = stablehlo.dynamic_broadcast_in_dim %getd2#0, %bshape, dims = [0] : (tensor<?xi32>, tensor<1xindex>) -> tensor<?xi32> loc("pad_s")
    %b2 = stablehlo.dynamic_broadcast_in_dim %getd3#0, %bshape, dims = [0] : (tensor<1xi32>, tensor<1xindex>) -> tensor<?xi32> loc("pad_s")
    // Break: the sub lacks annc.prov (provenance-only roles).
    %sub = stablehlo.subtract %b1, %b2 : tensor<?xi32> loc(fused["Sub:", "model/sub"])
    %setd2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.setDim"}, opType = "annc.setDim", reason = "unknown_op"} (%sub) : (tensor<?xi32>) -> tensor<*xi32> loc("pad_s")
    %getd4:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%setd2) : (tensor<*xi32>) -> (tensor<?xi32>, tensor<?xi64>) loc("pad_s")
    %c4 = stablehlo.constant dense<4> : tensor<i32> loc("pad_c4")
    %getd5:2 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.getDim"}, opType = "annc.getDim", reason = "unknown_op"} (%c4) : (tensor<i32>) -> (tensor<1xi32>, tensor<?xi64>) loc("pad_s")
    %dims = stablehlo.concatenate %getd4#0, %getd5#0, dim = 0 : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32> loc("pad_s")
    %setd3 = atir.unsupported {detail = "", metadata = {call_target_name = "annc.setDim"}, opType = "annc.setDim", reason = "unknown_op"} (%dims) : (tensor<?xi32>) -> tensor<*xi32> loc("pad_s")
    %c0f = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc("pad_c0f")
    %fill = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Fill", residue_reason = "unranked"}, opType = "Fill", reason = "unknown_op"} (%setd3, %c0f) : (tensor<*xi32>, tensor<f32>) -> tensor<?x4xf32> loc(fused["Fill:", "model/fill"])
    %concat = stablehlo.concatenate %data, %fill, dim = 0 : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32> loc(fused["ConcatV2:", "model/concat"])
    %c68 = stablehlo.constant dense<[-1, 240, 4]> : tensor<3xi32> loc("pad_c68")
    %out = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%concat, %c68) : (tensor<?x4xf32>, tensor<3xi32>) -> tensor<?x240x4xf32> loc(fused["Reshape:", "model/Reshape"])
    return %out : tensor<?x240x4xf32>
  }
}
