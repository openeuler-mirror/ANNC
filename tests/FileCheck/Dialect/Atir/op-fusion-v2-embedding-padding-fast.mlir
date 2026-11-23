// RUN: annc-opt %s -atir-op-fusion-v2 | FileCheck %s
// RUN: annc-opt %s -atir-op-fusion-v2 | annc-asm - --annc-fast-codegen | FileCheck %s --check-prefix=CODEGEN
//
// op-fusion-v2 pattern: kp_embedding_padding_fast (minimal-construct sink
// form, see EmbeddingPaddingFastV2.cpp):
//   %origin_i32 = convert(origin_shape)
//   %o0    = reshape(slice(%origin_i32, [0:1]))
//   %sub   = subtract(%o0, input_rows)          // rank0 i32
//   %dims  = concatenate(reshape(%sub), reshape(pack), dim=0)
//   %fill  = dynamic_broadcast_in_dim(0.0f, %dims)
//   %pad   = concatenate(data, %fill, dim=0)
//   %padr  = Reshape residue(%pad, reshape_sizes)
//   %rows  = reshape(slice(index_cast(shape_of(%padr)), [0:1]))  // anchor
// Outputs are the (padding_rows, reshape_rows) rank-0 pair; the site ABI
// shares the padding schema order. @neg_no_prov drops the site (roles are
// provenance-only, no annc.prov on the sub means no guessing).
module {
  func.func @main(%origin: tensor<?xi64>, %rows_in: tensor<i32>, %pack: tensor<i32>, %data: tensor<?x8xf32>) -> tensor<i32>
      attributes {tf.arg_names = ["origin_shape", "input_rows", "pack", "data"], tf.entry_function = {control_outputs = "", inputs = "origin_shape:0,input_rows:0,pack:0,data:0", outputs = "out:0"}} {
    %conv = stablehlo.convert %origin : (tensor<?xi64>) -> tensor<?xi32> loc("pad_chain")
    %sl = stablehlo.slice %conv [0:1] : (tensor<?xi32>) -> tensor<1xi32> loc("pad_chain")
    %o0 = stablehlo.reshape %sl : (tensor<1xi32>) -> tensor<i32> loc("pad_chain")
    %sub = stablehlo.subtract %o0, %rows_in {annc.prov = ["origin_shape:0", "input_rows:0"]} : tensor<i32> loc("model/sub")
    %subv = stablehlo.reshape %sub : (tensor<i32>) -> tensor<1xi32> loc("pad_chain")
    %packv = stablehlo.reshape %pack : (tensor<i32>) -> tensor<1xi32> loc("pad_chain")
    %dims = stablehlo.concatenate %subv, %packv, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc("pad_chain")
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc("pad_chain")
    %fill = stablehlo.dynamic_broadcast_in_dim %zero, %dims, dims = [] : (tensor<f32>, tensor<2xi32>) -> tensor<?x8xf32> loc("model/fill")
    %pad = stablehlo.concatenate %data, %fill, dim = 0 : (tensor<?x8xf32>, tensor<?x8xf32>) -> tensor<?x8xf32> loc("model/concat")
    %rs = stablehlo.constant dense<[-1, 8]> : tensor<2xi32> loc("pad_chain")
    %padr = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%pad, %rs) : (tensor<?x8xf32>, tensor<2xi32>) -> tensor<?x8xf32> loc("model/reshape")
    %shape = shape.shape_of %padr : tensor<?x8xf32> -> tensor<2xindex> loc("pad_chain")
    %cast = arith.index_cast %shape : tensor<2xindex> to tensor<2xi32> loc("pad_chain")
    %sl2 = stablehlo.slice %cast [0:1] : (tensor<2xi32>) -> tensor<1xi32> loc("pad_chain")
    %rows = stablehlo.reshape %sl2 : (tensor<1xi32>) -> tensor<i32> loc("pad_rows")
    return %rows : tensor<i32> loc("output")
  }
  func.func @neg_no_prov(%origin: tensor<?xi64>, %rows_in: tensor<i32>, %pack: tensor<i32>, %data: tensor<?x8xf32>) -> tensor<i32>
      attributes {tf.arg_names = ["origin_shape", "input_rows", "pack", "data"], tf.entry_function = {control_outputs = "", inputs = "origin_shape:0,input_rows:0,pack:0,data:0", outputs = "out:0"}} {
    %conv = stablehlo.convert %origin : (tensor<?xi64>) -> tensor<?xi32> loc("pad_chain")
    %sl = stablehlo.slice %conv [0:1] : (tensor<?xi32>) -> tensor<1xi32> loc("pad_chain")
    %o0 = stablehlo.reshape %sl : (tensor<1xi32>) -> tensor<i32> loc("pad_chain")
    %sub = stablehlo.subtract %o0, %rows_in : tensor<i32> loc("model/sub")
    %subv = stablehlo.reshape %sub : (tensor<i32>) -> tensor<1xi32> loc("pad_chain")
    %packv = stablehlo.reshape %pack : (tensor<i32>) -> tensor<1xi32> loc("pad_chain")
    %dims = stablehlo.concatenate %subv, %packv, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc("pad_chain")
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc("pad_chain")
    %fill = stablehlo.dynamic_broadcast_in_dim %zero, %dims, dims = [] : (tensor<f32>, tensor<2xi32>) -> tensor<?x8xf32> loc("model/fill")
    %pad = stablehlo.concatenate %data, %fill, dim = 0 : (tensor<?x8xf32>, tensor<?x8xf32>) -> tensor<?x8xf32> loc("model/concat")
    %rs = stablehlo.constant dense<[-1, 8]> : tensor<2xi32> loc("pad_chain")
    %padr = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%pad, %rs) : (tensor<?x8xf32>, tensor<2xi32>) -> tensor<?x8xf32> loc("model/reshape")
    %shape = shape.shape_of %padr : tensor<?x8xf32> -> tensor<2xindex> loc("pad_chain")
    %cast = arith.index_cast %shape : tensor<2xindex> to tensor<2xi32> loc("pad_chain")
    %sl2 = stablehlo.slice %cast [0:1] : (tensor<2xi32>) -> tensor<1xi32> loc("pad_chain")
    %rows = stablehlo.reshape %sl2 : (tensor<1xi32>) -> tensor<i32> loc("pad_rows_neg")
    return %rows : tensor<i32> loc("output")
  }
}

// CHECK: call @KPFusedEmbeddingPaddingFast_{{[0-9a-zA-Z_]+}}(
// CHECK-SAME: ) -> (tensor<i32>, tensor<i32>)
// CHECK: func.func @neg_no_prov
// CHECK-NOT: annc.kernel
// CHECK: func.func private @KPFusedEmbeddingPaddingFast_{{[0-9a-zA-Z_]+}}
// CHECK-SAME: {{.*abi = "annc_context_v3".*args = .*tf_name = "origin_shape".*tf_name = "input_rows".*tf_name = "pack".*tf_name = "data".*tf_name = "pad_rows/shape".*fusion.pattern = "kp_embedding_padding_fast".*tf_name = "model/sub:0".*tf_name = "pad_rows:0".*}}
// CHECK: return

// Second layer: Customize rewrite (registry builtin, dual rank-0 results).
// CODEGEN: atir.Customize
// CODEGEN-SAME: custom.op_name = "KPFusedEmbeddingPaddingFast"
// CODEGEN: return
