// RUN: annc-opt %s -atir-op-fusion-v2 | FileCheck %s
// RUN: annc-opt %s -atir-op-fusion-v2 | annc-asm - --annc-fast-codegen | FileCheck %s --check-prefix=CODEGEN
//
// op-fusion-v2 pattern: kp_embedding_action_id_gather (the V2 carrier port
// of the V1 KPFusedEmbeddingActionIdGather). Shape from a real pb sink
// artifact: gather legalized to torch_index_select x2, the runtime-shaped
// tf.Reshape falls back to the residue carrier, graph entries stay as
// placeholder residues (ABI names come from anchors, not arg names).
// ABI = schema order (params, indices1, indices2, pack_dim, pack), output
// (w:0), kernel name rank-specialized (R2x2).
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main() -> tensor<?x?xf32> attributes {tf.arg_names = [], tf.entry_function = {control_outputs = "", inputs = "", outputs = "w:0"}} {
    %c = stablehlo.constant dense<-1> : tensor<1xi32> loc(#loc)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc)
    %indices1 = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Placeholder", device = "", residue_reason = "no_legalize_pattern", shape = #tf_type.shape<2x1>}, opType = "Placeholder", reason = "unknown_op"} () : () -> tensor<2x1xi64> loc(fused["Placeholder:", "indices1"])
    %indices2 = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Placeholder", device = "", residue_reason = "no_legalize_pattern", shape = #tf_type.shape<1x2>}, opType = "Placeholder", reason = "unknown_op"} () : () -> tensor<1x2xi32> loc(fused["Placeholder:", "indices2"])
    %pack = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Placeholder", device = "", residue_reason = "no_legalize_pattern", shape = #tf_type.shape<>}, opType = "Placeholder", reason = "unknown_op"} () : () -> tensor<i32> loc(fused["Placeholder:", "pack"])
    %pack_dim = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Placeholder", device = "", residue_reason = "no_legalize_pattern", shape = #tf_type.shape<>}, opType = "Placeholder", reason = "unknown_op"} () : () -> tensor<i32> loc(fused["Placeholder:", "pack_dim"])
    %pack_dim_r = stablehlo.reshape %pack_dim : (tensor<i32>) -> tensor<1xi32> loc(fused["Pack:", "pack_dims"])
    %pack_r = stablehlo.reshape %pack : (tensor<i32>) -> tensor<1xi32> loc(fused["Pack:", "pack_dims"])
    %pack_dims = stablehlo.concatenate %pack_dim_r, %pack_r, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(fused["Pack:", "pack_dims"])
    %pack_dim_r2 = stablehlo.reshape %pack_dim : (tensor<i32>) -> tensor<1xi32> loc(fused["Pack:", "pack_shape"])
    %pack_shape = stablehlo.concatenate %pack_dim_r2, %c, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(fused["Pack:", "pack_shape"])
    %params = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Placeholder", device = "", residue_reason = "no_legalize_pattern", shape = #tf_type.shape<10x4>}, opType = "Placeholder", reason = "unknown_op"} () : () -> tensor<10x4xf32> loc(fused["Placeholder:", "params"])
    %g1 = "stablehlo.torch_index_select"(%params, %indices1) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<10x4xf32>, tensor<2x1xi64>) -> tensor<2x1x4xf32> loc(fused["GatherV2:", "inner_gather"])
    %g2 = "stablehlo.torch_index_select"(%g1, %indices2) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<2x1x4xf32>, tensor<1x2xi32>) -> tensor<1x2x1x4xf32> loc(fused["GatherV2:", "outer_gather"])
    %rs = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", device = "", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%g2, %pack_shape) : (tensor<1x2x1x4xf32>, tensor<2xi32>) -> tensor<?x?xf32> loc(fused["Reshape:", "reshaped"])
    %fill = stablehlo.dynamic_broadcast_in_dim %cst, %pack_dims, dims = [] : (tensor<f32>, tensor<2xi32>) -> tensor<?x?xf32> loc(fused["Fill:", "fill_out"])
    %w = stablehlo.concatenate %rs, %fill, dim = 1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32> loc(fused["ConcatV2:", "w"])
    return %w : tensor<?x?xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)

// CHECK: call @KPFusedEmbeddingActionIdGatherR2x2(
// CHECK-SAME: ) -> tensor<?x?xf32>
// CHECK: func.func private @KPFusedEmbeddingActionIdGatherR2x2
// CHECK-SAME: annc.kernel
// CHECK-SAME: abi = "annc_context_v3"
// CHECK-SAME: args = [{dtype = "f32", rank = 2 : i64, role = "dynamic", shape = [10, 4], tf_name = "params"}
// CHECK-SAME: {dtype = "i64", rank = 2 : i64, role = "dynamic", shape = [2, 1], tf_name = "indices1"}
// CHECK-SAME: {dtype = "i32", rank = 2 : i64, role = "dynamic", shape = [1, 2], tf_name = "indices2"}
// CHECK-SAME: {dtype = "i32", rank = 0 : i64, role = "dynamic", shape = [], tf_name = "pack_dim"}
// CHECK-SAME: {dtype = "i32", rank = 0 : i64, role = "dynamic", shape = [], tf_name = "pack"}]
// CHECK-SAME: fusion.pattern = "kp_embedding_action_id_gather"
// CHECK-SAME: kernel_name = "KPFusedEmbeddingActionIdGatherR2x2"
// CHECK-SAME: outputs = [{dtype = "f32", rank = 2 : i64, role = "output", shape = [-9223372036854775808, -9223372036854775808], tf_name = "w:0"}]


// CODEGEN: atir.Customize
// CODEGEN-SAME: custom.op_name = "KPFusedEmbeddingActionIdGatherR2x2"
// CODEGEN: return
