// RUN: annc-opt %s -atir-op-fusion-v2 | FileCheck %s
// RUN: annc-opt %s -atir-op-fusion-v2 | annc-asm - -atir-fast-codegen | FileCheck %s --check-prefix=CODEGEN
//
// op-fusion-v2 pattern: kp_gather (KP embedding gather, the V2 carrier
// port of the V1 KPFusedGather). Shapes from the cvr_3 sink artifact (keys
// retyped to i64 for the sole Tkeys=int64_t kernel specialization):
//   keys2d = maximum(keys, broadcast 0) (clip chain head, static dim1 == 1)
//   keys1d = unsupported{Reshape}(keys2d, reshape(reduce(index_cast(
//            shape_of(keys2d))))) (1-element shape chain - flatten or fail)
//   uniq:2 = unsupported{Unique}(keys1d)
//   gathered = torch_index_select(table, uniq#0)
//   restored = torch_index_select(gathered, uniq#1) (stays in main)
// Site ABI = schema order (data, keys, begin): table/keys2d promoted via
// promotedConstants, begin synthesized as [0,0] (kernel reads column 0;
// flatten == column 0 only when dim1 == 1); outputs (unique_values,
// unique_indices, gathered).
module {
  func.func @main(%table: tensor<100x4xf32>, %keys: tensor<?x1xi64>) -> tensor<?x4xf32>
      attributes {tf.arg_names = ["table", "keys"], tf.entry_function = {control_outputs = "", inputs = "table:0,keys:0", outputs = "restored:0"}} {
    %cst = stablehlo.constant dense<0> : tensor<i64> loc("Const:_limit")
    %shp = shape.shape_of %keys : tensor<?x1xi64> -> tensor<2xindex> loc(fused["Shape:", "shape_k"])
    %bc = stablehlo.dynamic_broadcast_in_dim %cst, %shp, dims = [] : (tensor<i64>, tensor<2xindex>) -> tensor<?x1xi64> loc(fused["Maximum:", "clip"])
    %keys2d = stablehlo.maximum %keys, %bc : tensor<?x1xi64> loc(fused["Maximum:", "clip"])
    %nshp = shape.shape_of %keys2d : tensor<?x1xi64> -> tensor<2xindex> loc(fused["Shape:", "numel"])
    %ncast = arith.index_cast %nshp : tensor<2xindex> to tensor<2xi32> loc(fused["Shape:", "numel"])
    %c_one = stablehlo.constant dense<1> : tensor<i32> loc("Const:_one")
    %prod = stablehlo.reduce(%ncast init: %c_one) applies stablehlo.multiply across dimensions = [0] : (tensor<2xi32>, tensor<i32>) -> tensor<i32> loc(fused["Prod:", "numel"])
    %shap1 = stablehlo.reshape %prod : (tensor<i32>) -> tensor<1xi32> loc(fused["Reshape:", "numel_r"])
    %keys1d = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", device = "/job:localhost/replica:0/task:0/device:CPU:0", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%keys2d, %shap1) : (tensor<?x1xi64>, tensor<1xi32>) -> tensor<?xi64> loc(fused["Reshape:", "keys_flat"])
    %uniq:2 = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Unique", device = "/job:localhost/replica:0/task:0/device:CPU:0", residue_reason = "no_legalize_pattern"}, opType = "Unique", reason = "unknown_op"} (%keys1d) : (tensor<?xi64>) -> (tensor<?xi64>, tensor<?xi32>) loc(fused["Unique:", "uniq_node"])
    %gathered = "stablehlo.torch_index_select"(%table, %uniq#0) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<100x4xf32>, tensor<?xi64>) -> tensor<?x4xf32> loc(fused["GatherV2:", "gather_node"])
    %restored = "stablehlo.torch_index_select"(%gathered, %uniq#1) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32> loc(fused["GatherV2:", "restored"])
    return %restored : tensor<?x4xf32> loc("output")
  }
}

// CHECK: call @KPFusedGather_{{[0-9a-zA-Z_]+}}(%arg0, %{{[0-9]+}}, %c)
// CHECK-SAME: (tensor<100x4xf32>, tensor<?x1xi64>, tensor<2xi32>) -> (tensor<?xi64>, tensor<?xi32>, tensor<?x4xf32>)
// CHECK: func.func private @KPFusedGather_{{[0-9a-zA-Z_]+}}
// CHECK-SAME: annc.kernel
// CHECK-SAME: abi = "annc_context_v3"
// CHECK-SAME: args = [{dtype = "f32", rank = 2 : i64, role = "dynamic", shape = [100, 4], tf_name = "table"}, {dtype = "i64", rank = 2 : i64, role = "dynamic", shape = [-9223372036854775808, 1], tf_name = "clip"}, {dtype = "i32", rank = 1 : i64, role = "constant", shape = [2], tf_name = "uniq_node_begin"}]
// CHECK-SAME: fallback_function = "original_subgraph"
// CHECK-SAME: fusion.pattern = "kp_gather"
// CHECK-SAME: kernel_name = "KPFusedGather_{{[0-9a-zA-Z_]+}}"
// CHECK-SAME: outputs = [{dtype = "i64", rank = 1 : i64, role = "output", shape = [-9223372036854775808], tf_name = "uniq_node:0"}, {dtype = "i32", rank = 1 : i64, role = "output", shape = [-9223372036854775808], tf_name = "uniq_node:1"}, {dtype = "f32", rank = 2 : i64, role = "output", shape = [-9223372036854775808, 4], tf_name = "gather_node:0"}]
// Kernel body: Unique -> torch_index_select (tail-dependency order).
// CHECK: atir.unsupported {{.*}}opType = "Unique"
// CHECK: "stablehlo.torch_index_select"

// Second layer: Customize rewrite (V1 same-name spec, registry builtin).
// CODEGEN: atir.Customize
// CODEGEN-SAME: custom.op_name = "KPFusedGather"
// CODEGEN: return
