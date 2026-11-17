// RUN: annc-opt %s -atir-op-fusion-v2 | FileCheck %s
// op-fusion-v2 kp_gather negatives (fail-closed; see KpGatherV2.cpp):
//   * @neg_multicol: keys2d static dim1 == 2 - flatten != column 0 (the
//     original unique spans n*k values; the kernel uniques one column
//     only): semantically inequivalent, rejected;
//   * @neg_i32keys: keys2d i32 - only the Tkeys=int64_t specialization is
//     registered, rejected;
//   * @neg_dbluse: uniq#0 has two users - the single-consumer constraint
//     is broken, rejected.
// CHECK: func.func @neg_multicol
// CHECK-NOT: annc.kernel
// CHECK: func.func @neg_i32keys
// CHECK-NOT: annc.kernel
// CHECK: func.func @neg_dbluse
// CHECK-NOT: annc.kernel
// CHECK: return
module {
  // The pass skips a module without @main; this empty entry unlocks
  // probing so the negatives below are truly exercised.
  func.func @main() {
    return
  }

  func.func @neg_multicol(%table: tensor<100x4xf32>, %keys: tensor<?x2xi64>) -> tensor<?x4xf32> {
    %cst = stablehlo.constant dense<0> : tensor<i64> loc("Const:_limit")
    %shp = shape.shape_of %keys : tensor<?x2xi64> -> tensor<2xindex> loc(fused["Shape:", "shape_k"])
    %bc = stablehlo.dynamic_broadcast_in_dim %cst, %shp, dims = [] : (tensor<i64>, tensor<2xindex>) -> tensor<?x2xi64> loc(fused["Maximum:", "clip"])
    %keys2d = stablehlo.maximum %keys, %bc : tensor<?x2xi64> loc(fused["Maximum:", "clip"])
    %nshp = shape.shape_of %keys2d : tensor<?x2xi64> -> tensor<2xindex> loc(fused["Shape:", "numel"])
    %ncast = arith.index_cast %nshp : tensor<2xindex> to tensor<2xi32> loc(fused["Shape:", "numel"])
    %c_one = stablehlo.constant dense<1> : tensor<i32> loc("Const:_one")
    %prod = stablehlo.reduce(%ncast init: %c_one) applies stablehlo.multiply across dimensions = [0] : (tensor<2xi32>, tensor<i32>) -> tensor<i32> loc(fused["Prod:", "numel"])
    %shap1 = stablehlo.reshape %prod : (tensor<i32>) -> tensor<1xi32> loc(fused["Reshape:", "numel_r"])
    %keys1d = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", device = "/job:localhost/replica:0/task:0/device:CPU:0", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%keys2d, %shap1) : (tensor<?x2xi64>, tensor<1xi32>) -> tensor<?xi64> loc(fused["Reshape:", "keys_flat"])
    %uniq:2 = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Unique", device = "/job:localhost/replica:0/task:0/device:CPU:0", residue_reason = "no_legalize_pattern"}, opType = "Unique", reason = "unknown_op"} (%keys1d) : (tensor<?xi64>) -> (tensor<?xi64>, tensor<?xi32>) loc(fused["Unique:", "uniq_node"])
    %gathered = "stablehlo.torch_index_select"(%table, %uniq#0) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<100x4xf32>, tensor<?xi64>) -> tensor<?x4xf32> loc(fused["GatherV2:", "gather_node"])
    %restored = "stablehlo.torch_index_select"(%gathered, %uniq#1) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32> loc(fused["GatherV2:", "restored"])
    return %restored : tensor<?x4xf32>
  }

  func.func @neg_i32keys(%table: tensor<100x4xf32>, %keys: tensor<?x1xi32>) -> tensor<?x4xf32> {
    %cst = stablehlo.constant dense<0> : tensor<i32> loc("Const:_limit")
    %shp = shape.shape_of %keys : tensor<?x1xi32> -> tensor<2xindex> loc(fused["Shape:", "shape_k"])
    %bc = stablehlo.dynamic_broadcast_in_dim %cst, %shp, dims = [] : (tensor<i32>, tensor<2xindex>) -> tensor<?x1xi32> loc(fused["Maximum:", "clip"])
    %keys2d = stablehlo.maximum %keys, %bc : tensor<?x1xi32> loc(fused["Maximum:", "clip"])
    %nshp = shape.shape_of %keys2d : tensor<?x1xi32> -> tensor<2xindex> loc(fused["Shape:", "numel"])
    %ncast = arith.index_cast %nshp : tensor<2xindex> to tensor<2xi32> loc(fused["Shape:", "numel"])
    %c_one = stablehlo.constant dense<1> : tensor<i32> loc("Const:_one")
    %prod = stablehlo.reduce(%ncast init: %c_one) applies stablehlo.multiply across dimensions = [0] : (tensor<2xi32>, tensor<i32>) -> tensor<i32> loc(fused["Prod:", "numel"])
    %shap1 = stablehlo.reshape %prod : (tensor<i32>) -> tensor<1xi32> loc(fused["Reshape:", "numel_r"])
    %keys1d = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", device = "/job:localhost/replica:0/task:0/device:CPU:0", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%keys2d, %shap1) : (tensor<?x1xi32>, tensor<1xi32>) -> tensor<?xi32> loc(fused["Reshape:", "keys_flat"])
    %uniq:2 = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Unique", device = "/job:localhost/replica:0/task:0/device:CPU:0", residue_reason = "no_legalize_pattern"}, opType = "Unique", reason = "unknown_op"} (%keys1d) : (tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>) loc(fused["Unique:", "uniq_node"])
    %gathered = "stablehlo.torch_index_select"(%table, %uniq#0) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<100x4xf32>, tensor<?xi32>) -> tensor<?x4xf32> loc(fused["GatherV2:", "gather_node"])
    %restored = "stablehlo.torch_index_select"(%gathered, %uniq#1) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32> loc(fused["GatherV2:", "restored"])
    return %restored : tensor<?x4xf32>
  }

  func.func @neg_dbluse(%table: tensor<100x4xf32>, %keys: tensor<?x1xi64>) -> tensor<?x4xf32> {
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
    %extra = "stablehlo.torch_index_select"(%table, %uniq#0) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<100x4xf32>, tensor<?xi64>) -> tensor<?x4xf32> loc(fused["GatherV2:", "extra"])
    %restored = "stablehlo.torch_index_select"(%gathered, %uniq#1) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32> loc(fused["GatherV2:", "restored"])
    %add = stablehlo.add %restored, %extra : tensor<?x4xf32>
    return %add : tensor<?x4xf32>
  }
}
