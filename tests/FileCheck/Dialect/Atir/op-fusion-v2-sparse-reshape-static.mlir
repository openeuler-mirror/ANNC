// RUN: annc-asm %s -atir-op-fusion-v2 | FileCheck %s
//
//
#loc = loc(unknown)
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main(%arg0: tensor<4x2xi64> loc(unknown)) -> tensor<4x2xi64> attributes {tf.arg_names = ["keys"], tf.entry_function = {control_outputs = "", inputs = "keys:0", outputs = "sr:0"}} {
    %c = stablehlo.constant dense<[-1, 2]> : tensor<2xi64> loc(#loc1)
    %c_0 = stablehlo.constant {annc.source_node = "in_shape", annc.source_op = "tf.Cast", annc.source_ports = dense<0> : tensor<1xi32>} dense<[4, 2]> : tensor<2xi64> loc(#loc10)
    %c_1 = stablehlo.constant {annc.source_node = "rng_c", annc.source_op = "tf.Cast", annc.source_ports = dense<0> : tensor<1xi32>} dense<[[0], [1], [2], [3]]> : tensor<4x1xi64> loc(#loc11)
    %0 = stablehlo.slice %arg0 [0:4, 0:1] {annc.begin_source = "begin", annc.source_node = "seg", annc.source_op = "tf.StridedSlice"} : (tensor<4x2xi64>) -> tensor<4x1xi64> loc(#loc12)
    %1 = stablehlo.concatenate %c_1, %0, dim = 1 {annc.const_source = "axisc", annc.source_node = "indices", annc.source_op = "tf.ConcatV2", annc.source_ports = dense<0> : tensor<1xi32>} : (tensor<4x1xi64>, tensor<4x1xi64>) -> tensor<4x2xi64> loc(#loc13)
    %outputIndices, %outputShape = "atir.sparse_reshape_v2"(%1, %c_0, %c) {T = i64, annc.source_node = "sr", annc.source_op = "tf.SparseReshape", call_target_name = "tf.SparseReshape", device = ""} : (tensor<4x2xi64>, tensor<2xi64>, tensor<2xi64>) -> (tensor<4x2xi64>, tensor<2xi64>) loc(#loc14)
    return %outputIndices : tensor<4x2xi64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("sr")
#loc2 = loc("Cast:")
#loc3 = loc("in_shape")
#loc4 = loc("rng_c")
#loc5 = loc("StridedSlice:")
#loc6 = loc("seg")
#loc7 = loc("ConcatV2:")
#loc8 = loc("indices")
#loc9 = loc("SparseReshape:")
#loc10 = loc(fused[#loc2, #loc3])
#loc11 = loc(fused[#loc2, #loc4])
#loc12 = loc(fused[#loc5, #loc6])
#loc13 = loc(fused[#loc7, #loc8])
#loc14 = loc(fused[#loc9, #loc1])

// CHECK: atir.sparse_reshape_v2
// CHECK-NOT: KPFusedSparseReshape
