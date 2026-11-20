// RUN: annc-asm %s -atir-op-fusion-v2 | FileCheck %s
// kp_sparse_dynamic_stitch residue-carrier form: the stitch anchor and both
// partitions stay atir.unsupported on dynamic shapes; the arithmetic chain
// folds into the site with x = Unique:0 and tables v0/v1 (N=2).
module {
  func.func @main(%x: tensor<?xi64>, %v0: tensor<4x2xf32>,
                  %v1: tensor<4x2xf32>) -> tensor<?x2xf32>
      attributes {tf.entry_function = {control_outputs = "", inputs = "x:0,v0:0,v1:0", outputs = "output:0"}} {
    %c2 = stablehlo.constant dense<2> : tensor<i64> loc("mod_const")
    %zero = stablehlo.constant dense<0> : tensor<i64> loc("zero_const")
    %uniq:2 = atir.unsupported {opType = "Unique"} (%x) : (tensor<?xi64>) -> (tensor<?xi64>, tensor<?xi32>) loc("Unique")
    %sh = shape.shape_of %uniq#0 : tensor<?xi64> -> tensor<1xindex> loc("mod_shape")
    %b2 = stablehlo.dynamic_broadcast_in_dim %c2, %sh, dims = [] : (tensor<i64>, tensor<1xindex>) -> tensor<?xi64> loc("mod_bcast")
    %b0 = stablehlo.dynamic_broadcast_in_dim %zero, %sh, dims = [] : (tensor<i64>, tensor<1xindex>) -> tensor<?xi64> loc("zero_bcast")
    %rem = stablehlo.remainder %uniq#0, %b2 : tensor<?xi64> loc("FloorMod")
    %neg = stablehlo.compare LT, %rem, %b0 : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi1> loc("FloorMod")
    %adj = stablehlo.add %rem, %b2 : tensor<?xi64> loc("FloorMod")
    %mod = stablehlo.select %neg, %adj, %rem : tensor<?xi1>, tensor<?xi64> loc("FloorMod")
    %cast = stablehlo.convert %mod : (tensor<?xi64>) -> tensor<?xi32> loc("Cast")
    %floordiv = stablehlo.divide %uniq#0, %b2 : tensor<?xi64> loc("floordiv")
    %range = stablehlo.abs %cast : tensor<?xi32> loc("range")
    %p:2 = atir.unsupported {opType = "DynamicPartition"} (%range, %cast) : (tensor<?xi32>, tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>) loc("DynamicPartition")
    %q:2 = atir.unsupported {opType = "DynamicPartition"} (%floordiv, %cast) : (tensor<?xi64>, tensor<?xi32>) -> (tensor<?xi64>, tensor<?xi64>) loc("DynamicPartition_1")
    %g0 = "stablehlo.torch_index_select"(%v0, %q#0) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<4x2xf32>, tensor<?xi64>) -> tensor<?x2xf32> loc("GatherV2")
    %g1 = "stablehlo.torch_index_select"(%v1, %q#1) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<4x2xf32>, tensor<?xi64>) -> tensor<?x2xf32> loc("GatherV2_1")
    %out = atir.unsupported {opType = "ParallelDynamicStitch"} (%p#0, %p#1, %g0, %g1) : (tensor<?xi32>, tensor<?xi32>, tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?x2xf32> loc("output")
    return %out : tensor<?x2xf32>
  }
}

// CHECK-NOT: opType = "DynamicPartition"
// CHECK-NOT: opType = "ParallelDynamicStitch"
// CHECK: call @KPFusedSparseDynamicStitch
// Contract is pinned exactly: closure constants must stay out of args.
// CHECK: annc.kernel
// CHECK-SAME: abi = "annc_context_v3" 
// CHECK: args = [{dtype = "i64", rank = 1 : i64, role = "dynamic", shape = [-9223372036854775808], tf_name = "Unique"}, {dtype = "f32", rank = 2 : i64, role = "fixed", shape = [4, 2], tf_name = "v0"}, {dtype = "f32", rank = 2 : i64, role = "fixed", shape = [4, 2], tf_name = "v1"}], context_i64_attrs = [2], dynamic_dims
// CHECK-SAME: fusion.pattern = "kp_sparse_dynamic_stitch"
// CHECK-SAME: kernel_name = "KPFusedSparseDynamicStitch"
// CHECK-SAME: tf_name = "output"
// CHECK-SAME: tf.name = "KPFusedSparseDynamicStitch"
// The synthesized site body must keep the x chain rank-1 dynamic (a (1,)
// splat operand would collapse it to one static row).
// CHECK: stablehlo.remainder %{{.*}} : tensor<?xi64>
// CHECK: stablehlo.divide %{{.*}} : tensor<?xi64>
