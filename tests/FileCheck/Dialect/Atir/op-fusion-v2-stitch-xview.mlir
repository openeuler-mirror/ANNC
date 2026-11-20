// RUN: rm -rf %t.dir && mkdir -p %t.dir && split-file %s %t.dir
// RUN: annc-asm %t.dir/positive.mlir -atir-op-fusion-v2 | FileCheck %t.dir/positive.mlir
// RUN: annc-asm %t.dir/anonymous.mlir -atir-op-fusion-v2 | FileCheck %t.dir/anonymous.mlir
// RUN: annc-asm %t.dir/i32x.mlir -atir-op-fusion-v2 | FileCheck %t.dir/i32x.mlir
// RUN: annc-asm %t.dir/zeroresult.mlir -atir-op-fusion-v2 | FileCheck %t.dir/zeroresult.mlir
//
// x-slot view adoption: the sink shortens an identity tf.Reshape into a
// get_dim_v2->set_dim_v2 view pair whose loc carries the pb node name;
// adopting the view keeps the ANNCFused input name byte-identical.
//
// Coverage: a named view pair moves the x boundary to set_dim_v2 with
// tf_name "model/x/Reshape"; an anonymous view keeps Unique (prefer
// missing over guessing - synthesized names the converter would reject).

//--- positive.mlir
module {
  func.func @main(%x: tensor<?xi64>, %v0: tensor<4x2xf32>,
                  %v1: tensor<4x2xf32>) -> tensor<?x2xf32>
      attributes {tf.entry_function = {control_outputs = "", inputs = "x:0,v0:0,v1:0", outputs = "output:0"}} {
    %c2 = stablehlo.constant dense<2> : tensor<i64> loc("mod_const")
    %zero = stablehlo.constant dense<0> : tensor<i64> loc("zero_const")
    %cshape = stablehlo.constant dense<[-1]> : tensor<1xi32> loc("ids_shape")
    %uniq:2 = atir.unsupported {opType = "Unique"} (%x) : (tensor<?xi64>) -> (tensor<?xi64>, tensor<?xi32>) loc("Unique")
    %flat, %gshp = "atir.get_dim_v2"(%uniq#0) {call_target_name = "annc.getDim"} : (tensor<?xi64>) -> (tensor<?xi64>, tensor<?xi64>) loc("getdim")
    %ids = "atir.set_dim_v2"(%flat, %cshape) {call_target_name = "annc.setDim"} : (tensor<?xi64>, tensor<1xi32>) -> tensor<?xi64> loc(fused["Reshape:", "model/x/Reshape"])
    %sh = shape.shape_of %ids : tensor<?xi64> -> tensor<1xindex> loc("mod_shape")
    %b2 = stablehlo.dynamic_broadcast_in_dim %c2, %sh, dims = [] : (tensor<i64>, tensor<1xindex>) -> tensor<?xi64> loc("mod_bcast")
    %b0 = stablehlo.dynamic_broadcast_in_dim %zero, %sh, dims = [] : (tensor<i64>, tensor<1xindex>) -> tensor<?xi64> loc("zero_bcast")
    %rem = stablehlo.remainder %ids, %b2 : tensor<?xi64> loc("FloorMod")
    %neg = stablehlo.compare LT, %rem, %b0 : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi1> loc("FloorMod")
    %adj = stablehlo.add %rem, %b2 : tensor<?xi64> loc("FloorMod")
    %mod = stablehlo.select %neg, %adj, %rem : tensor<?xi1>, tensor<?xi64> loc("FloorMod")
    %cast = stablehlo.convert %mod : (tensor<?xi64>) -> tensor<?xi32> loc("Cast")
    %floordiv = stablehlo.divide %ids, %b2 : tensor<?xi64> loc("floordiv")
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
// x boundary = the set_dim_v2 result, tf_name from the loc pb name.
// CHECK: args = [{dtype = "i64", rank = 1 : i64, role = "dynamic", shape = [-9223372036854775808], tf_name = "model/x/Reshape"}, {dtype = "f32", rank = 2 : i64, role = "fixed", shape = [4, 2], tf_name = "v0"}, {dtype = "f32", rank = 2 : i64, role = "fixed", shape = [4, 2], tf_name = "v1"}], context_i64_attrs = [2], dynamic_dims

//--- anonymous.mlir
module {
  func.func @main(%x: tensor<?xi64>, %v0: tensor<4x2xf32>,
                  %v1: tensor<4x2xf32>) -> tensor<?x2xf32>
      attributes {tf.entry_function = {control_outputs = "", inputs = "x:0,v0:0,v1:0", outputs = "output:0"}} {
    %c2 = stablehlo.constant dense<2> : tensor<i64> loc("mod_const")
    %zero = stablehlo.constant dense<0> : tensor<i64> loc("zero_const")
    %cshape = stablehlo.constant dense<[-1]> : tensor<1xi32> loc("ids_shape")
    %uniq:2 = atir.unsupported {opType = "Unique"} (%x) : (tensor<?xi64>) -> (tensor<?xi64>, tensor<?xi32>) loc("Unique")
    %flat, %gshp = "atir.get_dim_v2"(%uniq#0) {call_target_name = "annc.getDim"} : (tensor<?xi64>) -> (tensor<?xi64>, tensor<?xi64>) loc("getdim")
    %ids = "atir.set_dim_v2"(%flat, %cshape) {call_target_name = "annc.setDim"} : (tensor<?xi64>, tensor<1xi32>) -> tensor<?xi64> loc(unknown)
    %sh = shape.shape_of %ids : tensor<?xi64> -> tensor<1xindex> loc("mod_shape")
    %b2 = stablehlo.dynamic_broadcast_in_dim %c2, %sh, dims = [] : (tensor<i64>, tensor<1xindex>) -> tensor<?xi64> loc("mod_bcast")
    %b0 = stablehlo.dynamic_broadcast_in_dim %zero, %sh, dims = [] : (tensor<i64>, tensor<1xindex>) -> tensor<?xi64> loc("zero_bcast")
    %rem = stablehlo.remainder %ids, %b2 : tensor<?xi64> loc("FloorMod")
    %neg = stablehlo.compare LT, %rem, %b0 : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi1> loc("FloorMod")
    %adj = stablehlo.add %rem, %b2 : tensor<?xi64> loc("FloorMod")
    %mod = stablehlo.select %neg, %adj, %rem : tensor<?xi1>, tensor<?xi64> loc("FloorMod")
    %cast = stablehlo.convert %mod : (tensor<?xi64>) -> tensor<?xi32> loc("Cast")
    %floordiv = stablehlo.divide %ids, %b2 : tensor<?xi64> loc("floordiv")
    %range = stablehlo.abs %cast : tensor<?xi32> loc("range")
    %p:2 = atir.unsupported {opType = "DynamicPartition"} (%range, %cast) : (tensor<?xi32>, tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>) loc("DynamicPartition")
    %q:2 = atir.unsupported {opType = "DynamicPartition"} (%floordiv, %cast) : (tensor<?xi64>, tensor<?xi32>) -> (tensor<?xi64>, tensor<?xi64>) loc("DynamicPartition_1")
    %g0 = "stablehlo.torch_index_select"(%v0, %q#0) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<4x2xf32>, tensor<?xi64>) -> tensor<?x2xf32> loc("GatherV2")
    %g1 = "stablehlo.torch_index_select"(%v1, %q#1) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<4x2xf32>, tensor<?xi64>) -> tensor<?x2xf32> loc("GatherV2_1")
    %out = atir.unsupported {opType = "ParallelDynamicStitch"} (%p#0, %p#1, %g0, %g1) : (tensor<?xi32>, tensor<?xi32>, tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?x2xf32> loc("output")
    return %out : tensor<?x2xf32>
  }
}

// Anonymous view loc: no pb name to resolve, so the Unique boundary stays.
// CHECK: call @KPFusedSparseDynamicStitch
// CHECK: args = [{dtype = "i64", rank = 1 : i64, role = "dynamic", shape = [-9223372036854775808], tf_name = "Unique"}, {dtype = "f32", rank = 2 : i64, role = "fixed", shape = [4, 2], tf_name = "v0"}, {dtype = "f32", rank = 2 : i64, role = "fixed", shape = [4, 2], tf_name = "v1"}], context_i64_attrs = [2], dynamic_dims

//--- i32x.mlir
// Negative: an i32 x violates the kernel dtype (i64); no kernel is produced.
// CHECK-NOT: annc.kernel
module {
  func.func @main(%x: tensor<4xi32>, %v0: tensor<4x2xf32>, %v1: tensor<4x2xf32>) -> tensor<?x2xf32> {
    %c2 = stablehlo.constant dense<2> : tensor<1xi64>
    %c0 = stablehlo.constant dense<0> : tensor<1xi64>
    %pidx = arith.constant dense<1> : tensor<4xi32>
    %b0:2 = atir.unsupported {detail = "", metadata = {}, opType = "DynamicPartition", reason = "unknown_op"} (%x, %pidx) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>)
    %b1:2 = atir.unsupported {detail = "", metadata = {}, opType = "DynamicPartition", reason = "unknown_op"} (%x, %pidx) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>)
    %u:2 = atir.unsupported {detail = "", metadata = {}, opType = "Unique", reason = "unknown_op"} (%x) : (tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>)
    %s0 = "stablehlo.torch_index_select"(%v0, %b1#0) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<4x2xf32>, tensor<4xi32>) -> tensor<4x2xf32>
    %s1 = "stablehlo.torch_index_select"(%v1, %b1#1) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<4x2xf32>, tensor<4xi32>) -> tensor<4x2xf32>
    %st = atir.unsupported {detail = "", metadata = {}, opType = "ParallelDynamicStitch", reason = "unknown_op"} (%b0#0, %b0#1, %s0, %s1) : (tensor<4xi32>, tensor<4xi32>, tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<?x2xf32>
    return %st : tensor<?x2xf32>
  }
}

//--- zeroresult.mlir
// Negative: a zero-result ParallelDynamicStitch anchor is legal IR but not
// a fusible site; the pass must reject it, not abort.
// CHECK-NOT: annc.kernel
module {
  func.func @main(%x: tensor<4xi64>) {
    %pidx = arith.constant dense<1> : tensor<4xi32>
    atir.unsupported {detail = "", metadata = {}, opType = "ParallelDynamicStitch", reason = "unknown_op"} (%x, %x, %x, %x) : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> ()
    return
  }
}
