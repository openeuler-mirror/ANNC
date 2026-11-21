// RUN: rm -rf %t.dir && mkdir -p %t.dir && split-file %s %t.dir
// RUN: annc-asm %t.dir/positive.mlir -atir-op-fusion-v2 | FileCheck %t.dir/positive.mlir
// RUN: annc-asm %t.dir/negative.mlir -atir-op-fusion-v2 | FileCheck %t.dir/negative.mlir
//
// Shared-partition pair form (bst_mmoe measured): two embedding groups
// share one index partition; each ParallelDynamicStitch consumes two
// table slices.
//
// Match requirements: the shared position DP stays put this round
// (posShared), each data operand is gather(table_k, rowNumDP bucket k)
// over a shared partitions value, and x traces back to Unique:0.
// Negative: a rank-3 table gets rejected by the kernel contract.

//--- positive.mlir
module {
  func.func @main(%x: tensor<?xi64>, %t0: tensor<4x2xf32>, %t1: tensor<4x2xf32>,
                  %t2: tensor<4x2xf32>, %t3: tensor<4x2xf32>) -> tensor<?x2xf32>
      attributes {tf.arg_names = ["x", "t0", "t1", "t2", "t3"], tf.entry_function = {control_outputs = "", inputs = "x:0,t0:0,t1:0,t2:0,t3:0", outputs = "sum:0"}} {
    %c2 = stablehlo.constant dense<2> : tensor<i64> loc("mod_const")
    %uniq:2 = atir.unsupported {opType = "Unique"} (%x) : (tensor<?xi64>) -> (tensor<?xi64>, tensor<?xi32>) loc("Unique")
    %sh = shape.shape_of %uniq#0 : tensor<?xi64> -> tensor<1xindex> loc("mod_shape")
    %b2 = stablehlo.dynamic_broadcast_in_dim %c2, %sh, dims = [] : (tensor<i64>, tensor<1xindex>) -> tensor<?xi64> loc("mod_bcast")
    %rem = stablehlo.remainder %uniq#0, %b2 : tensor<?xi64> loc("FloorMod")
    %pidx = stablehlo.convert %rem : (tensor<?xi64>) -> tensor<?xi32> loc("Cast")
    %range = stablehlo.abs %pidx : tensor<?xi32> loc("range")
    %fdiv = stablehlo.divide %uniq#0, %b2 : tensor<?xi64> loc("floordiv")
    %p:2 = atir.unsupported {opType = "DynamicPartition"} (%range, %pidx) : (tensor<?xi32>, tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>) loc("DynamicPartition")
    %q:2 = atir.unsupported {opType = "DynamicPartition"} (%fdiv, %pidx) : (tensor<?xi64>, tensor<?xi32>) -> (tensor<?xi64>, tensor<?xi64>) loc("DynamicPartition_1")
    %r:2 = atir.unsupported {opType = "DynamicPartition"} (%fdiv, %pidx) : (tensor<?xi64>, tensor<?xi32>) -> (tensor<?xi64>, tensor<?xi64>) loc("DynamicPartition_2")
    %g0 = "stablehlo.torch_index_select"(%t0, %q#0) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<4x2xf32>, tensor<?xi64>) -> tensor<?x2xf32> loc("GatherV2")
    %g1 = "stablehlo.torch_index_select"(%t1, %q#1) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<4x2xf32>, tensor<?xi64>) -> tensor<?x2xf32> loc("GatherV2_1")
    %h0 = "stablehlo.torch_index_select"(%t2, %r#0) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<4x2xf32>, tensor<?xi64>) -> tensor<?x2xf32> loc("GatherV2_2")
    %h1 = "stablehlo.torch_index_select"(%t3, %r#1) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<4x2xf32>, tensor<?xi64>) -> tensor<?x2xf32> loc("GatherV2_3")
    %s1 = atir.unsupported {opType = "ParallelDynamicStitch"} (%p#0, %p#1, %g0, %g1) : (tensor<?xi32>, tensor<?xi32>, tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?x2xf32> loc("ParallelDynamicStitch")
    %s2 = atir.unsupported {opType = "ParallelDynamicStitch"} (%p#0, %p#1, %h0, %h1) : (tensor<?xi32>, tensor<?xi32>, tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?x2xf32> loc("ParallelDynamicStitch_1")
    %sum = stablehlo.add %s1, %s2 : tensor<?x2xf32> loc("Add")
    return %sum : tensor<?x2xf32>
  }
}

// After both rounds no partition stays in main (NOT placed first to
// cover everything before the first CHECK).
// CHECK-NOT: opType = "ParallelDynamicStitch"
// CHECK-NOT: opType = "DynamicPartition"
// Two sites yield N2 and N2_1 (suffix keeps names unique); the site set
// is checked as a DAG since greedy driver order is not asserted.
// CHECK-DAG: call @KPFusedSparseDynamicStitch(
// CHECK-DAG: call @KPFusedSparseDynamicStitch_1(
// CHECK-DAG: func.func private @KPFusedSparseDynamicStitch(
// CHECK-DAG: func.func private @KPFusedSparseDynamicStitch_1(
// CHECK-DAG: fusion.pattern = "kp_sparse_dynamic_stitch"
// CHECK-DAG: tf_name = "t0"
// CHECK-DAG: tf_name = "t3"

//--- negative.mlir
module {
  func.func @main(%x: tensor<?xi64>, %t0: tensor<4x2xf32>, %t1: tensor<4x2x2xf32>) -> tensor<?x2x2xf32>
      attributes {tf.arg_names = ["x", "t0", "t1"], tf.entry_function = {control_outputs = "", inputs = "x:0,t0:0,t1:0", outputs = "out:0"}} {
    %c2 = stablehlo.constant dense<2> : tensor<i64> loc("mod_const")
    %uniq:2 = atir.unsupported {opType = "Unique"} (%x) : (tensor<?xi64>) -> (tensor<?xi64>, tensor<?xi32>) loc("Unique")
    %sh = shape.shape_of %uniq#0 : tensor<?xi64> -> tensor<1xindex> loc("mod_shape")
    %b2 = stablehlo.dynamic_broadcast_in_dim %c2, %sh, dims = [] : (tensor<i64>, tensor<1xindex>) -> tensor<?xi64> loc("mod_bcast")
    %rem = stablehlo.remainder %uniq#0, %b2 : tensor<?xi64> loc("FloorMod")
    %pidx = stablehlo.convert %rem : (tensor<?xi64>) -> tensor<?xi32> loc("Cast")
    %range = stablehlo.abs %pidx : tensor<?xi32> loc("range")
    %fdiv = stablehlo.divide %uniq#0, %b2 : tensor<?xi64> loc("floordiv")
    %p:2 = atir.unsupported {opType = "DynamicPartition"} (%range, %pidx) : (tensor<?xi32>, tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>) loc("DynamicPartition")
    %q:2 = atir.unsupported {opType = "DynamicPartition"} (%fdiv, %pidx) : (tensor<?xi64>, tensor<?xi32>) -> (tensor<?xi64>, tensor<?xi64>) loc("DynamicPartition_1")
    %g0 = "stablehlo.torch_index_select"(%t0, %q#0) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<4x2xf32>, tensor<?xi64>) -> tensor<?x2xf32> loc("GatherV2")
    %g1 = "stablehlo.torch_index_select"(%t1, %q#1) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<4x2x2xf32>, tensor<?xi64>) -> tensor<?x2x2xf32> loc("GatherV2_1")
    %out = atir.unsupported {opType = "ParallelDynamicStitch"} (%p#0, %p#1, %g0, %g1) : (tensor<?xi32>, tensor<?xi32>, tensor<?x2xf32>, tensor<?x2x2xf32>) -> tensor<?x2x2xf32> loc("ParallelDynamicStitch")
    return %out : tensor<?x2x2xf32>
  }
}

// A rank-3 table violates the kernel contract; both sites stay unmatched.
// CHECK: opType = "ParallelDynamicStitch"
// CHECK-NOT: KPFusedSparseDynamicStitch
