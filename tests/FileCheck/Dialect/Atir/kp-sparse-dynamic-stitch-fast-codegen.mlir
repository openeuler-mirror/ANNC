// RUN: annc-asm --atir-fast-codegen %s | FileCheck %s
// KP-style sparse dynamic stitch fusion, second level: inside the kernel
// func extracted by OpFusion (args = x, v0, v1, output), the
// Range/Size/Cast/FloorMod/FloorDiv/DynamicPartition/Gather subgraph folds
// into a single KPFusedSparseDynamicStitch CustomizeOp (N=2).

// CHECK-LABEL: func.func private @fused_kp_sparse_dynamic_stitch
// CHECK: atir.Customize
// CHECK-SAME: custom.op_name = "KPFusedSparseDynamicStitchN2"
// CHECK-NOT: atir.ParallelDynamicStitch
// CHECK-NOT: atir.Gather
// CHECK-NOT: atir.DynamicPartition
module attributes {module.state = "atir"} {
  func.func private @fused_kp_sparse_dynamic_stitch(
      %arg0: !atir.tensor<4xi64, name = "x">,
      %arg1: !atir.tensor<2x3xf32, name = "v0">,
      %arg2: !atir.tensor<2x3xf32, name = "v1">,
      %arg3: !atir.tensor<4x3xf32, name = "output">) attributes {annc.kernel} {
    %0 = atir.Size %arg0 : (!atir.tensor<4xi64, name = "x">) -> !atir.tensor<i32, name = "size">
    %1 = "atir.buffer"() : () -> !atir.tensor<4xi32, name = "range_out">
    %2 = atir.constant "public" @"range_start" -> <i32, name = "range_start", data = dense<0> : tensor<i32>>
    %3 = atir.constant "public" @"range_delta" -> <i32, name = "range_delta", data = dense<1> : tensor<i32>>
    %4 = "atir.Range"(%1, %2, %0, %3) : (!atir.tensor<4xi32, name = "range_out">, !atir.tensor<i32, name = "range_start", data = dense<0> : tensor<i32>>, !atir.tensor<i32, name = "size">, !atir.tensor<i32, name = "range_delta", data = dense<1> : tensor<i32>>) -> !atir.tensor<4xi32, name = "range_out">
    %5 = "atir.buffer"() : () -> !atir.tensor<4xi64, name = "mod_out">
    %6 = atir.constant "public" @"n_const" -> <i64, name = "n_const", data = dense<2> : tensor<i64>>
    %7 = "atir.FloorMod"(%5, %arg0, %6) : (!atir.tensor<4xi64, name = "mod_out">, !atir.tensor<4xi64, name = "x">, !atir.tensor<i64, name = "n_const", data = dense<2> : tensor<i64>>) -> !atir.tensor<4xi64, name = "mod_out">
    %8 = "atir.buffer"() : () -> !atir.tensor<4xi32, name = "cast_out">
    %9 = "atir.Cast"(%8, %7) : (!atir.tensor<4xi32, name = "cast_out">, !atir.tensor<4xi64, name = "mod_out">) -> !atir.tensor<4xi32, name = "cast_out">
    %10, %11 = "atir.DynamicPartition"(%4, %9) <{numPartitions = 2 : i32}> : (!atir.tensor<4xi32, name = "range_out">, !atir.tensor<4xi32, name = "cast_out">) -> (!atir.tensor<?xi32, name = "idx0">, !atir.tensor<?xi32, name = "idx1">)
    %12 = "atir.buffer"() : () -> !atir.tensor<4xi64, name = "div_out">
    %13 = "atir.FloorDiv"(%12, %arg0, %6) : (!atir.tensor<4xi64, name = "div_out">, !atir.tensor<4xi64, name = "x">, !atir.tensor<i64, name = "n_const", data = dense<2> : tensor<i64>>) -> !atir.tensor<4xi64, name = "div_out">
    %14, %15 = "atir.DynamicPartition"(%13, %9) <{numPartitions = 2 : i32}> : (!atir.tensor<4xi64, name = "div_out">, !atir.tensor<4xi32, name = "cast_out">) -> (!atir.tensor<?xi64, name = "didx0">, !atir.tensor<?xi64, name = "didx1">)
    %16 = "atir.buffer"() : () -> !atir.tensor<?x3xf32, name = "g0_out">
    %17 = atir.constant "public" @"gather_axis" -> <i32, name = "gather_axis", data = dense<0> : tensor<i32>>
    %18 = "atir.Gather"(%16, %arg1, %14, %17) : (!atir.tensor<?x3xf32, name = "g0_out">, !atir.tensor<2x3xf32, name = "v0">, !atir.tensor<?xi64, name = "didx0">, !atir.tensor<i32, name = "gather_axis", data = dense<0> : tensor<i32>>) -> !atir.tensor<?x3xf32, name = "g0_out">
    %19 = "atir.buffer"() : () -> !atir.tensor<?x3xf32, name = "g1_out">
    %20 = "atir.Gather"(%19, %arg2, %15, %17) : (!atir.tensor<?x3xf32, name = "g1_out">, !atir.tensor<2x3xf32, name = "v1">, !atir.tensor<?xi64, name = "didx1">, !atir.tensor<i32, name = "gather_axis", data = dense<0> : tensor<i32>>) -> !atir.tensor<?x3xf32, name = "g1_out">
    %21 = "atir.ParallelDynamicStitch"(%arg3, %10, %11, %18, %20) : (!atir.tensor<4x3xf32, name = "output">, !atir.tensor<?xi32, name = "idx0">, !atir.tensor<?xi32, name = "idx1">, !atir.tensor<?x3xf32, name = "g0_out">, !atir.tensor<?x3xf32, name = "g1_out">) -> !atir.tensor<4x3xf32, name = "output">
    return
  }
}

// -----

// Negative case: right-branch gather uses axis=1 — reject.
// CHECK-LABEL: func.func private @bad_axis
// CHECK-NOT: atir.Customize
// CHECK: atir.ParallelDynamicStitch
module attributes {module.state = "atir"} {
  func.func private @bad_axis(
      %arg0: !atir.tensor<4xi64, name = "x_t">,
      %arg1: !atir.tensor<2x3xf32, name = "v0_t">,
      %arg2: !atir.tensor<2x3xf32, name = "v1_t">,
      %arg3: !atir.tensor<4x3xf32, name = "output_t">) attributes {annc.kernel} {
    %0 = atir.Size %arg0 : (!atir.tensor<4xi64, name = "x_t">) -> !atir.tensor<i32, name = "size_t">
    %1 = "atir.buffer"() : () -> !atir.tensor<4xi32, name = "range_out_t">
    %2 = atir.constant "public" @"range_start_t" -> <i32, name = "range_start_t", data = dense<0> : tensor<i32>>
    %3 = atir.constant "public" @"range_delta_t" -> <i32, name = "range_delta_t", data = dense<1> : tensor<i32>>
    %4 = "atir.Range"(%1, %2, %0, %3) : (!atir.tensor<4xi32, name = "range_out_t">, !atir.tensor<i32, name = "range_start_t", data = dense<0> : tensor<i32>>, !atir.tensor<i32, name = "size_t">, !atir.tensor<i32, name = "range_delta_t", data = dense<1> : tensor<i32>>) -> !atir.tensor<4xi32, name = "range_out_t">
    %5 = "atir.buffer"() : () -> !atir.tensor<4xi64, name = "mod_out_t">
    %6 = atir.constant "public" @"n_const_t" -> <i64, name = "n_const_t", data = dense<2> : tensor<i64>>
    %7 = "atir.FloorMod"(%5, %arg0, %6) : (!atir.tensor<4xi64, name = "mod_out_t">, !atir.tensor<4xi64, name = "x_t">, !atir.tensor<i64, name = "n_const_t", data = dense<2> : tensor<i64>>) -> !atir.tensor<4xi64, name = "mod_out_t">
    %8 = "atir.buffer"() : () -> !atir.tensor<4xi32, name = "cast_out_t">
    %9 = "atir.Cast"(%8, %7) : (!atir.tensor<4xi32, name = "cast_out_t">, !atir.tensor<4xi64, name = "mod_out_t">) -> !atir.tensor<4xi32, name = "cast_out_t">
    %10, %11 = "atir.DynamicPartition"(%4, %9) <{numPartitions = 2 : i32}> : (!atir.tensor<4xi32, name = "range_out_t">, !atir.tensor<4xi32, name = "cast_out_t">) -> (!atir.tensor<?xi32, name = "idx0_t">, !atir.tensor<?xi32, name = "idx1_t">)
    %12 = "atir.buffer"() : () -> !atir.tensor<4xi64, name = "div_out_t">
    %13 = "atir.FloorDiv"(%12, %arg0, %6) : (!atir.tensor<4xi64, name = "div_out_t">, !atir.tensor<4xi64, name = "x_t">, !atir.tensor<i64, name = "n_const_t", data = dense<2> : tensor<i64>>) -> !atir.tensor<4xi64, name = "div_out_t">
    %14, %15 = "atir.DynamicPartition"(%13, %9) <{numPartitions = 2 : i32}> : (!atir.tensor<4xi64, name = "div_out_t">, !atir.tensor<4xi32, name = "cast_out_t">) -> (!atir.tensor<?xi64, name = "didx0_t">, !atir.tensor<?xi64, name = "didx1_t">)
    %16 = "atir.buffer"() : () -> !atir.tensor<?x3xf32, name = "g0_out_t">
    %17 = atir.constant "public" @"gather_axis_bad" -> <i32, name = "gather_axis_bad", data = dense<1> : tensor<i32>>
    %18 = "atir.Gather"(%16, %arg1, %14, %17) : (!atir.tensor<?x3xf32, name = "g0_out_t">, !atir.tensor<2x3xf32, name = "v0_t">, !atir.tensor<?xi64, name = "didx0_t">, !atir.tensor<i32, name = "gather_axis_bad", data = dense<1> : tensor<i32>>) -> !atir.tensor<?x3xf32, name = "g0_out_t">
    %19 = "atir.buffer"() : () -> !atir.tensor<?x3xf32, name = "g1_out_t">
    %20 = "atir.Gather"(%19, %arg2, %15, %17) : (!atir.tensor<?x3xf32, name = "g1_out_t">, !atir.tensor<2x3xf32, name = "v1_t">, !atir.tensor<?xi64, name = "didx1_t">, !atir.tensor<i32, name = "gather_axis_bad", data = dense<1> : tensor<i32>>) -> !atir.tensor<?x3xf32, name = "g1_out_t">
    %21 = "atir.ParallelDynamicStitch"(%arg3, %10, %11, %18, %20) : (!atir.tensor<4x3xf32, name = "output_t">, !atir.tensor<?xi32, name = "idx0_t">, !atir.tensor<?xi32, name = "idx1_t">, !atir.tensor<?x3xf32, name = "g0_out_t">, !atir.tensor<?x3xf32, name = "g1_out_t">) -> !atir.tensor<4x3xf32, name = "output_t">
    return
  }
}
