// RUN: annc-opt --split-input-file %s -atir-config-fusion="config=%S/../../../../configs/fusion/kp_embedding_patterns.json run-builtin-after=false warn-limit=0" | FileCheck %s --check-prefix=CONFIG
// Config fusion first level for sparse dynamic stitch: the subgraph in
// main is extracted into a private kernel func with annc.kernel +
// fusion.metadata while preserving the legacy output-buffer ABI.

// CONFIG: func.func private @fused_kp_sparse_dynamic_stitch_n2_
// CONFIG-SAME: annc.kernel
// CONFIG-SAME: fusion.pattern = "kp_sparse_dynamic_stitch"
// CONFIG: func.func @main
// CONFIG: atir.ParallelDynamicStitch
// CONFIG-SAME: annc.fusion_materialized

// CHECK-LABEL: func.func @main
// CHECK: atir.ParallelDynamicStitch
// CHECK-SAME: annc.fusion_materialized
// CHECK: func.func private @fused_kp_sparse_dynamic_stitch_
// CHECK-SAME: annc.kernel
// CHECK-SAME: fusion.metadata
// CHECK-SAME: role = "dynamic"
// CHECK-SAME: role = "fixed"
// CHECK-SAME: kernel_name = "fused_kp_sparse_dynamic_stitch_
// CHECK-SAME: llvm.emit_c_interface
module attributes {module.state = "atir"} {
func.func @main(
    %arg0: !atir.tensor<4xi64, name = "x">,
    %arg1: !atir.tensor<2x3xf32, name = "v0">,
    %arg2: !atir.tensor<2x3xf32, name = "v1">)
    -> !atir.tensor<4x3xf32, name = "output"> {
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
    %22 = "atir.buffer"() : () -> !atir.tensor<4x3xf32, name = "output">
    %23 = "atir.ParallelDynamicStitch"(%22, %10, %11, %18, %20) : (!atir.tensor<4x3xf32, name = "output">, !atir.tensor<?xi32, name = "idx0">, !atir.tensor<?xi32, name = "idx1">, !atir.tensor<?x3xf32, name = "g0_out">, !atir.tensor<?x3xf32, name = "g1_out">) -> !atir.tensor<4x3xf32, name = "output">
    return %23 : !atir.tensor<4x3xf32, name = "output">
  }
}
