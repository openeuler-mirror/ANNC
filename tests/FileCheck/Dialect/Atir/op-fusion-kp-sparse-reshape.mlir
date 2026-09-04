// RUN: annc-opt --split-input-file %s -atir-config-fusion="config=%S/../../../../configs/fusion/kp_embedding_patterns.json run-builtin-after=false warn-limit=0" | FileCheck %s --check-prefix=CONFIG
// Config fusion first level for sparse reshape (KPFusedSparseReshape),
// Execution V2: kernel func signature (!llvm.ptr, keys, begin, pack_const,
// new_shape) -> (out_indices, out_shape).

// CONFIG: func.func private @fused_kp_sparse_reshape_
// CONFIG-SAME: %arg0: !llvm.ptr
// CONFIG-SAME: annc.kernel
// CONFIG-SAME: abi = "annc_execution_v2"
// CONFIG: func.func @main
// CONFIG: atir.SparseReshape
// CONFIG-SAME: annc.fusion_materialized

// CHECK-LABEL: func.func @main
// CHECK: atir.SparseReshape
// CHECK-SAME: annc.fusion_materialized
// CHECK-LABEL: func.func private @fused_kp_sparse_reshape_
// CHECK-SAME: %arg0: !llvm.ptr,
// CHECK-SAME: %arg1: !atir.tensor<4x3xi64, name = "keys">, %arg2: !atir.tensor<2xi32, name = "begin">, %arg3: !atir.tensor<i32, name = "pack_const">, %arg4: !atir.tensor<2xi64, name = "new_shape">
// CHECK-SAME: annc.kernel
// CHECK-SAME: abi = "annc_execution_v2"
// CHECK-SAME: kernel_name = "fused_kp_sparse_reshape_
module attributes {module.state = "atir"} {
  func.func @main(
      %arg0: !atir.tensor<4x3xi64, name = "keys">,
      %arg1: !atir.tensor<2xi32, name = "begin">,
      %arg2: !atir.tensor<2xi64, name = "new_shape">,
      %arg3: !atir.tensor<i32, name = "pack_const">)
      -> (!atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">) {
  %0 = atir.Shape %arg0 : (!atir.tensor<4x3xi64, name = "keys">) -> !atir.tensor<2xi32, name = "shape_out">
  %1 = "atir.buffer"() : () -> !atir.tensor<i32, name = "ss1_out">
  %2 = atir.constant "public" @"ss1_begin" -> <1xi32, name = "ss1_begin", data = dense<0> : tensor<1xi32>>
  %3 = atir.constant "public" @"ss1_end" -> <1xi32, name = "ss1_end", data = dense<1> : tensor<1xi32>>
  %4 = atir.constant "public" @"ss1_strides" -> <1xi32, name = "ss1_strides", data = dense<1> : tensor<1xi32>>
  %5 = "atir.StridedSlice"(%1, %0, %2, %3, %4) <{beginMask = 0 : i32, endMask = 0 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 1 : i32}> : (!atir.tensor<i32, name = "ss1_out">, !atir.tensor<2xi32, name = "shape_out">, !atir.tensor<1xi32, name = "ss1_begin", data = dense<0> : tensor<1xi32>>, !atir.tensor<1xi32, name = "ss1_end", data = dense<1> : tensor<1xi32>>, !atir.tensor<1xi32, name = "ss1_strides", data = dense<1> : tensor<1xi32>>) -> !atir.tensor<i32, name = "ss1_out">
  %6 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "pack_out">
  %7 = "atir.Pack"(%6, %5, %arg3) <{axis = 0 : i64}> : (!atir.tensor<2xi32, name = "pack_out">, !atir.tensor<i32, name = "ss1_out">, !atir.tensor<i32, name = "pack_const">) -> !atir.tensor<2xi32, name = "pack_out">
  %8 = "atir.buffer"() : () -> !atir.tensor<2xi64, name = "cast_out">
  %9 = "atir.Cast"(%8, %7) : (!atir.tensor<2xi64, name = "cast_out">, !atir.tensor<2xi32, name = "pack_out">) -> !atir.tensor<2xi64, name = "cast_out">
  %10 = "atir.buffer"() : () -> !atir.tensor<i64, name = "ss_out">
  %11 = atir.constant "public" @"ss_end" -> <2xi32, name = "ss_end", data = dense<[0, 0]> : tensor<2xi32>>
  %12 = atir.constant "public" @"ss_strides" -> <2xi32, name = "ss_strides", data = dense<[1, 1]> : tensor<2xi32>>
  %13 = "atir.StridedSlice"(%10, %arg0, %arg1, %11, %12) <{beginMask = 1 : i32, endMask = 1 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 2 : i32}> : (!atir.tensor<i64, name = "ss_out">, !atir.tensor<4x3xi64, name = "keys">, !atir.tensor<2xi32, name = "begin">, !atir.tensor<2xi32, name = "ss_end", data = dense<[0, 0]> : tensor<2xi32>>, !atir.tensor<2xi32, name = "ss_strides", data = dense<[1, 1]> : tensor<2xi32>>) -> !atir.tensor<i64, name = "ss_out">
  %14 = "atir.buffer"() : () -> !atir.tensor<?x1xi64, name = "reshape_out">
  %15 = atir.constant "public" @"reshape_shape" -> <2xi32, name = "reshape_shape", data = dense<[-1, 1]> : tensor<2xi32>>
  %16 = "atir.Reshape"(%14, %13, %15) : (!atir.tensor<?x1xi64, name = "reshape_out">, !atir.tensor<i64, name = "ss_out">, !atir.tensor<2xi32, name = "reshape_shape", data = dense<[-1, 1]> : tensor<2xi32>>) -> !atir.tensor<?x1xi64, name = "reshape_out">
  %17 = "atir.buffer"() : () -> !atir.tensor<4xi32, name = "range_out">
  %18 = atir.constant "public" @"range_start" -> <i32, name = "range_start", data = dense<0> : tensor<i32>>
  %19 = atir.constant "public" @"range_limit" -> <i32, name = "range_limit", data = dense<4> : tensor<i32>>
  %20 = atir.constant "public" @"range_delta" -> <i32, name = "range_delta", data = dense<1> : tensor<i32>>
  %21 = "atir.Range"(%17, %18, %19, %20) : (!atir.tensor<4xi32, name = "range_out">, !atir.tensor<i32, name = "range_start", data = dense<0> : tensor<i32>>, !atir.tensor<i32, name = "range_limit", data = dense<4> : tensor<i32>>, !atir.tensor<i32, name = "range_delta", data = dense<1> : tensor<i32>>) -> !atir.tensor<4xi32, name = "range_out">
  %22 = "atir.buffer"() : () -> !atir.tensor<4x1xi32, name = "reshape1_out">
  %23 = atir.constant "public" @"reshape1_shape" -> <2xi32, name = "reshape1_shape", data = dense<[-1, 1]> : tensor<2xi32>>
  %24 = "atir.Reshape"(%22, %21, %23) : (!atir.tensor<4x1xi32, name = "reshape1_out">, !atir.tensor<4xi32, name = "range_out">, !atir.tensor<2xi32, name = "reshape1_shape", data = dense<[-1, 1]> : tensor<2xi32>>) -> !atir.tensor<4x1xi32, name = "reshape1_out">
  %25 = "atir.buffer"() : () -> !atir.tensor<4x1xi64, name = "cast1_out">
  %26 = "atir.Cast"(%25, %24) : (!atir.tensor<4x1xi64, name = "cast1_out">, !atir.tensor<4x1xi32, name = "reshape1_out">) -> !atir.tensor<4x1xi64, name = "cast1_out">
  %27 = "atir.buffer"() : () -> !atir.tensor<4x2xi64, name = "concat_out">
  %28 = atir.constant "public" @"concat_axis" -> <i32, name = "concat_axis", data = dense<-1> : tensor<i32>>
  %29 = "atir.ConcatV2"(%27, %26, %16, %28) : (!atir.tensor<4x2xi64, name = "concat_out">, !atir.tensor<4x1xi64, name = "cast1_out">, !atir.tensor<?x1xi64, name = "reshape_out">, !atir.tensor<i32, name = "concat_axis", data = dense<-1> : tensor<i32>>) -> !atir.tensor<4x2xi64, name = "concat_out">
  %30:2 = atir.SparseReshape %29, %9, %arg2 : (!atir.tensor<4x2xi64, name = "concat_out">, !atir.tensor<2xi64, name = "cast_out">, !atir.tensor<2xi64, name = "new_shape">) -> (!atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">)
  return %30#0, %30#1 : !atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">
  }
}

// -----

// T = int64 specialisation (pack_const i64) — second kernel variant.
// CHECK-LABEL: func.func @main
// CHECK: atir.SparseReshape
// CHECK-SAME: annc.fusion_materialized
// CHECK-LABEL: func.func private @fused_kp_sparse_reshape_
// CHECK-SAME: %arg0: !llvm.ptr,
// CHECK-SAME: %arg3: !atir.tensor<i64, name = "pack_const">
// CHECK-SAME: annc.kernel
// CHECK-SAME: abi = "annc_execution_v2"
module attributes {module.state = "atir"} {
  func.func @main(
      %arg0: !atir.tensor<4x3xi64, name = "keys">,
      %arg1: !atir.tensor<2xi32, name = "begin">,
      %arg2: !atir.tensor<2xi64, name = "new_shape">,
      %arg3: !atir.tensor<i64, name = "pack_const">)
      -> (!atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">) {
  %0 = atir.Shape %arg0 : (!atir.tensor<4x3xi64, name = "keys">) -> !atir.tensor<2xi32, name = "shape_out">
  %1 = "atir.buffer"() : () -> !atir.tensor<i32, name = "ss1_out">
  %2 = atir.constant "public" @"ss1_begin" -> <1xi32, name = "ss1_begin", data = dense<0> : tensor<1xi32>>
  %3 = atir.constant "public" @"ss1_end" -> <1xi32, name = "ss1_end", data = dense<1> : tensor<1xi32>>
  %4 = atir.constant "public" @"ss1_strides" -> <1xi32, name = "ss1_strides", data = dense<1> : tensor<1xi32>>
  %5 = "atir.StridedSlice"(%1, %0, %2, %3, %4) <{beginMask = 0 : i32, endMask = 0 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 1 : i32}> : (!atir.tensor<i32, name = "ss1_out">, !atir.tensor<2xi32, name = "shape_out">, !atir.tensor<1xi32, name = "ss1_begin", data = dense<0> : tensor<1xi32>>, !atir.tensor<1xi32, name = "ss1_end", data = dense<1> : tensor<1xi32>>, !atir.tensor<1xi32, name = "ss1_strides", data = dense<1> : tensor<1xi32>>) -> !atir.tensor<i32, name = "ss1_out">
  %6 = "atir.buffer"() : () -> !atir.tensor<2xi64, name = "pack_out">
  %7 = "atir.Pack"(%6, %5, %arg3) <{axis = 0 : i64}> : (!atir.tensor<2xi64, name = "pack_out">, !atir.tensor<i32, name = "ss1_out">, !atir.tensor<i64, name = "pack_const">) -> !atir.tensor<2xi64, name = "pack_out">
  %8 = "atir.buffer"() : () -> !atir.tensor<2xi64, name = "cast_out">
  %9 = "atir.Cast"(%8, %7) : (!atir.tensor<2xi64, name = "cast_out">, !atir.tensor<2xi64, name = "pack_out">) -> !atir.tensor<2xi64, name = "cast_out">
  %10 = "atir.buffer"() : () -> !atir.tensor<i64, name = "ss_out">
  %11 = atir.constant "public" @"ss_end" -> <2xi32, name = "ss_end", data = dense<[0, 0]> : tensor<2xi32>>
  %12 = atir.constant "public" @"ss_strides" -> <2xi32, name = "ss_strides", data = dense<[1, 1]> : tensor<2xi32>>
  %13 = "atir.StridedSlice"(%10, %arg0, %arg1, %11, %12) <{beginMask = 1 : i32, endMask = 1 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 2 : i32}> : (!atir.tensor<i64, name = "ss_out">, !atir.tensor<4x3xi64, name = "keys">, !atir.tensor<2xi32, name = "begin">, !atir.tensor<2xi32, name = "ss_end", data = dense<[0, 0]> : tensor<2xi32>>, !atir.tensor<2xi32, name = "ss_strides", data = dense<[1, 1]> : tensor<2xi32>>) -> !atir.tensor<i64, name = "ss_out">
  %14 = "atir.buffer"() : () -> !atir.tensor<?x1xi64, name = "reshape_out">
  %15 = atir.constant "public" @"reshape_shape" -> <2xi32, name = "reshape_shape", data = dense<[-1, 1]> : tensor<2xi32>>
  %16 = "atir.Reshape"(%14, %13, %15) : (!atir.tensor<?x1xi64, name = "reshape_out">, !atir.tensor<i64, name = "ss_out">, !atir.tensor<2xi32, name = "reshape_shape", data = dense<[-1, 1]> : tensor<2xi32>>) -> !atir.tensor<?x1xi64, name = "reshape_out">
  %17 = "atir.buffer"() : () -> !atir.tensor<4xi32, name = "range_out">
  %18 = atir.constant "public" @"range_start" -> <i32, name = "range_start", data = dense<0> : tensor<i32>>
  %19 = atir.constant "public" @"range_limit" -> <i32, name = "range_limit", data = dense<4> : tensor<i32>>
  %20 = atir.constant "public" @"range_delta" -> <i32, name = "range_delta", data = dense<1> : tensor<i32>>
  %21 = "atir.Range"(%17, %18, %19, %20) : (!atir.tensor<4xi32, name = "range_out">, !atir.tensor<i32, name = "range_start", data = dense<0> : tensor<i32>>, !atir.tensor<i32, name = "range_limit", data = dense<4> : tensor<i32>>, !atir.tensor<i32, name = "range_delta", data = dense<1> : tensor<i32>>) -> !atir.tensor<4xi32, name = "range_out">
  %22 = "atir.buffer"() : () -> !atir.tensor<4x1xi32, name = "reshape1_out">
  %23 = atir.constant "public" @"reshape1_shape" -> <2xi32, name = "reshape1_shape", data = dense<[-1, 1]> : tensor<2xi32>>
  %24 = "atir.Reshape"(%22, %21, %23) : (!atir.tensor<4x1xi32, name = "reshape1_out">, !atir.tensor<4xi32, name = "range_out">, !atir.tensor<2xi32, name = "reshape1_shape", data = dense<[-1, 1]> : tensor<2xi32>>) -> !atir.tensor<4x1xi32, name = "reshape1_out">
  %25 = "atir.buffer"() : () -> !atir.tensor<4x1xi64, name = "cast1_out">
  %26 = "atir.Cast"(%25, %24) : (!atir.tensor<4x1xi64, name = "cast1_out">, !atir.tensor<4x1xi32, name = "reshape1_out">) -> !atir.tensor<4x1xi64, name = "cast1_out">
  %27 = "atir.buffer"() : () -> !atir.tensor<4x2xi64, name = "concat_out">
  %28 = atir.constant "public" @"concat_axis" -> <i32, name = "concat_axis", data = dense<-1> : tensor<i32>>
  %29 = "atir.ConcatV2"(%27, %26, %16, %28) : (!atir.tensor<4x2xi64, name = "concat_out">, !atir.tensor<4x1xi64, name = "cast1_out">, !atir.tensor<?x1xi64, name = "reshape_out">, !atir.tensor<i32, name = "concat_axis", data = dense<-1> : tensor<i32>>) -> !atir.tensor<4x2xi64, name = "concat_out">
  %30:2 = atir.SparseReshape %29, %9, %arg2 : (!atir.tensor<4x2xi64, name = "concat_out">, !atir.tensor<2xi64, name = "cast_out">, !atir.tensor<2xi64, name = "new_shape">) -> (!atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">)
  return %30#0, %30#1 : !atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">
  }
}

// -----

// Negative: ConcatV2 axis = 0 (812 要求 axis={-1} → 拒绝).
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_kp_sparse_reshape_
// CHECK: atir.ConcatV2
module attributes {module.state = "atir"} {
  func.func @main(
      %arg0: !atir.tensor<4x3xi64, name = "keys">,
      %arg1: !atir.tensor<2xi32, name = "begin">,
      %arg2: !atir.tensor<2xi64, name = "new_shape">,
      %arg3: !atir.tensor<i32, name = "pack_const">)
      -> (!atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">) {
  %0 = atir.Shape %arg0 : (!atir.tensor<4x3xi64, name = "keys">) -> !atir.tensor<2xi32, name = "shape_out">
  %1 = "atir.buffer"() : () -> !atir.tensor<i32, name = "ss1_out">
  %2 = atir.constant "public" @"ss1_begin" -> <1xi32, name = "ss1_begin", data = dense<0> : tensor<1xi32>>
  %3 = atir.constant "public" @"ss1_end" -> <1xi32, name = "ss1_end", data = dense<1> : tensor<1xi32>>
  %4 = atir.constant "public" @"ss1_strides" -> <1xi32, name = "ss1_strides", data = dense<1> : tensor<1xi32>>
  %5 = "atir.StridedSlice"(%1, %0, %2, %3, %4) <{beginMask = 0 : i32, endMask = 0 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 1 : i32}> : (!atir.tensor<i32, name = "ss1_out">, !atir.tensor<2xi32, name = "shape_out">, !atir.tensor<1xi32, name = "ss1_begin", data = dense<0> : tensor<1xi32>>, !atir.tensor<1xi32, name = "ss1_end", data = dense<1> : tensor<1xi32>>, !atir.tensor<1xi32, name = "ss1_strides", data = dense<1> : tensor<1xi32>>) -> !atir.tensor<i32, name = "ss1_out">
  %6 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "pack_out">
  %7 = "atir.Pack"(%6, %5, %arg3) <{axis = 0 : i64}> : (!atir.tensor<2xi32, name = "pack_out">, !atir.tensor<i32, name = "ss1_out">, !atir.tensor<i32, name = "pack_const">) -> !atir.tensor<2xi32, name = "pack_out">
  %8 = "atir.buffer"() : () -> !atir.tensor<2xi64, name = "cast_out">
  %9 = "atir.Cast"(%8, %7) : (!atir.tensor<2xi64, name = "cast_out">, !atir.tensor<2xi32, name = "pack_out">) -> !atir.tensor<2xi64, name = "cast_out">
  %10 = "atir.buffer"() : () -> !atir.tensor<i64, name = "ss_out">
  %11 = atir.constant "public" @"ss_end" -> <2xi32, name = "ss_end", data = dense<[0, 0]> : tensor<2xi32>>
  %12 = atir.constant "public" @"ss_strides" -> <2xi32, name = "ss_strides", data = dense<[1, 1]> : tensor<2xi32>>
  %13 = "atir.StridedSlice"(%10, %arg0, %arg1, %11, %12) <{beginMask = 1 : i32, endMask = 1 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 2 : i32}> : (!atir.tensor<i64, name = "ss_out">, !atir.tensor<4x3xi64, name = "keys">, !atir.tensor<2xi32, name = "begin">, !atir.tensor<2xi32, name = "ss_end", data = dense<[0, 0]> : tensor<2xi32>>, !atir.tensor<2xi32, name = "ss_strides", data = dense<[1, 1]> : tensor<2xi32>>) -> !atir.tensor<i64, name = "ss_out">
  %14 = "atir.buffer"() : () -> !atir.tensor<?x1xi64, name = "reshape_out">
  %15 = atir.constant "public" @"reshape_shape" -> <2xi32, name = "reshape_shape", data = dense<[-1, 1]> : tensor<2xi32>>
  %16 = "atir.Reshape"(%14, %13, %15) : (!atir.tensor<?x1xi64, name = "reshape_out">, !atir.tensor<i64, name = "ss_out">, !atir.tensor<2xi32, name = "reshape_shape", data = dense<[-1, 1]> : tensor<2xi32>>) -> !atir.tensor<?x1xi64, name = "reshape_out">
  %17 = "atir.buffer"() : () -> !atir.tensor<4xi32, name = "range_out">
  %18 = atir.constant "public" @"range_start" -> <i32, name = "range_start", data = dense<0> : tensor<i32>>
  %19 = atir.constant "public" @"range_limit" -> <i32, name = "range_limit", data = dense<4> : tensor<i32>>
  %20 = atir.constant "public" @"range_delta" -> <i32, name = "range_delta", data = dense<1> : tensor<i32>>
  %21 = "atir.Range"(%17, %18, %19, %20) : (!atir.tensor<4xi32, name = "range_out">, !atir.tensor<i32, name = "range_start", data = dense<0> : tensor<i32>>, !atir.tensor<i32, name = "range_limit", data = dense<4> : tensor<i32>>, !atir.tensor<i32, name = "range_delta", data = dense<1> : tensor<i32>>) -> !atir.tensor<4xi32, name = "range_out">
  %22 = "atir.buffer"() : () -> !atir.tensor<4x1xi32, name = "reshape1_out">
  %23 = atir.constant "public" @"reshape1_shape" -> <2xi32, name = "reshape1_shape", data = dense<[-1, 1]> : tensor<2xi32>>
  %24 = "atir.Reshape"(%22, %21, %23) : (!atir.tensor<4x1xi32, name = "reshape1_out">, !atir.tensor<4xi32, name = "range_out">, !atir.tensor<2xi32, name = "reshape1_shape", data = dense<[-1, 1]> : tensor<2xi32>>) -> !atir.tensor<4x1xi32, name = "reshape1_out">
  %25 = "atir.buffer"() : () -> !atir.tensor<4x1xi64, name = "cast1_out">
  %26 = "atir.Cast"(%25, %24) : (!atir.tensor<4x1xi64, name = "cast1_out">, !atir.tensor<4x1xi32, name = "reshape1_out">) -> !atir.tensor<4x1xi64, name = "cast1_out">
  %27 = "atir.buffer"() : () -> !atir.tensor<4x2xi64, name = "concat_out">
  %28 = atir.constant "public" @"concat_axis" -> <i32, name = "concat_axis", data = dense<0> : tensor<i32>>
  %29 = "atir.ConcatV2"(%27, %26, %16, %28) : (!atir.tensor<4x2xi64, name = "concat_out">, !atir.tensor<4x1xi64, name = "cast1_out">, !atir.tensor<?x1xi64, name = "reshape_out">, !atir.tensor<i32, name = "concat_axis", data = dense<0> : tensor<i32>>) -> !atir.tensor<4x2xi64, name = "concat_out">
  %30:2 = atir.SparseReshape %29, %9, %arg2 : (!atir.tensor<4x2xi64, name = "concat_out">, !atir.tensor<2xi64, name = "cast_out">, !atir.tensor<2xi64, name = "new_shape">) -> (!atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">)
  return %30#0, %30#1 : !atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">
  }
}
