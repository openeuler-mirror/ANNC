// RUN: annc-asm --split-input-file %s -atir-fast-codegen | FileCheck %s
// KP-style sparse reshape fusion, second level, Execution V2: inside the
// kernel func (signature !llvm.ptr + 4 inputs, 2 results) the chain folds
// into a single KPFusedSparseReshape<Idx> CustomizeOp; the two .Result()
// entries map to execution slots 0 (out_indices) and 1 (out_shape).

// 正例 1: T = int32 (pack_const i32)。
// CHECK-LABEL: func.func private @fused_kp_sparse_reshape
// CHECK-SAME: %arg0: !llvm.ptr,
// CHECK: atir.Customize
// CHECK-SAME: custom.op_name = "KPFusedSparseReshapeI32"
// CHECK-SAME: custom.result_names = ["out_indices", "out_shape"]
// CHECK-NOT: atir.SparseReshape
// CHECK-NOT: atir.ConcatV2
module attributes {module.state = "atir"} {
  func.func private @fused_kp_sparse_reshape(
      %arg0: !llvm.ptr,
      %arg1: !atir.tensor<4x3xi64, name = "keys">,
      %arg2: !atir.tensor<2xi32, name = "begin">,
      %arg3: !atir.tensor<i32, name = "pack_const">,
      %arg4: !atir.tensor<2xi64, name = "new_shape">)
      -> (!atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">)
      attributes {annc.kernel, fusion.metadata = {abi = "annc_execution_v2"}} {
    %0 = atir.Shape %arg1 : (!atir.tensor<4x3xi64, name = "keys">) -> !atir.tensor<2xi32, name = "shape_out">
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
    %13 = "atir.StridedSlice"(%10, %arg1, %arg2, %11, %12) <{beginMask = 1 : i32, endMask = 1 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 2 : i32}> : (!atir.tensor<i64, name = "ss_out">, !atir.tensor<4x3xi64, name = "keys">, !atir.tensor<2xi32, name = "begin">, !atir.tensor<2xi32, name = "ss_end", data = dense<[0, 0]> : tensor<2xi32>>, !atir.tensor<2xi32, name = "ss_strides", data = dense<[1, 1]> : tensor<2xi32>>) -> !atir.tensor<i64, name = "ss_out">
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
    %30:2 = atir.SparseReshape %29, %9, %arg4 : (!atir.tensor<4x2xi64, name = "concat_out">, !atir.tensor<2xi64, name = "cast_out">, !atir.tensor<2xi64, name = "new_shape">) -> (!atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">)
    return %30#0, %30#1 : !atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">
  }
}

// -----

// 正例 1b: T = int32 + 动态 Range.limit (0-D i32 block arg, varlen batch) —
// DynLimit kernel 变体 (limit 参数 kernel 不用, 只占 ABI 槽位)。
// CHECK-LABEL: func.func private @fused_kp_sparse_reshape_dynlimit
// CHECK-SAME: %arg0: !llvm.ptr,
// CHECK-SAME: %arg5: !atir.tensor<i32, name = "limit">
// CHECK: atir.Customize
// CHECK-SAME: custom.op_name = "KPFusedSparseReshapeI32DynLimit"
// CHECK-SAME: custom.result_names = ["out_indices", "out_shape"]
// CHECK-NOT: atir.SparseReshape
// CHECK-NOT: atir.ConcatV2
module attributes {module.state = "atir"} {
  func.func private @fused_kp_sparse_reshape_dynlimit(
      %arg0: !llvm.ptr,
      %arg1: !atir.tensor<4x3xi64, name = "keys">,
      %arg2: !atir.tensor<2xi32, name = "begin">,
      %arg3: !atir.tensor<i32, name = "pack_const">,
      %arg4: !atir.tensor<2xi64, name = "new_shape">,
      %arg5: !atir.tensor<i32, name = "limit">)
      -> (!atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">)
      attributes {annc.kernel, fusion.metadata = {abi = "annc_execution_v2"}} {
    %0 = atir.Shape %arg1 : (!atir.tensor<4x3xi64, name = "keys">) -> !atir.tensor<2xi32, name = "shape_out">
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
    %13 = "atir.StridedSlice"(%10, %arg1, %arg2, %11, %12) <{beginMask = 1 : i32, endMask = 1 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 2 : i32}> : (!atir.tensor<i64, name = "ss_out">, !atir.tensor<4x3xi64, name = "keys">, !atir.tensor<2xi32, name = "begin">, !atir.tensor<2xi32, name = "ss_end", data = dense<[0, 0]> : tensor<2xi32>>, !atir.tensor<2xi32, name = "ss_strides", data = dense<[1, 1]> : tensor<2xi32>>) -> !atir.tensor<i64, name = "ss_out">
    %14 = "atir.buffer"() : () -> !atir.tensor<?x1xi64, name = "reshape_out">
    %15 = atir.constant "public" @"reshape_shape" -> <2xi32, name = "reshape_shape", data = dense<[-1, 1]> : tensor<2xi32>>
    %16 = "atir.Reshape"(%14, %13, %15) : (!atir.tensor<?x1xi64, name = "reshape_out">, !atir.tensor<i64, name = "ss_out">, !atir.tensor<2xi32, name = "reshape_shape", data = dense<[-1, 1]> : tensor<2xi32>>) -> !atir.tensor<?x1xi64, name = "reshape_out">
    %17 = "atir.buffer"() : () -> !atir.tensor<?xi32, name = "range_out">
    %18 = atir.constant "public" @"range_start" -> <i32, name = "range_start", data = dense<0> : tensor<i32>>
    %19 = atir.constant "public" @"range_delta" -> <i32, name = "range_delta", data = dense<1> : tensor<i32>>
    %20 = "atir.Range"(%17, %18, %arg5, %19) : (!atir.tensor<?xi32, name = "range_out">, !atir.tensor<i32, name = "range_start", data = dense<0> : tensor<i32>>, !atir.tensor<i32, name = "limit">, !atir.tensor<i32, name = "range_delta", data = dense<1> : tensor<i32>>) -> !atir.tensor<?xi32, name = "range_out">
    %21 = "atir.buffer"() : () -> !atir.tensor<?x1xi32, name = "reshape1_out">
    %22 = atir.constant "public" @"reshape1_shape" -> <2xi32, name = "reshape1_shape", data = dense<[-1, 1]> : tensor<2xi32>>
    %23 = "atir.Reshape"(%21, %20, %22) : (!atir.tensor<?x1xi32, name = "reshape1_out">, !atir.tensor<?xi32, name = "range_out">, !atir.tensor<2xi32, name = "reshape1_shape", data = dense<[-1, 1]> : tensor<2xi32>>) -> !atir.tensor<?x1xi32, name = "reshape1_out">
    %24 = "atir.buffer"() : () -> !atir.tensor<?x1xi64, name = "cast1_out">
    %25 = "atir.Cast"(%24, %23) : (!atir.tensor<?x1xi64, name = "cast1_out">, !atir.tensor<?x1xi32, name = "reshape1_out">) -> !atir.tensor<?x1xi64, name = "cast1_out">
    %26 = "atir.buffer"() : () -> !atir.tensor<?x2xi64, name = "concat_out">
    %27 = atir.constant "public" @"concat_axis" -> <i32, name = "concat_axis", data = dense<-1> : tensor<i32>>
    %28 = "atir.ConcatV2"(%26, %25, %16, %27) : (!atir.tensor<?x2xi64, name = "concat_out">, !atir.tensor<?x1xi64, name = "cast1_out">, !atir.tensor<?x1xi64, name = "reshape_out">, !atir.tensor<i32, name = "concat_axis", data = dense<-1> : tensor<i32>>) -> !atir.tensor<?x2xi64, name = "concat_out">
    %29:2 = atir.SparseReshape %28, %9, %arg4 : (!atir.tensor<?x2xi64, name = "concat_out">, !atir.tensor<2xi64, name = "cast_out">, !atir.tensor<2xi64, name = "new_shape">) -> (!atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">)
    return %29#0, %29#1 : !atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">
  }
}

// -----

// 正例 2: T = int64 (pack_const i64) — 第二个 kernel 特化。
// CHECK-LABEL: func.func private @fused_kp_sparse_reshape_i64
// CHECK-SAME: %arg0: !llvm.ptr,
// CHECK: atir.Customize
// CHECK-SAME: custom.op_name = "KPFusedSparseReshapeI64"
module attributes {module.state = "atir"} {
  func.func private @fused_kp_sparse_reshape_i64(
      %arg0: !llvm.ptr,
      %arg1: !atir.tensor<4x3xi64, name = "keys">,
      %arg2: !atir.tensor<2xi32, name = "begin">,
      %arg3: !atir.tensor<i64, name = "pack_const">,
      %arg4: !atir.tensor<2xi64, name = "new_shape">)
      -> (!atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">)
      attributes {annc.kernel, fusion.metadata = {abi = "annc_execution_v2"}} {
    %0 = atir.Shape %arg1 : (!atir.tensor<4x3xi64, name = "keys">) -> !atir.tensor<2xi32, name = "shape_out">
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
    %13 = "atir.StridedSlice"(%10, %arg1, %arg2, %11, %12) <{beginMask = 1 : i32, endMask = 1 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 2 : i32}> : (!atir.tensor<i64, name = "ss_out">, !atir.tensor<4x3xi64, name = "keys">, !atir.tensor<2xi32, name = "begin">, !atir.tensor<2xi32, name = "ss_end", data = dense<[0, 0]> : tensor<2xi32>>, !atir.tensor<2xi32, name = "ss_strides", data = dense<[1, 1]> : tensor<2xi32>>) -> !atir.tensor<i64, name = "ss_out">
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
    %30:2 = atir.SparseReshape %29, %9, %arg4 : (!atir.tensor<4x2xi64, name = "concat_out">, !atir.tensor<2xi64, name = "cast_out">, !atir.tensor<2xi64, name = "new_shape">) -> (!atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">)
    return %30#0, %30#1 : !atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">
  }
}

// -----

// 正例 1c (D-2 放宽 2): Range.limit 是链内交叉引用 ss1 (= Shape(keys)[0]),
// 不是 block arg。ss1 有两个用户 {Pack, Range}, 两端都在融合区内, 不是逃逸。
// 签名仍是 5 参 (无 range_limit 槽), 因此选非 DynLimit 变体。
// CHECK-LABEL: func.func private @fused_kp_sparse_reshape_xref
// CHECK-SAME: %arg0: !llvm.ptr,
// CHECK: atir.Customize
// CHECK-SAME: custom.op_name = "KPFusedSparseReshapeI32"
// CHECK-SAME: custom.result_names = ["out_indices", "out_shape"]
// CHECK-NOT: atir.SparseReshape
// CHECK-NOT: atir.ConcatV2
module attributes {module.state = "atir"} {
  func.func private @fused_kp_sparse_reshape_xref(
      %arg0: !llvm.ptr,
      %arg1: !atir.tensor<4x3xi64, name = "keys">,
      %arg2: !atir.tensor<2xi32, name = "begin">,
      %arg3: !atir.tensor<i32, name = "pack_const">,
      %arg4: !atir.tensor<2xi64, name = "new_shape">)
      -> (!atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">)
      attributes {annc.kernel, fusion.metadata = {abi = "annc_execution_v2"}} {
    %0 = atir.Shape %arg1 : (!atir.tensor<4x3xi64, name = "keys">) -> !atir.tensor<2xi32, name = "shape_out">
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
    %13 = "atir.StridedSlice"(%10, %arg1, %arg2, %11, %12) <{beginMask = 1 : i32, endMask = 1 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 2 : i32}> : (!atir.tensor<i64, name = "ss_out">, !atir.tensor<4x3xi64, name = "keys">, !atir.tensor<2xi32, name = "begin">, !atir.tensor<2xi32, name = "ss_end", data = dense<[0, 0]> : tensor<2xi32>>, !atir.tensor<2xi32, name = "ss_strides", data = dense<[1, 1]> : tensor<2xi32>>) -> !atir.tensor<i64, name = "ss_out">
    %14 = "atir.buffer"() : () -> !atir.tensor<?x1xi64, name = "reshape_out">
    %15 = atir.constant "public" @"reshape_shape" -> <2xi32, name = "reshape_shape", data = dense<[-1, 1]> : tensor<2xi32>>
    %16 = "atir.Reshape"(%14, %13, %15) : (!atir.tensor<?x1xi64, name = "reshape_out">, !atir.tensor<i64, name = "ss_out">, !atir.tensor<2xi32, name = "reshape_shape", data = dense<[-1, 1]> : tensor<2xi32>>) -> !atir.tensor<?x1xi64, name = "reshape_out">
    %17 = "atir.buffer"() : () -> !atir.tensor<?xi32, name = "range_out">
    %18 = atir.constant "public" @"range_start" -> <i32, name = "range_start", data = dense<0> : tensor<i32>>
    %19 = atir.constant "public" @"range_delta" -> <i32, name = "range_delta", data = dense<1> : tensor<i32>>
    %20 = "atir.Range"(%17, %18, %5, %19) : (!atir.tensor<?xi32, name = "range_out">, !atir.tensor<i32, name = "range_start", data = dense<0> : tensor<i32>>, !atir.tensor<i32, name = "ss1_out">, !atir.tensor<i32, name = "range_delta", data = dense<1> : tensor<i32>>) -> !atir.tensor<?xi32, name = "range_out">
    %21 = "atir.buffer"() : () -> !atir.tensor<?x1xi32, name = "reshape1_out">
    %22 = atir.constant "public" @"reshape1_shape" -> <2xi32, name = "reshape1_shape", data = dense<[-1, 1]> : tensor<2xi32>>
    %23 = "atir.Reshape"(%21, %20, %22) : (!atir.tensor<?x1xi32, name = "reshape1_out">, !atir.tensor<?xi32, name = "range_out">, !atir.tensor<2xi32, name = "reshape1_shape", data = dense<[-1, 1]> : tensor<2xi32>>) -> !atir.tensor<?x1xi32, name = "reshape1_out">
    %24 = "atir.buffer"() : () -> !atir.tensor<?x1xi64, name = "cast1_out">
    %25 = "atir.Cast"(%24, %23) : (!atir.tensor<?x1xi64, name = "cast1_out">, !atir.tensor<?x1xi32, name = "reshape1_out">) -> !atir.tensor<?x1xi64, name = "cast1_out">
    %26 = "atir.buffer"() : () -> !atir.tensor<?x2xi64, name = "concat_out">
    %27 = atir.constant "public" @"concat_axis" -> <i32, name = "concat_axis", data = dense<-1> : tensor<i32>>
    %28 = "atir.ConcatV2"(%26, %25, %16, %27) : (!atir.tensor<?x2xi64, name = "concat_out">, !atir.tensor<?x1xi64, name = "cast1_out">, !atir.tensor<?x1xi64, name = "reshape_out">, !atir.tensor<i32, name = "concat_axis", data = dense<-1> : tensor<i32>>) -> !atir.tensor<?x2xi64, name = "concat_out">
    %29:2 = atir.SparseReshape %28, %9, %arg4 : (!atir.tensor<?x2xi64, name = "concat_out">, !atir.tensor<2xi64, name = "cast_out">, !atir.tensor<2xi64, name = "new_shape">) -> (!atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">)
    return %29#0, %29#1 : !atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">
  }
}

// -----

// 负例: ss shrink_mask = 1 (812 要求 2 → 拒绝).
// CHECK-LABEL: func.func private @bad_shrink
// CHECK-NOT: atir.Customize
// CHECK: atir.SparseReshape
module attributes {module.state = "atir"} {
  func.func private @bad_shrink(
      %arg0: !llvm.ptr,
      %arg1: !atir.tensor<4x3xi64, name = "keys">,
      %arg2: !atir.tensor<2xi32, name = "begin">,
      %arg3: !atir.tensor<i32, name = "pack_const">,
      %arg4: !atir.tensor<2xi64, name = "new_shape">)
      -> (!atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">)
      attributes {annc.kernel, fusion.metadata = {abi = "annc_execution_v2"}} {
    %0 = atir.Shape %arg1 : (!atir.tensor<4x3xi64, name = "keys">) -> !atir.tensor<2xi32, name = "shape_out">
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
    %13 = "atir.StridedSlice"(%10, %arg1, %arg2, %11, %12) <{beginMask = 1 : i32, endMask = 1 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 1 : i32}> : (!atir.tensor<i64, name = "ss_out">, !atir.tensor<4x3xi64, name = "keys">, !atir.tensor<2xi32, name = "begin">, !atir.tensor<2xi32, name = "ss_end", data = dense<[0, 0]> : tensor<2xi32>>, !atir.tensor<2xi32, name = "ss_strides", data = dense<[1, 1]> : tensor<2xi32>>) -> !atir.tensor<i64, name = "ss_out">
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
    %30:2 = atir.SparseReshape %29, %9, %arg4 : (!atir.tensor<4x2xi64, name = "concat_out">, !atir.tensor<2xi64, name = "cast_out">, !atir.tensor<2xi64, name = "new_shape">) -> (!atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">)
    return %30#0, %30#1 : !atir.tensor<?x2xi64, name = "out_indices">, !atir.tensor<2xi64, name = "out_shape">
  }
}
