// RUN: annc-asm --split-input-file %s -atir-fast-codegen | FileCheck %s
// KP-style embedding padding fusion, second level, Execution V2: inside the
// kernel func (signature !llvm.ptr + 5 inputs, 2 results) the padding chain
// folds into a single CustomizeOp; results map to execution slots
// (0 = padding_rows, 1 = out_data / reshape_rows).

// CHECK-LABEL: func.func private @fused_kp_embedding_padding
// CHECK-SAME: %arg0: !llvm.ptr,
// CHECK-SAME: %arg1: !atir.tensor<2xi64, name = "origin_shape">, %arg2: !atir.tensor<i32, name = "input_rows">, %arg3: !atir.tensor<i32, name = "pack">, %arg4: !atir.tensor<4x3xf32, name = "data">, %arg5: !atir.tensor<2xi32, name = "reshape_sizes">
// CHECK: atir.Customize
// CHECK-SAME: custom.op_name = "KPFusedEmbeddingPadding"
// CHECK-SAME: custom.result_names = ["out_padding_rows", "out_data"]
// CHECK-NOT: atir.Sub
// CHECK-NOT: atir.Reshape
// 注: kernel func 只克隆定义链, anchor 的 ConcatV2 用户 (812 要求) 在一级
// 主图层面检查, 不出现在 kernel func 里。
module attributes {module.state = "atir"} {
  func.func private @fused_kp_embedding_padding(
      %arg0: !llvm.ptr,
      %arg1: !atir.tensor<2xi64, name = "origin_shape">,
      %arg2: !atir.tensor<i32, name = "input_rows">,
      %arg3: !atir.tensor<i32, name = "pack">,
      %arg4: !atir.tensor<4x3xf32, name = "data">,
      %arg5: !atir.tensor<2xi32, name = "reshape_sizes">)
      -> (!atir.tensor<i32, name = "out_padding">, !atir.tensor<?x3xf32, name = "out_data">)
      attributes {annc.kernel, fusion.metadata = {abi = "annc_execution_v2"}} {
    %0 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "cast_out">
    %1 = "atir.Cast"(%0, %arg1) : (!atir.tensor<2xi32, name = "cast_out">, !atir.tensor<2xi64, name = "origin_shape">) -> !atir.tensor<2xi32, name = "cast_out">
    %2 = "atir.buffer"() : () -> !atir.tensor<i32, name = "ss_out">
    %3 = atir.constant "public" @"ss_begin" -> <1xi32, name = "ss_begin", data = dense<0> : tensor<1xi32>>
    %4 = atir.constant "public" @"ss_end" -> <1xi32, name = "ss_end", data = dense<1> : tensor<1xi32>>
    %5 = atir.constant "public" @"ss_strides" -> <1xi32, name = "ss_strides", data = dense<1> : tensor<1xi32>>
    %6 = "atir.StridedSlice"(%2, %1, %3, %4, %5) <{beginMask = 0 : i32, endMask = 0 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 1 : i32}> : (!atir.tensor<i32, name = "ss_out">, !atir.tensor<2xi32, name = "cast_out">, !atir.tensor<1xi32, name = "ss_begin", data = dense<0> : tensor<1xi32>>, !atir.tensor<1xi32, name = "ss_end", data = dense<1> : tensor<1xi32>>, !atir.tensor<1xi32, name = "ss_strides", data = dense<1> : tensor<1xi32>>) -> !atir.tensor<i32, name = "ss_out">
    %7 = "atir.buffer"() : () -> !atir.tensor<i32, name = "out_padding">
    %8 = "atir.Sub"(%7, %6, %arg2) : (!atir.tensor<i32, name = "out_padding">, !atir.tensor<i32, name = "ss_out">, !atir.tensor<i32, name = "input_rows">) -> !atir.tensor<i32, name = "out_padding">
    %9 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "pack_out">
    %10 = "atir.Pack"(%9, %8, %arg3) <{axis = 0 : i64}> : (!atir.tensor<2xi32, name = "pack_out">, !atir.tensor<i32, name = "out_padding">, !atir.tensor<i32, name = "pack">) -> !atir.tensor<2xi32, name = "pack_out">
    %11 = "atir.buffer"() : () -> !atir.tensor<?x3xf32, name = "fill_out">
    %12 = atir.constant "public" @"fill_value" -> <f32, name = "fill_value", data = dense<0.000000e+00> : tensor<f32>>
    %13 = "atir.Fill"(%11, %10, %12) : (!atir.tensor<?x3xf32, name = "fill_out">, !atir.tensor<2xi32, name = "pack_out">, !atir.tensor<f32, name = "fill_value", data = dense<0.000000e+00> : tensor<f32>>) -> !atir.tensor<?x3xf32, name = "fill_out">
    %14 = "atir.buffer"() : () -> !atir.tensor<?x3xf32, name = "concat_out">
    %15 = atir.constant "public" @"concat_axis" -> <i32, name = "concat_axis", data = dense<0> : tensor<i32>>
    %16 = "atir.ConcatV2"(%14, %arg4, %13, %15) : (!atir.tensor<?x3xf32, name = "concat_out">, !atir.tensor<4x3xf32, name = "data">, !atir.tensor<?x3xf32, name = "fill_out">, !atir.tensor<i32, name = "concat_axis", data = dense<0> : tensor<i32>>) -> !atir.tensor<?x3xf32, name = "concat_out">
    %17 = "atir.buffer"() : () -> !atir.tensor<?x3xf32, name = "out_data">
    %18 = "atir.Reshape"(%17, %16, %arg5) : (!atir.tensor<?x3xf32, name = "out_data">, !atir.tensor<?x3xf32, name = "concat_out">, !atir.tensor<2xi32, name = "reshape_sizes">) -> !atir.tensor<?x3xf32, name = "out_data">
    return %8, %18 : !atir.tensor<i32, name = "out_padding">, !atir.tensor<?x3xf32, name = "out_data">
  }
}

// -----

// CHECK-LABEL: func.func private @fused_kp_embedding_padding_fast
// CHECK-SAME: %arg0: !llvm.ptr,
// CHECK: atir.Customize
// CHECK-SAME: custom.op_name = "KPFusedEmbeddingPaddingFast"
// CHECK-SAME: custom.result_names = ["out_padding_rows", "out_reshape_rows"]
module attributes {module.state = "atir"} {
  func.func private @fused_kp_embedding_padding_fast(
      %arg0: !llvm.ptr,
      %arg1: !atir.tensor<2xi64, name = "origin_shape">,
      %arg2: !atir.tensor<i32, name = "input_rows">,
      %arg3: !atir.tensor<i32, name = "pack">,
      %arg4: !atir.tensor<4x3xf32, name = "data">,
      %arg5: !atir.tensor<2xi32, name = "reshape_sizes">)
      -> (!atir.tensor<i32, name = "out_padding">, !atir.tensor<i32, name = "out_reshape_rows">)
      attributes {annc.kernel, fusion.metadata = {abi = "annc_execution_v2"}} {
    %0 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "cast_out">
    %1 = "atir.Cast"(%0, %arg1) : (!atir.tensor<2xi32, name = "cast_out">, !atir.tensor<2xi64, name = "origin_shape">) -> !atir.tensor<2xi32, name = "cast_out">
    %2 = "atir.buffer"() : () -> !atir.tensor<i32, name = "ss_out">
    %3 = atir.constant "public" @"ss_begin" -> <1xi32, name = "ss_begin", data = dense<0> : tensor<1xi32>>
    %4 = atir.constant "public" @"ss_end" -> <1xi32, name = "ss_end", data = dense<1> : tensor<1xi32>>
    %5 = atir.constant "public" @"ss_strides" -> <1xi32, name = "ss_strides", data = dense<1> : tensor<1xi32>>
    %6 = "atir.StridedSlice"(%2, %1, %3, %4, %5) <{beginMask = 0 : i32, endMask = 0 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 1 : i32}> : (!atir.tensor<i32, name = "ss_out">, !atir.tensor<2xi32, name = "cast_out">, !atir.tensor<1xi32, name = "ss_begin", data = dense<0> : tensor<1xi32>>, !atir.tensor<1xi32, name = "ss_end", data = dense<1> : tensor<1xi32>>, !atir.tensor<1xi32, name = "ss_strides", data = dense<1> : tensor<1xi32>>) -> !atir.tensor<i32, name = "ss_out">
    %7 = "atir.buffer"() : () -> !atir.tensor<i32, name = "out_padding">
    %8 = "atir.Sub"(%7, %6, %arg2) : (!atir.tensor<i32, name = "out_padding">, !atir.tensor<i32, name = "ss_out">, !atir.tensor<i32, name = "input_rows">) -> !atir.tensor<i32, name = "out_padding">
    %9 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "pack_out">
    %10 = "atir.Pack"(%9, %8, %arg3) <{axis = 0 : i64}> : (!atir.tensor<2xi32, name = "pack_out">, !atir.tensor<i32, name = "out_padding">, !atir.tensor<i32, name = "pack">) -> !atir.tensor<2xi32, name = "pack_out">
    %11 = "atir.buffer"() : () -> !atir.tensor<?x3xf32, name = "fill_out">
    %12 = atir.constant "public" @"fill_value" -> <f32, name = "fill_value", data = dense<0.000000e+00> : tensor<f32>>
    %13 = "atir.Fill"(%11, %10, %12) : (!atir.tensor<?x3xf32, name = "fill_out">, !atir.tensor<2xi32, name = "pack_out">, !atir.tensor<f32, name = "fill_value", data = dense<0.000000e+00> : tensor<f32>>) -> !atir.tensor<?x3xf32, name = "fill_out">
    %14 = "atir.buffer"() : () -> !atir.tensor<?x3xf32, name = "concat_out">
    %15 = atir.constant "public" @"concat_axis" -> <i32, name = "concat_axis", data = dense<0> : tensor<i32>>
    %16 = "atir.ConcatV2"(%14, %arg4, %13, %15) : (!atir.tensor<?x3xf32, name = "concat_out">, !atir.tensor<4x3xf32, name = "data">, !atir.tensor<?x3xf32, name = "fill_out">, !atir.tensor<i32, name = "concat_axis", data = dense<0> : tensor<i32>>) -> !atir.tensor<?x3xf32, name = "concat_out">
    %17 = "atir.buffer"() : () -> !atir.tensor<?x3xf32, name = "reshaped">
    %18 = "atir.Reshape"(%17, %16, %arg5) : (!atir.tensor<?x3xf32, name = "reshaped">, !atir.tensor<?x3xf32, name = "concat_out">, !atir.tensor<2xi32, name = "reshape_sizes">) -> !atir.tensor<?x3xf32, name = "reshaped">
    %19 = atir.Shape %18 : (!atir.tensor<?x3xf32, name = "reshaped">) -> !atir.tensor<2xi32, name = "reshape_shape">
    %20 = atir.constant "public" @"anchor_begin" -> <1xi32, name = "anchor_begin", data = dense<0> : tensor<1xi32>>
    %21 = atir.constant "public" @"anchor_end" -> <1xi32, name = "anchor_end", data = dense<1> : tensor<1xi32>>
    %22 = atir.constant "public" @"anchor_strides" -> <1xi32, name = "anchor_strides", data = dense<1> : tensor<1xi32>>
    %23 = "atir.buffer"() : () -> !atir.tensor<i32, name = "out_reshape_rows">
    %24 = "atir.StridedSlice"(%23, %19, %20, %21, %22) <{beginMask = 0 : i32, endMask = 0 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 1 : i32}> : (!atir.tensor<i32, name = "out_reshape_rows">, !atir.tensor<2xi32, name = "reshape_shape">, !atir.tensor<1xi32, name = "anchor_begin", data = dense<0> : tensor<1xi32>>, !atir.tensor<1xi32, name = "anchor_end", data = dense<1> : tensor<1xi32>>, !atir.tensor<1xi32, name = "anchor_strides", data = dense<1> : tensor<1xi32>>) -> !atir.tensor<i32, name = "out_reshape_rows">
    return %8, %24 : !atir.tensor<i32, name = "out_padding">, !atir.tensor<i32, name = "out_reshape_rows">
  }
}

// -----

// Negative: chain StridedSlice begin = [1] (812 未查, 移植补查 → 拒绝).
// CHECK-LABEL: func.func private @bad_ss_begin
// CHECK-NOT: atir.Customize
// CHECK: atir.Sub
module attributes {module.state = "atir"} {
  func.func private @bad_ss_begin(
      %arg0: !llvm.ptr,
      %arg1: !atir.tensor<2xi64, name = "origin_shape">,
      %arg2: !atir.tensor<i32, name = "input_rows">,
      %arg3: !atir.tensor<i32, name = "pack">,
      %arg4: !atir.tensor<4x3xf32, name = "data">,
      %arg5: !atir.tensor<2xi32, name = "reshape_sizes">)
      -> (!atir.tensor<i32, name = "out_padding">, !atir.tensor<?x3xf32, name = "out_data">)
      attributes {annc.kernel, fusion.metadata = {abi = "annc_execution_v2"}} {
    %0 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "cast_out">
    %1 = "atir.Cast"(%0, %arg1) : (!atir.tensor<2xi32, name = "cast_out">, !atir.tensor<2xi64, name = "origin_shape">) -> !atir.tensor<2xi32, name = "cast_out">
    %2 = "atir.buffer"() : () -> !atir.tensor<i32, name = "ss_out">
    %3 = atir.constant "public" @"ss_begin" -> <1xi32, name = "ss_begin", data = dense<1> : tensor<1xi32>>
    %4 = atir.constant "public" @"ss_end" -> <1xi32, name = "ss_end", data = dense<1> : tensor<1xi32>>
    %5 = atir.constant "public" @"ss_strides" -> <1xi32, name = "ss_strides", data = dense<1> : tensor<1xi32>>
    %6 = "atir.StridedSlice"(%2, %1, %3, %4, %5) <{beginMask = 0 : i32, endMask = 0 : i32, ellipsisMask = 0 : i32, newAxisMask = 0 : i32, shrinkAxisMask = 1 : i32}> : (!atir.tensor<i32, name = "ss_out">, !atir.tensor<2xi32, name = "cast_out">, !atir.tensor<1xi32, name = "ss_begin", data = dense<1> : tensor<1xi32>>, !atir.tensor<1xi32, name = "ss_end", data = dense<1> : tensor<1xi32>>, !atir.tensor<1xi32, name = "ss_strides", data = dense<1> : tensor<1xi32>>) -> !atir.tensor<i32, name = "ss_out">
    %7 = "atir.buffer"() : () -> !atir.tensor<i32, name = "out_padding">
    %8 = "atir.Sub"(%7, %6, %arg2) : (!atir.tensor<i32, name = "out_padding">, !atir.tensor<i32, name = "ss_out">, !atir.tensor<i32, name = "input_rows">) -> !atir.tensor<i32, name = "out_padding">
    %9 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "pack_out">
    %10 = "atir.Pack"(%9, %8, %arg3) <{axis = 0 : i64}> : (!atir.tensor<2xi32, name = "pack_out">, !atir.tensor<i32, name = "out_padding">, !atir.tensor<i32, name = "pack">) -> !atir.tensor<2xi32, name = "pack_out">
    %11 = "atir.buffer"() : () -> !atir.tensor<?x3xf32, name = "fill_out">
    %12 = atir.constant "public" @"fill_value" -> <f32, name = "fill_value", data = dense<0.000000e+00> : tensor<f32>>
    %13 = "atir.Fill"(%11, %10, %12) : (!atir.tensor<?x3xf32, name = "fill_out">, !atir.tensor<2xi32, name = "pack_out">, !atir.tensor<f32, name = "fill_value", data = dense<0.000000e+00> : tensor<f32>>) -> !atir.tensor<?x3xf32, name = "fill_out">
    %14 = "atir.buffer"() : () -> !atir.tensor<?x3xf32, name = "concat_out">
    %15 = atir.constant "public" @"concat_axis" -> <i32, name = "concat_axis", data = dense<0> : tensor<i32>>
    %16 = "atir.ConcatV2"(%14, %arg4, %13, %15) : (!atir.tensor<?x3xf32, name = "concat_out">, !atir.tensor<4x3xf32, name = "data">, !atir.tensor<?x3xf32, name = "fill_out">, !atir.tensor<i32, name = "concat_axis", data = dense<0> : tensor<i32>>) -> !atir.tensor<?x3xf32, name = "concat_out">
    %17 = "atir.buffer"() : () -> !atir.tensor<?x3xf32, name = "out_data">
    %18 = "atir.Reshape"(%17, %16, %arg5) : (!atir.tensor<?x3xf32, name = "out_data">, !atir.tensor<?x3xf32, name = "concat_out">, !atir.tensor<2xi32, name = "reshape_sizes">) -> !atir.tensor<?x3xf32, name = "out_data">
    return %8, %18 : !atir.tensor<i32, name = "out_padding">, !atir.tensor<?x3xf32, name = "out_data">
  }
}
