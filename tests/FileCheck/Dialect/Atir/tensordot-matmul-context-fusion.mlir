// RUN: annc-opt --split-input-file %s --atir-op-fusion | FileCheck %s
// FTDMM 站点（execution v2 形态）：kernel 函数首参为执行上下文，fusion.metadata
// 的 abi 为 annc_execution_v2、kernel_arg_order 为空（输出由 kernel 经上下文
// 分配），成员名单记录整块；主图区域不再重写，只标记 annc.fusion_materialized
// （AOT 裁剪丢弃主图函数）。
//
// CHECK-LABEL: func.func @main
// CHECK: "atir.MatMul"
// CHECK-SAME: annc.fusion_materialized
// CHECK-LABEL: func.func private @fused_tensordot_matmul_
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: annc.execution_mode = "aot"
// CHECK-SAME: annc.kernel
// CHECK: abi = "annc_execution_v2"
// CHECK: kernel_arg_order = []
// CHECK: member_node_names = [
// CHECK-SAME: "transpose:0"
// CHECK-SAME: "Shape:0"
// CHECK-SAME: "MatMul:0"
// CHECK-SAME: "out"
// CHECK: fusion.pattern = "tensordot_matmul"
// CHECK: atir.Customize
// CHECK-SAME: custom.arg_names = ["input", "weight"]
// CHECK-SAME: custom.arg_ranks = [3, 2]
// CHECK-SAME: custom.op_name = "TensordotMatMul"
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<1x1x32xf32, name = "x">) -> !atir.tensor<1x1x16xf32, name = "out"> {
    %perm = atir.constant "private" @"perm" -> <3xi64, name = "perm", data = dense<[0, 1, 2]> : tensor<3xi64>>
    %free = atir.constant "public" @"free" -> <2xi32, name = "free", data = dense<[0, 1]> : tensor<2xi32>>
    %axes = atir.constant "public" @"axes" -> <1xi32, name = "axes", data = dense<2> : tensor<1xi32>>
    %zero1 = atir.constant "private" @"zero1" -> <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>>
    %zero0 = atir.constant "private" @"zero0" -> <i32, name = "zero0", data = dense<0> : tensor<i32>>
    %dim_n = atir.constant "public" @"dim_n" -> <1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>
    %w = atir.constant "public" @"w" -> <32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>

    %t_buf = "atir.buffer"() : () -> !atir.tensor<1x1x32xf32, name = "transpose:0">
    %t = atir.Transpose %t_buf, %x, %perm {permutation = [0, 1, 2]} : <1x1x32xf32, name = "transpose:0">, <1x1x32xf32, name = "x">, <3xi64, name = "perm", data = dense<[0, 1, 2]> : tensor<3xi64>> -> <1x1x32xf32, name = "transpose:0">
    %shape = atir.Shape %x : (!atir.tensor<1x1x32xf32, name = "x">) -> !atir.tensor<3xi64, name = "Shape:0">
    %s32_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "Shape__:0">
    %s32 = atir.Cast %s32_buf, %shape : <3xi32, name = "Shape__:0">, <3xi64, name = "Shape:0"> -> <3xi32, name = "Shape__:0">
    %g_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "GatherV2_1:0">
    %g_axes = atir.Gather %g_axes_buf, %s32, %axes, %zero0 : <1xi32, name = "GatherV2_1:0">, <3xi32, name = "Shape__:0">, <1xi32, name = "axes", data = dense<2> : tensor<1xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <1xi32, name = "GatherV2_1:0">
    %g_free_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "GatherV2:0">
    %g_free = atir.Gather %g_free_buf, %s32, %free, %zero0 : <2xi32, name = "GatherV2:0">, <3xi32, name = "Shape__:0">, <2xi32, name = "free", data = dense<[0, 1]> : tensor<2xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <2xi32, name = "GatherV2:0">
    %p_axes_buf = "atir.buffer"() : () -> !atir.tensor<i32, name = "Prod_1:0">
    %p_axes = atir.Prod %p_axes_buf, %g_axes, %zero1 : <i32, name = "Prod_1:0">, <1xi32, name = "GatherV2_1:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <i32, name = "Prod_1:0">
    %p_free_buf = "atir.buffer"() : () -> !atir.tensor<i32, name = "Prod:0">
    %p_free = atir.Prod %p_free_buf, %g_free, %zero1 : <i32, name = "Prod:0">, <2xi32, name = "GatherV2:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <i32, name = "Prod:0">
    %e_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "stack_Unsqueeze__729:0">
    %e_axes = atir.ExpandDims %e_axes_buf, %p_axes, %zero1 : <1xi32, name = "stack_Unsqueeze__729:0">, <i32, name = "Prod_1:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <1xi32, name = "stack_Unsqueeze__729:0">
    %e_free_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "stack_Unsqueeze__727:0">
    %e_free = atir.ExpandDims %e_free_buf, %p_free, %zero1 : <1xi32, name = "stack_Unsqueeze__727:0">, <i32, name = "Prod:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <1xi32, name = "stack_Unsqueeze__727:0">
    %cat_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "stack_Concat__730:0">
    %cat = "atir.Concat"(%cat_buf, %e_free, %e_axes) <{axis = 0 : i32}> : (!atir.tensor<2xi32, name = "stack_Concat__730:0">, !atir.tensor<1xi32, name = "stack_Unsqueeze__727:0">, !atir.tensor<1xi32, name = "stack_Unsqueeze__729:0">) -> !atir.tensor<2xi32, name = "stack_Concat__730:0">
    %cat64_buf = "atir.buffer"() : () -> !atir.tensor<2xi64, name = "Reshape__731:0">
    %cat64 = atir.Cast %cat64_buf, %cat : <2xi64, name = "Reshape__731:0">, <2xi32, name = "stack_Concat__730:0"> -> <2xi64, name = "Reshape__731:0">
    %flat_buf = "atir.buffer"() : () -> !atir.tensor<?x32xf32, name = "Reshape:0">
    %flat = atir.Reshape %flat_buf, %t, %cat64 : <?x32xf32, name = "Reshape:0">, <1x1x32xf32, name = "transpose:0">, <2xi64, name = "Reshape__731:0"> -> <?x32xf32, name = "Reshape:0">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %mm = "atir.MatMul"(%mm_buf, %flat, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<?x16xf32, name = "MatMul:0">, !atir.tensor<?x32xf32, name = "Reshape:0">, !atir.tensor<32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>) -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %tail_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail = "atir.Concat"(%tail_buf, %g_free, %dim_n) <{axis = 0 : i32}> : (!atir.tensor<3xi32, name = "concat_1:0">, !atir.tensor<2xi32, name = "GatherV2:0">, !atir.tensor<1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>) -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail64_buf = "atir.buffer"() : () -> !atir.tensor<3xi64, name = "Tensordot__:0">
    %tail64 = atir.Cast %tail64_buf, %tail : <3xi64, name = "Tensordot__:0">, <3xi32, name = "concat_1:0"> -> <3xi64, name = "Tensordot__:0">
    %out_buf = "atir.buffer"() : () -> !atir.tensor<1x1x16xf32, name = "out">
    %out = atir.Reshape %out_buf, %mm, %tail64 : <1x1x16xf32, name = "out">, <?x16xf32, name = "MatMul:0">, <3xi64, name = "Tensordot__:0"> -> <1x1x16xf32, name = "out">
    return %out : !atir.tensor<1x1x16xf32, name = "out">
  }
}


// -----

// 负例：perm 非恒等（[1, 0, 2]）——ORT 侧 oracle 不检查、两侧 kernel 会算错，
// 必须不融合。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_tensordot_matmul
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<1x1x32xf32, name = "x">) -> !atir.tensor<1x1x16xf32, name = "out"> {
    %perm = atir.constant "private" @"perm" -> <3xi64, name = "perm", data = dense<[1, 0, 2]> : tensor<3xi64>>
    %free = atir.constant "public" @"free" -> <2xi32, name = "free", data = dense<[0, 1]> : tensor<2xi32>>
    %axes = atir.constant "public" @"axes" -> <1xi32, name = "axes", data = dense<2> : tensor<1xi32>>
    %zero1 = atir.constant "private" @"zero1" -> <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>>
    %zero0 = atir.constant "private" @"zero0" -> <i32, name = "zero0", data = dense<0> : tensor<i32>>
    %dim_n = atir.constant "public" @"dim_n" -> <1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>
    %w = atir.constant "public" @"w" -> <32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>

    %t_buf = "atir.buffer"() : () -> !atir.tensor<1x1x32xf32, name = "transpose:0">
    %t = atir.Transpose %t_buf, %x, %perm {permutation = [0, 1, 2]} : <1x1x32xf32, name = "transpose:0">, <1x1x32xf32, name = "x">, <3xi64, name = "perm", data = dense<[1, 0, 2]> : tensor<3xi64>> -> <1x1x32xf32, name = "transpose:0">
    %shape = atir.Shape %x : (!atir.tensor<1x1x32xf32, name = "x">) -> !atir.tensor<3xi64, name = "Shape:0">
    %s32_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "Shape__:0">
    %s32 = atir.Cast %s32_buf, %shape : <3xi32, name = "Shape__:0">, <3xi64, name = "Shape:0"> -> <3xi32, name = "Shape__:0">
    %g_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "GatherV2_1:0">
    %g_axes = atir.Gather %g_axes_buf, %s32, %axes, %zero0 : <1xi32, name = "GatherV2_1:0">, <3xi32, name = "Shape__:0">, <1xi32, name = "axes", data = dense<2> : tensor<1xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <1xi32, name = "GatherV2_1:0">
    %g_free_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "GatherV2:0">
    %g_free = atir.Gather %g_free_buf, %s32, %free, %zero0 : <2xi32, name = "GatherV2:0">, <3xi32, name = "Shape__:0">, <2xi32, name = "free", data = dense<[0, 1]> : tensor<2xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <2xi32, name = "GatherV2:0">
    %p_axes_buf = "atir.buffer"() : () -> !atir.tensor<i32, name = "Prod_1:0">
    %p_axes = atir.Prod %p_axes_buf, %g_axes, %zero1 : <i32, name = "Prod_1:0">, <1xi32, name = "GatherV2_1:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <i32, name = "Prod_1:0">
    %p_free_buf = "atir.buffer"() : () -> !atir.tensor<i32, name = "Prod:0">
    %p_free = atir.Prod %p_free_buf, %g_free, %zero1 : <i32, name = "Prod:0">, <2xi32, name = "GatherV2:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <i32, name = "Prod:0">
    %e_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "stack_Unsqueeze__729:0">
    %e_axes = atir.ExpandDims %e_axes_buf, %p_axes, %zero1 : <1xi32, name = "stack_Unsqueeze__729:0">, <i32, name = "Prod_1:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <1xi32, name = "stack_Unsqueeze__729:0">
    %e_free_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "stack_Unsqueeze__727:0">
    %e_free = atir.ExpandDims %e_free_buf, %p_free, %zero1 : <1xi32, name = "stack_Unsqueeze__727:0">, <i32, name = "Prod:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <1xi32, name = "stack_Unsqueeze__727:0">
    %cat_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "stack_Concat__730:0">
    %cat = "atir.Concat"(%cat_buf, %e_free, %e_axes) <{axis = 0 : i32}> : (!atir.tensor<2xi32, name = "stack_Concat__730:0">, !atir.tensor<1xi32, name = "stack_Unsqueeze__727:0">, !atir.tensor<1xi32, name = "stack_Unsqueeze__729:0">) -> !atir.tensor<2xi32, name = "stack_Concat__730:0">
    %cat64_buf = "atir.buffer"() : () -> !atir.tensor<2xi64, name = "Reshape__731:0">
    %cat64 = atir.Cast %cat64_buf, %cat : <2xi64, name = "Reshape__731:0">, <2xi32, name = "stack_Concat__730:0"> -> <2xi64, name = "Reshape__731:0">
    %flat_buf = "atir.buffer"() : () -> !atir.tensor<?x32xf32, name = "Reshape:0">
    %flat = atir.Reshape %flat_buf, %t, %cat64 : <?x32xf32, name = "Reshape:0">, <1x1x32xf32, name = "transpose:0">, <2xi64, name = "Reshape__731:0"> -> <?x32xf32, name = "Reshape:0">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %mm = "atir.MatMul"(%mm_buf, %flat, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<?x16xf32, name = "MatMul:0">, !atir.tensor<?x32xf32, name = "Reshape:0">, !atir.tensor<32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>) -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %tail_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail = "atir.Concat"(%tail_buf, %g_free, %dim_n) <{axis = 0 : i32}> : (!atir.tensor<3xi32, name = "concat_1:0">, !atir.tensor<2xi32, name = "GatherV2:0">, !atir.tensor<1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>) -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail64_buf = "atir.buffer"() : () -> !atir.tensor<3xi64, name = "Tensordot__:0">
    %tail64 = atir.Cast %tail64_buf, %tail : <3xi64, name = "Tensordot__:0">, <3xi32, name = "concat_1:0"> -> <3xi64, name = "Tensordot__:0">
    %out_buf = "atir.buffer"() : () -> !atir.tensor<1x1x16xf32, name = "out">
    %out = atir.Reshape %out_buf, %mm, %tail64 : <1x1x16xf32, name = "out">, <?x16xf32, name = "MatMul:0">, <3xi64, name = "Tensordot__:0"> -> <1x1x16xf32, name = "out">
    return %out : !atir.tensor<1x1x16xf32, name = "out">
  }
}

// -----

// 负例 2：同一块但模块没有 ONNX 前端标记（annc.contract_version）——
// 站点只服务 ONNX 宿主（kernel 期望 ONNX 执行上下文），TF 侧不产出。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_tensordot_matmul
module attributes {module.state = "atir"} {
  func.func @main(%x: !atir.tensor<1x1x32xf32, name = "x">) -> !atir.tensor<1x1x16xf32, name = "out"> {
    %perm = atir.constant "private" @"perm" -> <3xi64, name = "perm", data = dense<[0, 1, 2]> : tensor<3xi64>>
    %free = atir.constant "public" @"free" -> <2xi32, name = "free", data = dense<[0, 1]> : tensor<2xi32>>
    %axes = atir.constant "public" @"axes" -> <1xi32, name = "axes", data = dense<2> : tensor<1xi32>>
    %zero1 = atir.constant "private" @"zero1" -> <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>>
    %zero0 = atir.constant "private" @"zero0" -> <i32, name = "zero0", data = dense<0> : tensor<i32>>
    %dim_n = atir.constant "public" @"dim_n" -> <1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>
    %w = atir.constant "public" @"w" -> <32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>

    %t_buf = "atir.buffer"() : () -> !atir.tensor<1x1x32xf32, name = "transpose:0">
    %t = atir.Transpose %t_buf, %x, %perm {permutation = [0, 1, 2]} : <1x1x32xf32, name = "transpose:0">, <1x1x32xf32, name = "x">, <3xi64, name = "perm", data = dense<[0, 1, 2]> : tensor<3xi64>> -> <1x1x32xf32, name = "transpose:0">
    %shape = atir.Shape %x : (!atir.tensor<1x1x32xf32, name = "x">) -> !atir.tensor<3xi64, name = "Shape:0">
    %s32_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "Shape__:0">
    %s32 = atir.Cast %s32_buf, %shape : <3xi32, name = "Shape__:0">, <3xi64, name = "Shape:0"> -> <3xi32, name = "Shape__:0">
    %g_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "GatherV2_1:0">
    %g_axes = atir.Gather %g_axes_buf, %s32, %axes, %zero0 : <1xi32, name = "GatherV2_1:0">, <3xi32, name = "Shape__:0">, <1xi32, name = "axes", data = dense<2> : tensor<1xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <1xi32, name = "GatherV2_1:0">
    %g_free_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "GatherV2:0">
    %g_free = atir.Gather %g_free_buf, %s32, %free, %zero0 : <2xi32, name = "GatherV2:0">, <3xi32, name = "Shape__:0">, <2xi32, name = "free", data = dense<[0, 1]> : tensor<2xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <2xi32, name = "GatherV2:0">
    %p_axes_buf = "atir.buffer"() : () -> !atir.tensor<i32, name = "Prod_1:0">
    %p_axes = atir.Prod %p_axes_buf, %g_axes, %zero1 : <i32, name = "Prod_1:0">, <1xi32, name = "GatherV2_1:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <i32, name = "Prod_1:0">
    %p_free_buf = "atir.buffer"() : () -> !atir.tensor<i32, name = "Prod:0">
    %p_free = atir.Prod %p_free_buf, %g_free, %zero1 : <i32, name = "Prod:0">, <2xi32, name = "GatherV2:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <i32, name = "Prod:0">
    %e_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "stack_Unsqueeze__729:0">
    %e_axes = atir.ExpandDims %e_axes_buf, %p_axes, %zero1 : <1xi32, name = "stack_Unsqueeze__729:0">, <i32, name = "Prod_1:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <1xi32, name = "stack_Unsqueeze__729:0">
    %e_free_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "stack_Unsqueeze__727:0">
    %e_free = atir.ExpandDims %e_free_buf, %p_free, %zero1 : <1xi32, name = "stack_Unsqueeze__727:0">, <i32, name = "Prod:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <1xi32, name = "stack_Unsqueeze__727:0">
    %cat_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "stack_Concat__730:0">
    %cat = "atir.Concat"(%cat_buf, %e_free, %e_axes) <{axis = 0 : i32}> : (!atir.tensor<2xi32, name = "stack_Concat__730:0">, !atir.tensor<1xi32, name = "stack_Unsqueeze__727:0">, !atir.tensor<1xi32, name = "stack_Unsqueeze__729:0">) -> !atir.tensor<2xi32, name = "stack_Concat__730:0">
    %cat64_buf = "atir.buffer"() : () -> !atir.tensor<2xi64, name = "Reshape__731:0">
    %cat64 = atir.Cast %cat64_buf, %cat : <2xi64, name = "Reshape__731:0">, <2xi32, name = "stack_Concat__730:0"> -> <2xi64, name = "Reshape__731:0">
    %flat_buf = "atir.buffer"() : () -> !atir.tensor<?x32xf32, name = "Reshape:0">
    %flat = atir.Reshape %flat_buf, %t, %cat64 : <?x32xf32, name = "Reshape:0">, <1x1x32xf32, name = "transpose:0">, <2xi64, name = "Reshape__731:0"> -> <?x32xf32, name = "Reshape:0">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %mm = "atir.MatMul"(%mm_buf, %flat, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<?x16xf32, name = "MatMul:0">, !atir.tensor<?x32xf32, name = "Reshape:0">, !atir.tensor<32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>) -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %tail_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail = "atir.Concat"(%tail_buf, %g_free, %dim_n) <{axis = 0 : i32}> : (!atir.tensor<3xi32, name = "concat_1:0">, !atir.tensor<2xi32, name = "GatherV2:0">, !atir.tensor<1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>) -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail64_buf = "atir.buffer"() : () -> !atir.tensor<3xi64, name = "Tensordot__:0">
    %tail64 = atir.Cast %tail64_buf, %tail : <3xi64, name = "Tensordot__:0">, <3xi32, name = "concat_1:0"> -> <3xi64, name = "Tensordot__:0">
    %out_buf = "atir.buffer"() : () -> !atir.tensor<1x1x16xf32, name = "out">
    %out = atir.Reshape %out_buf, %mm, %tail64 : <1x1x16xf32, name = "out">, <?x16xf32, name = "MatMul:0">, <3xi64, name = "Tensordot__:0"> -> <1x1x16xf32, name = "out">
    return %out : !atir.tensor<1x1x16xf32, name = "out">
  }
}
