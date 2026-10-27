// RUN: annc-opt --split-input-file %s --atir-op-fusion | FileCheck %s
// RUN: env ANNC_EP_FUSION_TENSORDOT_MATMUL=0 annc-opt --split-input-file %s --atir-op-fusion | FileCheck %s --check-prefix=DISABLED
// DISABLED: "atir.MatMul"
// DISABLED-NOT: fusion.pattern = "tensordot_matmul"
// DISABLED-NOT: custom.op_name = "TensordotMatMul"
// DISABLED-NOT: func.func private @fused_tensordot_matmul_
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

// 正例 2：展平输入无 Transpose（直接来自上游 3D 张量，ONNX 动态 shape 图的
// 常见形态：reshape 直接吃 Concat/BiasAdd 输出）。恒等 Transpose 是数据
// 布局 no-op，其缺失不得改变融合资格。
// CHECK-LABEL: func.func @main
// CHECK: "atir.MatMul"
// CHECK-SAME: annc.fusion_materialized
// CHECK: func.func private @fused_tensordot_matmul_
// CHECK: fusion.pattern = "tensordot_matmul"
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<1x1x32xf32, name = "x">) -> !atir.tensor<1x1x16xf32, name = "out"> {
    %free = atir.constant "public" @"free" -> <2xi32, name = "free", data = dense<[0, 1]> : tensor<2xi32>>
    %axes = atir.constant "public" @"axes" -> <1xi32, name = "axes", data = dense<2> : tensor<1xi32>>
    %zero1 = atir.constant "private" @"zero1" -> <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>>
    %zero0 = atir.constant "private" @"zero0" -> <i32, name = "zero0", data = dense<0> : tensor<i32>>
    %dim_n = atir.constant "public" @"dim_n" -> <1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>
    %w = atir.constant "public" @"w" -> <32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>

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
    %flat = atir.Reshape %flat_buf, %x, %cat64 : <?x32xf32, name = "Reshape:0">, <1x1x32xf32, name = "x">, <2xi64, name = "Reshape__731:0"> -> <?x32xf32, name = "Reshape:0">
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

// 正例 3：free axes 分量直接用 Prod{keep_dims = true}（ReduceProd
// keepdims=1 的直译形态，直接产出秩 1 张量，无后续 ExpandDims），contract
// axes 仍为 Prod + ExpandDims 旧形态——两种等价形态可在同一形状 Concat 内
// 混用，keep_dims 分支不改变融合资格。
// CHECK-LABEL: func.func @main
// CHECK: "atir.MatMul"
// CHECK-SAME: annc.fusion_materialized
// CHECK: func.func private @fused_tensordot_matmul_
// CHECK: fusion.pattern = "tensordot_matmul"
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<1x1x32xf32, name = "x">) -> !atir.tensor<1x1x16xf32, name = "out"> {
    %free = atir.constant "public" @"free" -> <2xi32, name = "free", data = dense<[0, 1]> : tensor<2xi32>>
    %axes = atir.constant "public" @"axes" -> <1xi32, name = "axes", data = dense<2> : tensor<1xi32>>
    %zero1 = atir.constant "private" @"zero1" -> <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>>
    %zero0 = atir.constant "private" @"zero0" -> <i32, name = "zero0", data = dense<0> : tensor<i32>>
    %dim_n = atir.constant "public" @"dim_n" -> <1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>
    %w = atir.constant "public" @"w" -> <32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>

    %shape = atir.Shape %x : (!atir.tensor<1x1x32xf32, name = "x">) -> !atir.tensor<3xi64, name = "Shape:0">
    %s32_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "Shape__:0">
    %s32 = atir.Cast %s32_buf, %shape : <3xi32, name = "Shape__:0">, <3xi64, name = "Shape:0"> -> <3xi32, name = "Shape__:0">
    %g_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "GatherV2_1:0">
    %g_axes = atir.Gather %g_axes_buf, %s32, %axes, %zero0 : <1xi32, name = "GatherV2_1:0">, <3xi32, name = "Shape__:0">, <1xi32, name = "axes", data = dense<2> : tensor<1xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <1xi32, name = "GatherV2_1:0">
    %g_free_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "GatherV2:0">
    %g_free = atir.Gather %g_free_buf, %s32, %free, %zero0 : <2xi32, name = "GatherV2:0">, <3xi32, name = "Shape__:0">, <2xi32, name = "free", data = dense<[0, 1]> : tensor<2xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <2xi32, name = "GatherV2:0">
    %p_axes_buf = "atir.buffer"() : () -> !atir.tensor<i32, name = "Prod_1:0">
    %p_axes = atir.Prod %p_axes_buf, %g_axes, %zero1 : <i32, name = "Prod_1:0">, <1xi32, name = "GatherV2_1:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <i32, name = "Prod_1:0">
    %p_free_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "Prod:0">
    %p_free = atir.Prod %p_free_buf, %g_free, %zero1 {keep_dims = true} : <1xi32, name = "Prod:0">, <2xi32, name = "GatherV2:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <1xi32, name = "Prod:0">
    %e_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "stack_Unsqueeze__729:0">
    %e_axes = atir.ExpandDims %e_axes_buf, %p_axes, %zero1 : <1xi32, name = "stack_Unsqueeze__729:0">, <i32, name = "Prod_1:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <1xi32, name = "stack_Unsqueeze__729:0">
    %cat_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "stack_Concat__730:0">
    %cat = "atir.Concat"(%cat_buf, %p_free, %e_axes) <{axis = 0 : i32}> : (!atir.tensor<2xi32, name = "stack_Concat__730:0">, !atir.tensor<1xi32, name = "Prod:0">, !atir.tensor<1xi32, name = "stack_Unsqueeze__729:0">) -> !atir.tensor<2xi32, name = "stack_Concat__730:0">
    %cat64_buf = "atir.buffer"() : () -> !atir.tensor<2xi64, name = "Reshape__731:0">
    %cat64 = atir.Cast %cat64_buf, %cat : <2xi64, name = "Reshape__731:0">, <2xi32, name = "stack_Concat__730:0"> -> <2xi64, name = "Reshape__731:0">
    %flat_buf = "atir.buffer"() : () -> !atir.tensor<?x32xf32, name = "Reshape:0">
    %flat = atir.Reshape %flat_buf, %x, %cat64 : <?x32xf32, name = "Reshape:0">, <1x1x32xf32, name = "x">, <2xi64, name = "Reshape__731:0"> -> <?x32xf32, name = "Reshape:0">
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

// 正例 4：contract axes 分量直接用 Prod{keep_dims = true}，free axes 仍为
// Prod + ExpandDims 旧形态——与正例 3 互补，覆盖形状 Concat 两个分量各自的
// keep_dims 匹配分支。
// CHECK-LABEL: func.func @main
// CHECK: "atir.MatMul"
// CHECK-SAME: annc.fusion_materialized
// CHECK: func.func private @fused_tensordot_matmul_
// CHECK: fusion.pattern = "tensordot_matmul"
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<1x1x32xf32, name = "x">) -> !atir.tensor<1x1x16xf32, name = "out"> {
    %free = atir.constant "public" @"free" -> <2xi32, name = "free", data = dense<[0, 1]> : tensor<2xi32>>
    %axes = atir.constant "public" @"axes" -> <1xi32, name = "axes", data = dense<2> : tensor<1xi32>>
    %zero1 = atir.constant "private" @"zero1" -> <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>>
    %zero0 = atir.constant "private" @"zero0" -> <i32, name = "zero0", data = dense<0> : tensor<i32>>
    %dim_n = atir.constant "public" @"dim_n" -> <1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>
    %w = atir.constant "public" @"w" -> <32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>

    %shape = atir.Shape %x : (!atir.tensor<1x1x32xf32, name = "x">) -> !atir.tensor<3xi64, name = "Shape:0">
    %s32_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "Shape__:0">
    %s32 = atir.Cast %s32_buf, %shape : <3xi32, name = "Shape__:0">, <3xi64, name = "Shape:0"> -> <3xi32, name = "Shape__:0">
    %g_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "GatherV2_1:0">
    %g_axes = atir.Gather %g_axes_buf, %s32, %axes, %zero0 : <1xi32, name = "GatherV2_1:0">, <3xi32, name = "Shape__:0">, <1xi32, name = "axes", data = dense<2> : tensor<1xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <1xi32, name = "GatherV2_1:0">
    %g_free_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "GatherV2:0">
    %g_free = atir.Gather %g_free_buf, %s32, %free, %zero0 : <2xi32, name = "GatherV2:0">, <3xi32, name = "Shape__:0">, <2xi32, name = "free", data = dense<[0, 1]> : tensor<2xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <2xi32, name = "GatherV2:0">
    %p_free_buf = "atir.buffer"() : () -> !atir.tensor<i32, name = "Prod:0">
    %p_free = atir.Prod %p_free_buf, %g_free, %zero1 : <i32, name = "Prod:0">, <2xi32, name = "GatherV2:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <i32, name = "Prod:0">
    %e_free_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "stack_Unsqueeze__727:0">
    %e_free = atir.ExpandDims %e_free_buf, %p_free, %zero1 : <1xi32, name = "stack_Unsqueeze__727:0">, <i32, name = "Prod:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <1xi32, name = "stack_Unsqueeze__727:0">
    %p_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "Prod_1:0">
    %p_axes = atir.Prod %p_axes_buf, %g_axes, %zero1 {keep_dims = true} : <1xi32, name = "Prod_1:0">, <1xi32, name = "GatherV2_1:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <1xi32, name = "Prod_1:0">
    %cat_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "stack_Concat__730:0">
    %cat = "atir.Concat"(%cat_buf, %e_free, %p_axes) <{axis = 0 : i32}> : (!atir.tensor<2xi32, name = "stack_Concat__730:0">, !atir.tensor<1xi32, name = "stack_Unsqueeze__727:0">, !atir.tensor<1xi32, name = "Prod_1:0">) -> !atir.tensor<2xi32, name = "stack_Concat__730:0">
    %cat64_buf = "atir.buffer"() : () -> !atir.tensor<2xi64, name = "Reshape__731:0">
    %cat64 = atir.Cast %cat64_buf, %cat : <2xi64, name = "Reshape__731:0">, <2xi32, name = "stack_Concat__730:0"> -> <2xi64, name = "Reshape__731:0">
    %flat_buf = "atir.buffer"() : () -> !atir.tensor<?x32xf32, name = "Reshape:0">
    %flat = atir.Reshape %flat_buf, %x, %cat64 : <?x32xf32, name = "Reshape:0">, <1x1x32xf32, name = "x">, <2xi64, name = "Reshape__731:0"> -> <?x32xf32, name = "Reshape:0">
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

// -----

// 正例 5：rank-3 输入的非退化前导维（4x5 而非 1x1）。free 轴仍是前导维
// [0,1]、contract 轴是末维 2；展平目标 [prod(free)=20, 32]、反展平
// [4, 5, 16] 都由 shape 链在运行期算出，站点资格与单位维无关。
// CHECK-LABEL: func.func @main
// CHECK: "atir.MatMul"
// CHECK-SAME: annc.fusion_materialized
// CHECK: func.func private @fused_tensordot_matmul_
// CHECK-SAME: args = [{dtype = "f32", rank = 3 : i64, role = "dynamic", shape = [4, 5, 32]
// CHECK: fusion.pattern = "tensordot_matmul"
// CHECK: atir.Customize
// CHECK-SAME: custom.op_name = "TensordotMatMul"
// CHECK-SAME: custom.result_ranks = [3]
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<4x5x32xf32, name = "x">) -> !atir.tensor<4x5x16xf32, name = "out"> {
    %perm = atir.constant "private" @"perm" -> <3xi64, name = "perm", data = dense<[0,1,2]> : tensor<3xi64>>
    %free = atir.constant "public" @"free" -> <2xi32, name = "free", data = dense<[0,1]> : tensor<2xi32>>
    %axes = atir.constant "public" @"axes" -> <1xi32, name = "axes", data = dense<[2]> : tensor<1xi32>>
    %zero1 = atir.constant "private" @"zero1" -> <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>>
    %zero0 = atir.constant "private" @"zero0" -> <i32, name = "zero0", data = dense<0> : tensor<i32>>
    %dim_n = atir.constant "public" @"dim_n" -> <1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>
    %w = atir.constant "public" @"w" -> <32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>

    %t_buf = "atir.buffer"() : () -> !atir.tensor<4x5x32xf32, name = "transpose:0">
    %t = atir.Transpose %t_buf, %x, %perm {permutation = [0,1,2]} : <4x5x32xf32, name = "transpose:0">, <4x5x32xf32, name = "x">, <3xi64, name = "perm", data = dense<[0,1,2]> : tensor<3xi64>> -> <4x5x32xf32, name = "transpose:0">
    %shape = atir.Shape %x : (!atir.tensor<4x5x32xf32, name = "x">) -> !atir.tensor<3xi64, name = "Shape:0">
    %s32_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "Shape__:0">
    %s32 = atir.Cast %s32_buf, %shape : <3xi32, name = "Shape__:0">, <3xi64, name = "Shape:0"> -> <3xi32, name = "Shape__:0">
    %g_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "GatherV2_1:0">
    %g_axes = atir.Gather %g_axes_buf, %s32, %axes, %zero0 : <1xi32, name = "GatherV2_1:0">, <3xi32, name = "Shape__:0">, <1xi32, name = "axes", data = dense<[2]> : tensor<1xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <1xi32, name = "GatherV2_1:0">
    %g_free_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "GatherV2:0">
    %g_free = atir.Gather %g_free_buf, %s32, %free, %zero0 : <2xi32, name = "GatherV2:0">, <3xi32, name = "Shape__:0">, <2xi32, name = "free", data = dense<[0,1]> : tensor<2xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <2xi32, name = "GatherV2:0">
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
    %flat = atir.Reshape %flat_buf, %t, %cat64 : <?x32xf32, name = "Reshape:0">, <4x5x32xf32, name = "transpose:0">, <2xi64, name = "Reshape__731:0"> -> <?x32xf32, name = "Reshape:0">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %mm = "atir.MatMul"(%mm_buf, %flat, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<?x16xf32, name = "MatMul:0">, !atir.tensor<?x32xf32, name = "Reshape:0">, !atir.tensor<32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>) -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %tail_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail = "atir.Concat"(%tail_buf, %g_free, %dim_n) <{axis = 0 : i32}> : (!atir.tensor<3xi32, name = "concat_1:0">, !atir.tensor<2xi32, name = "GatherV2:0">, !atir.tensor<1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>) -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail64_buf = "atir.buffer"() : () -> !atir.tensor<3xi64, name = "Tensordot__:0">
    %tail64 = atir.Cast %tail64_buf, %tail : <3xi64, name = "Tensordot__:0">, <3xi32, name = "concat_1:0"> -> <3xi64, name = "Tensordot__:0">
    %out_buf = "atir.buffer"() : () -> !atir.tensor<4x5x16xf32, name = "out">
    %out = atir.Reshape %out_buf, %mm, %tail64 : <4x5x16xf32, name = "out">, <?x16xf32, name = "MatMul:0">, <3xi64, name = "Tensordot__:0"> -> <4x5x16xf32, name = "out">
    return %out : !atir.tensor<4x5x16xf32, name = "out">
  }
}

// -----

// 负例：展平输入 rank 2。matcher 固定 kRank = 3（free 轴恰为前导维、
// contract 轴为末维）；秩不是 3 的 tensordot 站点不融，留给通用路径。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_tensordot_matmul
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<8x32xf32, name = "x">) -> !atir.tensor<8x16xf32, name = "out"> {
    %perm = atir.constant "private" @"perm" -> <2xi64, name = "perm", data = dense<[0,1]> : tensor<2xi64>>
    %free = atir.constant "public" @"free" -> <1xi32, name = "free", data = dense<[0]> : tensor<1xi32>>
    %axes = atir.constant "public" @"axes" -> <1xi32, name = "axes", data = dense<[1]> : tensor<1xi32>>
    %zero1 = atir.constant "private" @"zero1" -> <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>>
    %zero0 = atir.constant "private" @"zero0" -> <i32, name = "zero0", data = dense<0> : tensor<i32>>
    %dim_n = atir.constant "public" @"dim_n" -> <1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>
    %w = atir.constant "public" @"w" -> <32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>

    %t_buf = "atir.buffer"() : () -> !atir.tensor<8x32xf32, name = "transpose:0">
    %t = atir.Transpose %t_buf, %x, %perm {permutation = [0,1]} : <8x32xf32, name = "transpose:0">, <8x32xf32, name = "x">, <2xi64, name = "perm", data = dense<[0,1]> : tensor<2xi64>> -> <8x32xf32, name = "transpose:0">
    %shape = atir.Shape %x : (!atir.tensor<8x32xf32, name = "x">) -> !atir.tensor<2xi64, name = "Shape:0">
    %s32_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "Shape__:0">
    %s32 = atir.Cast %s32_buf, %shape : <2xi32, name = "Shape__:0">, <2xi64, name = "Shape:0"> -> <2xi32, name = "Shape__:0">
    %g_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "GatherV2_1:0">
    %g_axes = atir.Gather %g_axes_buf, %s32, %axes, %zero0 : <1xi32, name = "GatherV2_1:0">, <2xi32, name = "Shape__:0">, <1xi32, name = "axes", data = dense<[1]> : tensor<1xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <1xi32, name = "GatherV2_1:0">
    %g_free_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "GatherV2:0">
    %g_free = atir.Gather %g_free_buf, %s32, %free, %zero0 : <1xi32, name = "GatherV2:0">, <2xi32, name = "Shape__:0">, <1xi32, name = "free", data = dense<[0]> : tensor<1xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <1xi32, name = "GatherV2:0">
    %p_axes_buf = "atir.buffer"() : () -> !atir.tensor<i32, name = "Prod_1:0">
    %p_axes = atir.Prod %p_axes_buf, %g_axes, %zero1 : <i32, name = "Prod_1:0">, <1xi32, name = "GatherV2_1:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <i32, name = "Prod_1:0">
    %p_free_buf = "atir.buffer"() : () -> !atir.tensor<i32, name = "Prod:0">
    %p_free = atir.Prod %p_free_buf, %g_free, %zero1 : <i32, name = "Prod:0">, <1xi32, name = "GatherV2:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <i32, name = "Prod:0">
    %e_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "stack_Unsqueeze__729:0">
    %e_axes = atir.ExpandDims %e_axes_buf, %p_axes, %zero1 : <1xi32, name = "stack_Unsqueeze__729:0">, <i32, name = "Prod_1:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <1xi32, name = "stack_Unsqueeze__729:0">
    %e_free_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "stack_Unsqueeze__727:0">
    %e_free = atir.ExpandDims %e_free_buf, %p_free, %zero1 : <1xi32, name = "stack_Unsqueeze__727:0">, <i32, name = "Prod:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <1xi32, name = "stack_Unsqueeze__727:0">
    %cat_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "stack_Concat__730:0">
    %cat = "atir.Concat"(%cat_buf, %e_free, %e_axes) <{axis = 0 : i32}> : (!atir.tensor<2xi32, name = "stack_Concat__730:0">, !atir.tensor<1xi32, name = "stack_Unsqueeze__727:0">, !atir.tensor<1xi32, name = "stack_Unsqueeze__729:0">) -> !atir.tensor<2xi32, name = "stack_Concat__730:0">
    %cat64_buf = "atir.buffer"() : () -> !atir.tensor<2xi64, name = "Reshape__731:0">
    %cat64 = atir.Cast %cat64_buf, %cat : <2xi64, name = "Reshape__731:0">, <2xi32, name = "stack_Concat__730:0"> -> <2xi64, name = "Reshape__731:0">
    %flat_buf = "atir.buffer"() : () -> !atir.tensor<?x32xf32, name = "Reshape:0">
    %flat = atir.Reshape %flat_buf, %t, %cat64 : <?x32xf32, name = "Reshape:0">, <8x32xf32, name = "transpose:0">, <2xi64, name = "Reshape__731:0"> -> <?x32xf32, name = "Reshape:0">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %mm = "atir.MatMul"(%mm_buf, %flat, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<?x16xf32, name = "MatMul:0">, !atir.tensor<?x32xf32, name = "Reshape:0">, !atir.tensor<32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>) -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %tail_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "concat_1:0">
    %tail = "atir.Concat"(%tail_buf, %g_free, %dim_n) <{axis = 0 : i32}> : (!atir.tensor<2xi32, name = "concat_1:0">, !atir.tensor<1xi32, name = "GatherV2:0">, !atir.tensor<1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>) -> !atir.tensor<2xi32, name = "concat_1:0">
    %tail64_buf = "atir.buffer"() : () -> !atir.tensor<2xi64, name = "Tensordot__:0">
    %tail64 = atir.Cast %tail64_buf, %tail : <2xi64, name = "Tensordot__:0">, <2xi32, name = "concat_1:0"> -> <2xi64, name = "Tensordot__:0">
    %out_buf = "atir.buffer"() : () -> !atir.tensor<8x16xf32, name = "out">
    %out = atir.Reshape %out_buf, %mm, %tail64 : <8x16xf32, name = "out">, <?x16xf32, name = "MatMul:0">, <2xi64, name = "Tensordot__:0"> -> <8x16xf32, name = "out">
    return %out : !atir.tensor<8x16xf32, name = "out">
  }
}

// -----

// 负例：展平输入 rank 4。与 rank-2 同理，matcher 只接受 rank-3 的
// [free..., contract] 展平形态；rank-4 站点不融。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_tensordot_matmul
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<2x3x4x32xf32, name = "x">) -> !atir.tensor<2x3x4x16xf32, name = "out"> {
    %perm = atir.constant "private" @"perm" -> <4xi64, name = "perm", data = dense<[0,1,2,3]> : tensor<4xi64>>
    %free = atir.constant "public" @"free" -> <3xi32, name = "free", data = dense<[0,1,2]> : tensor<3xi32>>
    %axes = atir.constant "public" @"axes" -> <1xi32, name = "axes", data = dense<[3]> : tensor<1xi32>>
    %zero1 = atir.constant "private" @"zero1" -> <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>>
    %zero0 = atir.constant "private" @"zero0" -> <i32, name = "zero0", data = dense<0> : tensor<i32>>
    %dim_n = atir.constant "public" @"dim_n" -> <1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>
    %w = atir.constant "public" @"w" -> <32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>

    %t_buf = "atir.buffer"() : () -> !atir.tensor<2x3x4x32xf32, name = "transpose:0">
    %t = atir.Transpose %t_buf, %x, %perm {permutation = [0,1,2,3]} : <2x3x4x32xf32, name = "transpose:0">, <2x3x4x32xf32, name = "x">, <4xi64, name = "perm", data = dense<[0,1,2,3]> : tensor<4xi64>> -> <2x3x4x32xf32, name = "transpose:0">
    %shape = atir.Shape %x : (!atir.tensor<2x3x4x32xf32, name = "x">) -> !atir.tensor<4xi64, name = "Shape:0">
    %s32_buf = "atir.buffer"() : () -> !atir.tensor<4xi32, name = "Shape__:0">
    %s32 = atir.Cast %s32_buf, %shape : <4xi32, name = "Shape__:0">, <4xi64, name = "Shape:0"> -> <4xi32, name = "Shape__:0">
    %g_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "GatherV2_1:0">
    %g_axes = atir.Gather %g_axes_buf, %s32, %axes, %zero0 : <1xi32, name = "GatherV2_1:0">, <4xi32, name = "Shape__:0">, <1xi32, name = "axes", data = dense<[3]> : tensor<1xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <1xi32, name = "GatherV2_1:0">
    %g_free_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "GatherV2:0">
    %g_free = atir.Gather %g_free_buf, %s32, %free, %zero0 : <3xi32, name = "GatherV2:0">, <4xi32, name = "Shape__:0">, <3xi32, name = "free", data = dense<[0,1,2]> : tensor<3xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <3xi32, name = "GatherV2:0">
    %p_axes_buf = "atir.buffer"() : () -> !atir.tensor<i32, name = "Prod_1:0">
    %p_axes = atir.Prod %p_axes_buf, %g_axes, %zero1 : <i32, name = "Prod_1:0">, <1xi32, name = "GatherV2_1:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <i32, name = "Prod_1:0">
    %p_free_buf = "atir.buffer"() : () -> !atir.tensor<i32, name = "Prod:0">
    %p_free = atir.Prod %p_free_buf, %g_free, %zero1 : <i32, name = "Prod:0">, <3xi32, name = "GatherV2:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <i32, name = "Prod:0">
    %e_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "stack_Unsqueeze__729:0">
    %e_axes = atir.ExpandDims %e_axes_buf, %p_axes, %zero1 : <1xi32, name = "stack_Unsqueeze__729:0">, <i32, name = "Prod_1:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <1xi32, name = "stack_Unsqueeze__729:0">
    %e_free_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "stack_Unsqueeze__727:0">
    %e_free = atir.ExpandDims %e_free_buf, %p_free, %zero1 : <1xi32, name = "stack_Unsqueeze__727:0">, <i32, name = "Prod:0">, <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>> -> <1xi32, name = "stack_Unsqueeze__727:0">
    %cat_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "stack_Concat__730:0">
    %cat = "atir.Concat"(%cat_buf, %e_free, %e_axes) <{axis = 0 : i32}> : (!atir.tensor<2xi32, name = "stack_Concat__730:0">, !atir.tensor<1xi32, name = "stack_Unsqueeze__727:0">, !atir.tensor<1xi32, name = "stack_Unsqueeze__729:0">) -> !atir.tensor<2xi32, name = "stack_Concat__730:0">
    %cat64_buf = "atir.buffer"() : () -> !atir.tensor<2xi64, name = "Reshape__731:0">
    %cat64 = atir.Cast %cat64_buf, %cat : <2xi64, name = "Reshape__731:0">, <2xi32, name = "stack_Concat__730:0"> -> <2xi64, name = "Reshape__731:0">
    %flat_buf = "atir.buffer"() : () -> !atir.tensor<?x32xf32, name = "Reshape:0">
    %flat = atir.Reshape %flat_buf, %t, %cat64 : <?x32xf32, name = "Reshape:0">, <2x3x4x32xf32, name = "transpose:0">, <2xi64, name = "Reshape__731:0"> -> <?x32xf32, name = "Reshape:0">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %mm = "atir.MatMul"(%mm_buf, %flat, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<?x16xf32, name = "MatMul:0">, !atir.tensor<?x32xf32, name = "Reshape:0">, !atir.tensor<32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>) -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %tail_buf = "atir.buffer"() : () -> !atir.tensor<4xi32, name = "concat_1:0">
    %tail = "atir.Concat"(%tail_buf, %g_free, %dim_n) <{axis = 0 : i32}> : (!atir.tensor<4xi32, name = "concat_1:0">, !atir.tensor<3xi32, name = "GatherV2:0">, !atir.tensor<1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>) -> !atir.tensor<4xi32, name = "concat_1:0">
    %tail64_buf = "atir.buffer"() : () -> !atir.tensor<4xi64, name = "Tensordot__:0">
    %tail64 = atir.Cast %tail64_buf, %tail : <4xi64, name = "Tensordot__:0">, <4xi32, name = "concat_1:0"> -> <4xi64, name = "Tensordot__:0">
    %out_buf = "atir.buffer"() : () -> !atir.tensor<2x3x4x16xf32, name = "out">
    %out = atir.Reshape %out_buf, %mm, %tail64 : <2x3x4x16xf32, name = "out">, <?x16xf32, name = "MatMul:0">, <4xi64, name = "Tensordot__:0"> -> <2x3x4x16xf32, name = "out">
    return %out : !atir.tensor<2x3x4x16xf32, name = "out">
  }
}

// -----

// 负例：free 轴非前导（free = [0,2]、contract 轴 = 1）。x 末维 4 与权重
// contract 维 4 一致，所以 rejection 不是末维形状失配，而是展平形状链的
// free 轴序列检查——ANNC 要求 free 轴恰为前导维 [0,1]（OpFusion.cpp 的
// isFtdmmFlattenShapeTarget），否则 kernel 按连续 row-major 读取会算错。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_tensordot_matmul
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<2x4x4xf32, name = "x">) -> !atir.tensor<2x4x16xf32, name = "out"> {
    %perm = atir.constant "private" @"perm" -> <3xi64, name = "perm", data = dense<[0,1,2]> : tensor<3xi64>>
    %free = atir.constant "public" @"free" -> <2xi32, name = "free", data = dense<[0,2]> : tensor<2xi32>>
    %axes = atir.constant "public" @"axes" -> <1xi32, name = "axes", data = dense<[1]> : tensor<1xi32>>
    %zero1 = atir.constant "private" @"zero1" -> <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>>
    %zero0 = atir.constant "private" @"zero0" -> <i32, name = "zero0", data = dense<0> : tensor<i32>>
    %dim_n = atir.constant "public" @"dim_n" -> <1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>
    %w = atir.constant "public" @"w" -> <4x16xf32, name = "w", data = dense<0.100000> : tensor<4x16xf32>>

    %t_buf = "atir.buffer"() : () -> !atir.tensor<2x4x4xf32, name = "transpose:0">
    %t = atir.Transpose %t_buf, %x, %perm {permutation = [0,1,2]} : <2x4x4xf32, name = "transpose:0">, <2x4x4xf32, name = "x">, <3xi64, name = "perm", data = dense<[0,1,2]> : tensor<3xi64>> -> <2x4x4xf32, name = "transpose:0">
    %shape = atir.Shape %x : (!atir.tensor<2x4x4xf32, name = "x">) -> !atir.tensor<3xi64, name = "Shape:0">
    %s32_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "Shape__:0">
    %s32 = atir.Cast %s32_buf, %shape : <3xi32, name = "Shape__:0">, <3xi64, name = "Shape:0"> -> <3xi32, name = "Shape__:0">
    %g_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "GatherV2_1:0">
    %g_axes = atir.Gather %g_axes_buf, %s32, %axes, %zero0 : <1xi32, name = "GatherV2_1:0">, <3xi32, name = "Shape__:0">, <1xi32, name = "axes", data = dense<[1]> : tensor<1xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <1xi32, name = "GatherV2_1:0">
    %g_free_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "GatherV2:0">
    %g_free = atir.Gather %g_free_buf, %s32, %free, %zero0 : <2xi32, name = "GatherV2:0">, <3xi32, name = "Shape__:0">, <2xi32, name = "free", data = dense<[0,2]> : tensor<2xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <2xi32, name = "GatherV2:0">
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
    %flat_buf = "atir.buffer"() : () -> !atir.tensor<?x4xf32, name = "Reshape:0">
    %flat = atir.Reshape %flat_buf, %t, %cat64 : <?x4xf32, name = "Reshape:0">, <2x4x4xf32, name = "transpose:0">, <2xi64, name = "Reshape__731:0"> -> <?x4xf32, name = "Reshape:0">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %mm = "atir.MatMul"(%mm_buf, %flat, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<?x16xf32, name = "MatMul:0">, !atir.tensor<?x4xf32, name = "Reshape:0">, !atir.tensor<4x16xf32, name = "w", data = dense<0.100000> : tensor<4x16xf32>>) -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %tail_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail = "atir.Concat"(%tail_buf, %g_free, %dim_n) <{axis = 0 : i32}> : (!atir.tensor<3xi32, name = "concat_1:0">, !atir.tensor<2xi32, name = "GatherV2:0">, !atir.tensor<1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>) -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail64_buf = "atir.buffer"() : () -> !atir.tensor<3xi64, name = "Tensordot__:0">
    %tail64 = atir.Cast %tail64_buf, %tail : <3xi64, name = "Tensordot__:0">, <3xi32, name = "concat_1:0"> -> <3xi64, name = "Tensordot__:0">
    %out_buf = "atir.buffer"() : () -> !atir.tensor<2x4x16xf32, name = "out">
    %out = atir.Reshape %out_buf, %mm, %tail64 : <2x4x16xf32, name = "out">, <?x16xf32, name = "MatMul:0">, <3xi64, name = "Tensordot__:0"> -> <2x4x16xf32, name = "out">
    return %out : !atir.tensor<2x4x16xf32, name = "out">
  }
}


// -----

// 负例：free 轴顺序反转（free = [1,0]、contract 轴仍是末维 2）。这是
// ANNC 相对 ORT 单独收紧的一条：ORT 只校验 contract 轴是末维、free 取
// Gather 常量原值，不检查顺序；展平后行序会与 row-major 读取不一致，
// 必须不融。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_tensordot_matmul
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<2x4x8xf32, name = "x">) -> !atir.tensor<4x2x16xf32, name = "out"> {
    %perm = atir.constant "private" @"perm" -> <3xi64, name = "perm", data = dense<[0,1,2]> : tensor<3xi64>>
    %free = atir.constant "public" @"free" -> <2xi32, name = "free", data = dense<[1,0]> : tensor<2xi32>>
    %axes = atir.constant "public" @"axes" -> <1xi32, name = "axes", data = dense<[2]> : tensor<1xi32>>
    %zero1 = atir.constant "private" @"zero1" -> <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>>
    %zero0 = atir.constant "private" @"zero0" -> <i32, name = "zero0", data = dense<0> : tensor<i32>>
    %dim_n = atir.constant "public" @"dim_n" -> <1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>
    %w = atir.constant "public" @"w" -> <8x16xf32, name = "w", data = dense<0.100000> : tensor<8x16xf32>>

    %t_buf = "atir.buffer"() : () -> !atir.tensor<2x4x8xf32, name = "transpose:0">
    %t = atir.Transpose %t_buf, %x, %perm {permutation = [0,1,2]} : <2x4x8xf32, name = "transpose:0">, <2x4x8xf32, name = "x">, <3xi64, name = "perm", data = dense<[0,1,2]> : tensor<3xi64>> -> <2x4x8xf32, name = "transpose:0">
    %shape = atir.Shape %x : (!atir.tensor<2x4x8xf32, name = "x">) -> !atir.tensor<3xi64, name = "Shape:0">
    %s32_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "Shape__:0">
    %s32 = atir.Cast %s32_buf, %shape : <3xi32, name = "Shape__:0">, <3xi64, name = "Shape:0"> -> <3xi32, name = "Shape__:0">
    %g_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "GatherV2_1:0">
    %g_axes = atir.Gather %g_axes_buf, %s32, %axes, %zero0 : <1xi32, name = "GatherV2_1:0">, <3xi32, name = "Shape__:0">, <1xi32, name = "axes", data = dense<[2]> : tensor<1xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <1xi32, name = "GatherV2_1:0">
    %g_free_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "GatherV2:0">
    %g_free = atir.Gather %g_free_buf, %s32, %free, %zero0 : <2xi32, name = "GatherV2:0">, <3xi32, name = "Shape__:0">, <2xi32, name = "free", data = dense<[1,0]> : tensor<2xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <2xi32, name = "GatherV2:0">
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
    %flat_buf = "atir.buffer"() : () -> !atir.tensor<?x8xf32, name = "Reshape:0">
    %flat = atir.Reshape %flat_buf, %t, %cat64 : <?x8xf32, name = "Reshape:0">, <2x4x8xf32, name = "transpose:0">, <2xi64, name = "Reshape__731:0"> -> <?x8xf32, name = "Reshape:0">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %mm = "atir.MatMul"(%mm_buf, %flat, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<?x16xf32, name = "MatMul:0">, !atir.tensor<?x8xf32, name = "Reshape:0">, !atir.tensor<8x16xf32, name = "w", data = dense<0.100000> : tensor<8x16xf32>>) -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %tail_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail = "atir.Concat"(%tail_buf, %g_free, %dim_n) <{axis = 0 : i32}> : (!atir.tensor<3xi32, name = "concat_1:0">, !atir.tensor<2xi32, name = "GatherV2:0">, !atir.tensor<1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>) -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail64_buf = "atir.buffer"() : () -> !atir.tensor<3xi64, name = "Tensordot__:0">
    %tail64 = atir.Cast %tail64_buf, %tail : <3xi64, name = "Tensordot__:0">, <3xi32, name = "concat_1:0"> -> <3xi64, name = "Tensordot__:0">
    %out_buf = "atir.buffer"() : () -> !atir.tensor<4x2x16xf32, name = "out">
    %out = atir.Reshape %out_buf, %mm, %tail64 : <4x2x16xf32, name = "out">, <?x16xf32, name = "MatMul:0">, <3xi64, name = "Tensordot__:0"> -> <4x2x16xf32, name = "out">
    return %out : !atir.tensor<4x2x16xf32, name = "out">
  }
}

// -----

// 负例：left_transpose=true。FTDMM kernel 按连续 row-major 读左侧操作数，
// 转置旗标一旦保留会静默算错；matcher 显式拒绝（OpFusion.cpp 的
// getLeftTranspose/getRightTranspose/getOutputTranspose 守卫）。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_tensordot_matmul
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<4x5x32xf32, name = "x">) -> !atir.tensor<4x5x16xf32, name = "out"> {
    %perm = atir.constant "private" @"perm" -> <3xi64, name = "perm", data = dense<[0,1,2]> : tensor<3xi64>>
    %free = atir.constant "public" @"free" -> <2xi32, name = "free", data = dense<[0,1]> : tensor<2xi32>>
    %axes = atir.constant "public" @"axes" -> <1xi32, name = "axes", data = dense<[2]> : tensor<1xi32>>
    %zero1 = atir.constant "private" @"zero1" -> <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>>
    %zero0 = atir.constant "private" @"zero0" -> <i32, name = "zero0", data = dense<0> : tensor<i32>>
    %dim_n = atir.constant "public" @"dim_n" -> <1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>
    %w = atir.constant "public" @"w" -> <32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>

    %t_buf = "atir.buffer"() : () -> !atir.tensor<4x5x32xf32, name = "transpose:0">
    %t = atir.Transpose %t_buf, %x, %perm {permutation = [0,1,2]} : <4x5x32xf32, name = "transpose:0">, <4x5x32xf32, name = "x">, <3xi64, name = "perm", data = dense<[0,1,2]> : tensor<3xi64>> -> <4x5x32xf32, name = "transpose:0">
    %shape = atir.Shape %x : (!atir.tensor<4x5x32xf32, name = "x">) -> !atir.tensor<3xi64, name = "Shape:0">
    %s32_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "Shape__:0">
    %s32 = atir.Cast %s32_buf, %shape : <3xi32, name = "Shape__:0">, <3xi64, name = "Shape:0"> -> <3xi32, name = "Shape__:0">
    %g_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "GatherV2_1:0">
    %g_axes = atir.Gather %g_axes_buf, %s32, %axes, %zero0 : <1xi32, name = "GatherV2_1:0">, <3xi32, name = "Shape__:0">, <1xi32, name = "axes", data = dense<[2]> : tensor<1xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <1xi32, name = "GatherV2_1:0">
    %g_free_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "GatherV2:0">
    %g_free = atir.Gather %g_free_buf, %s32, %free, %zero0 : <2xi32, name = "GatherV2:0">, <3xi32, name = "Shape__:0">, <2xi32, name = "free", data = dense<[0,1]> : tensor<2xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <2xi32, name = "GatherV2:0">
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
    %flat = atir.Reshape %flat_buf, %t, %cat64 : <?x32xf32, name = "Reshape:0">, <4x5x32xf32, name = "transpose:0">, <2xi64, name = "Reshape__731:0"> -> <?x32xf32, name = "Reshape:0">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %mm = "atir.MatMul"(%mm_buf, %flat, %w) <{do_relu = false, left_transpose = true, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<?x16xf32, name = "MatMul:0">, !atir.tensor<?x32xf32, name = "Reshape:0">, !atir.tensor<32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>) -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %tail_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail = "atir.Concat"(%tail_buf, %g_free, %dim_n) <{axis = 0 : i32}> : (!atir.tensor<3xi32, name = "concat_1:0">, !atir.tensor<2xi32, name = "GatherV2:0">, !atir.tensor<1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>) -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail64_buf = "atir.buffer"() : () -> !atir.tensor<3xi64, name = "Tensordot__:0">
    %tail64 = atir.Cast %tail64_buf, %tail : <3xi64, name = "Tensordot__:0">, <3xi32, name = "concat_1:0"> -> <3xi64, name = "Tensordot__:0">
    %out_buf = "atir.buffer"() : () -> !atir.tensor<4x5x16xf32, name = "out">
    %out = atir.Reshape %out_buf, %mm, %tail64 : <4x5x16xf32, name = "out">, <?x16xf32, name = "MatMul:0">, <3xi64, name = "Tensordot__:0"> -> <4x5x16xf32, name = "out">
    return %out : !atir.tensor<4x5x16xf32, name = "out">
  }
}


// -----

// 负例：right_transpose=true，同一组转置守卫。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_tensordot_matmul
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<4x5x32xf32, name = "x">) -> !atir.tensor<4x5x16xf32, name = "out"> {
    %perm = atir.constant "private" @"perm" -> <3xi64, name = "perm", data = dense<[0,1,2]> : tensor<3xi64>>
    %free = atir.constant "public" @"free" -> <2xi32, name = "free", data = dense<[0,1]> : tensor<2xi32>>
    %axes = atir.constant "public" @"axes" -> <1xi32, name = "axes", data = dense<[2]> : tensor<1xi32>>
    %zero1 = atir.constant "private" @"zero1" -> <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>>
    %zero0 = atir.constant "private" @"zero0" -> <i32, name = "zero0", data = dense<0> : tensor<i32>>
    %dim_n = atir.constant "public" @"dim_n" -> <1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>
    %w = atir.constant "public" @"w" -> <32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>

    %t_buf = "atir.buffer"() : () -> !atir.tensor<4x5x32xf32, name = "transpose:0">
    %t = atir.Transpose %t_buf, %x, %perm {permutation = [0,1,2]} : <4x5x32xf32, name = "transpose:0">, <4x5x32xf32, name = "x">, <3xi64, name = "perm", data = dense<[0,1,2]> : tensor<3xi64>> -> <4x5x32xf32, name = "transpose:0">
    %shape = atir.Shape %x : (!atir.tensor<4x5x32xf32, name = "x">) -> !atir.tensor<3xi64, name = "Shape:0">
    %s32_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "Shape__:0">
    %s32 = atir.Cast %s32_buf, %shape : <3xi32, name = "Shape__:0">, <3xi64, name = "Shape:0"> -> <3xi32, name = "Shape__:0">
    %g_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "GatherV2_1:0">
    %g_axes = atir.Gather %g_axes_buf, %s32, %axes, %zero0 : <1xi32, name = "GatherV2_1:0">, <3xi32, name = "Shape__:0">, <1xi32, name = "axes", data = dense<[2]> : tensor<1xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <1xi32, name = "GatherV2_1:0">
    %g_free_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "GatherV2:0">
    %g_free = atir.Gather %g_free_buf, %s32, %free, %zero0 : <2xi32, name = "GatherV2:0">, <3xi32, name = "Shape__:0">, <2xi32, name = "free", data = dense<[0,1]> : tensor<2xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <2xi32, name = "GatherV2:0">
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
    %flat = atir.Reshape %flat_buf, %t, %cat64 : <?x32xf32, name = "Reshape:0">, <4x5x32xf32, name = "transpose:0">, <2xi64, name = "Reshape__731:0"> -> <?x32xf32, name = "Reshape:0">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %mm = "atir.MatMul"(%mm_buf, %flat, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = true, withBias = false}> : (!atir.tensor<?x16xf32, name = "MatMul:0">, !atir.tensor<?x32xf32, name = "Reshape:0">, !atir.tensor<32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>) -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %tail_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail = "atir.Concat"(%tail_buf, %g_free, %dim_n) <{axis = 0 : i32}> : (!atir.tensor<3xi32, name = "concat_1:0">, !atir.tensor<2xi32, name = "GatherV2:0">, !atir.tensor<1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>) -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail64_buf = "atir.buffer"() : () -> !atir.tensor<3xi64, name = "Tensordot__:0">
    %tail64 = atir.Cast %tail64_buf, %tail : <3xi64, name = "Tensordot__:0">, <3xi32, name = "concat_1:0"> -> <3xi64, name = "Tensordot__:0">
    %out_buf = "atir.buffer"() : () -> !atir.tensor<4x5x16xf32, name = "out">
    %out = atir.Reshape %out_buf, %mm, %tail64 : <4x5x16xf32, name = "out">, <?x16xf32, name = "MatMul:0">, <3xi64, name = "Tensordot__:0"> -> <4x5x16xf32, name = "out">
    return %out : !atir.tensor<4x5x16xf32, name = "out">
  }
}


// -----

// 负例：output_transpose=true，同一组转置守卫。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_tensordot_matmul
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<4x5x32xf32, name = "x">) -> !atir.tensor<4x5x16xf32, name = "out"> {
    %perm = atir.constant "private" @"perm" -> <3xi64, name = "perm", data = dense<[0,1,2]> : tensor<3xi64>>
    %free = atir.constant "public" @"free" -> <2xi32, name = "free", data = dense<[0,1]> : tensor<2xi32>>
    %axes = atir.constant "public" @"axes" -> <1xi32, name = "axes", data = dense<[2]> : tensor<1xi32>>
    %zero1 = atir.constant "private" @"zero1" -> <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>>
    %zero0 = atir.constant "private" @"zero0" -> <i32, name = "zero0", data = dense<0> : tensor<i32>>
    %dim_n = atir.constant "public" @"dim_n" -> <1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>
    %w = atir.constant "public" @"w" -> <32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>

    %t_buf = "atir.buffer"() : () -> !atir.tensor<4x5x32xf32, name = "transpose:0">
    %t = atir.Transpose %t_buf, %x, %perm {permutation = [0,1,2]} : <4x5x32xf32, name = "transpose:0">, <4x5x32xf32, name = "x">, <3xi64, name = "perm", data = dense<[0,1,2]> : tensor<3xi64>> -> <4x5x32xf32, name = "transpose:0">
    %shape = atir.Shape %x : (!atir.tensor<4x5x32xf32, name = "x">) -> !atir.tensor<3xi64, name = "Shape:0">
    %s32_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "Shape__:0">
    %s32 = atir.Cast %s32_buf, %shape : <3xi32, name = "Shape__:0">, <3xi64, name = "Shape:0"> -> <3xi32, name = "Shape__:0">
    %g_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "GatherV2_1:0">
    %g_axes = atir.Gather %g_axes_buf, %s32, %axes, %zero0 : <1xi32, name = "GatherV2_1:0">, <3xi32, name = "Shape__:0">, <1xi32, name = "axes", data = dense<[2]> : tensor<1xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <1xi32, name = "GatherV2_1:0">
    %g_free_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "GatherV2:0">
    %g_free = atir.Gather %g_free_buf, %s32, %free, %zero0 : <2xi32, name = "GatherV2:0">, <3xi32, name = "Shape__:0">, <2xi32, name = "free", data = dense<[0,1]> : tensor<2xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <2xi32, name = "GatherV2:0">
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
    %flat = atir.Reshape %flat_buf, %t, %cat64 : <?x32xf32, name = "Reshape:0">, <4x5x32xf32, name = "transpose:0">, <2xi64, name = "Reshape__731:0"> -> <?x32xf32, name = "Reshape:0">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %mm = "atir.MatMul"(%mm_buf, %flat, %w) <{do_relu = false, left_transpose = false, output_transpose = true, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<?x16xf32, name = "MatMul:0">, !atir.tensor<?x32xf32, name = "Reshape:0">, !atir.tensor<32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>) -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %tail_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail = "atir.Concat"(%tail_buf, %g_free, %dim_n) <{axis = 0 : i32}> : (!atir.tensor<3xi32, name = "concat_1:0">, !atir.tensor<2xi32, name = "GatherV2:0">, !atir.tensor<1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>) -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail64_buf = "atir.buffer"() : () -> !atir.tensor<3xi64, name = "Tensordot__:0">
    %tail64 = atir.Cast %tail64_buf, %tail : <3xi64, name = "Tensordot__:0">, <3xi32, name = "concat_1:0"> -> <3xi64, name = "Tensordot__:0">
    %out_buf = "atir.buffer"() : () -> !atir.tensor<4x5x16xf32, name = "out">
    %out = atir.Reshape %out_buf, %mm, %tail64 : <4x5x16xf32, name = "out">, <?x16xf32, name = "MatMul:0">, <3xi64, name = "Tensordot__:0"> -> <4x5x16xf32, name = "out">
    return %out : !atir.tensor<4x5x16xf32, name = "out">
  }
}


// -----

// 负例：MatMul 带 do_relu。站点不吸收 relu，保留给通用逐元素路径。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_tensordot_matmul
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<4x5x32xf32, name = "x">) -> !atir.tensor<4x5x16xf32, name = "out"> {
    %perm = atir.constant "private" @"perm" -> <3xi64, name = "perm", data = dense<[0,1,2]> : tensor<3xi64>>
    %free = atir.constant "public" @"free" -> <2xi32, name = "free", data = dense<[0,1]> : tensor<2xi32>>
    %axes = atir.constant "public" @"axes" -> <1xi32, name = "axes", data = dense<[2]> : tensor<1xi32>>
    %zero1 = atir.constant "private" @"zero1" -> <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>>
    %zero0 = atir.constant "private" @"zero0" -> <i32, name = "zero0", data = dense<0> : tensor<i32>>
    %dim_n = atir.constant "public" @"dim_n" -> <1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>
    %w = atir.constant "public" @"w" -> <32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>

    %t_buf = "atir.buffer"() : () -> !atir.tensor<4x5x32xf32, name = "transpose:0">
    %t = atir.Transpose %t_buf, %x, %perm {permutation = [0,1,2]} : <4x5x32xf32, name = "transpose:0">, <4x5x32xf32, name = "x">, <3xi64, name = "perm", data = dense<[0,1,2]> : tensor<3xi64>> -> <4x5x32xf32, name = "transpose:0">
    %shape = atir.Shape %x : (!atir.tensor<4x5x32xf32, name = "x">) -> !atir.tensor<3xi64, name = "Shape:0">
    %s32_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "Shape__:0">
    %s32 = atir.Cast %s32_buf, %shape : <3xi32, name = "Shape__:0">, <3xi64, name = "Shape:0"> -> <3xi32, name = "Shape__:0">
    %g_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "GatherV2_1:0">
    %g_axes = atir.Gather %g_axes_buf, %s32, %axes, %zero0 : <1xi32, name = "GatherV2_1:0">, <3xi32, name = "Shape__:0">, <1xi32, name = "axes", data = dense<[2]> : tensor<1xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <1xi32, name = "GatherV2_1:0">
    %g_free_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "GatherV2:0">
    %g_free = atir.Gather %g_free_buf, %s32, %free, %zero0 : <2xi32, name = "GatherV2:0">, <3xi32, name = "Shape__:0">, <2xi32, name = "free", data = dense<[0,1]> : tensor<2xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <2xi32, name = "GatherV2:0">
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
    %flat = atir.Reshape %flat_buf, %t, %cat64 : <?x32xf32, name = "Reshape:0">, <4x5x32xf32, name = "transpose:0">, <2xi64, name = "Reshape__731:0"> -> <?x32xf32, name = "Reshape:0">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %mm = "atir.MatMul"(%mm_buf, %flat, %w) <{do_relu = true, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<?x16xf32, name = "MatMul:0">, !atir.tensor<?x32xf32, name = "Reshape:0">, !atir.tensor<32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>) -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %tail_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail = "atir.Concat"(%tail_buf, %g_free, %dim_n) <{axis = 0 : i32}> : (!atir.tensor<3xi32, name = "concat_1:0">, !atir.tensor<2xi32, name = "GatherV2:0">, !atir.tensor<1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>) -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail64_buf = "atir.buffer"() : () -> !atir.tensor<3xi64, name = "Tensordot__:0">
    %tail64 = atir.Cast %tail64_buf, %tail : <3xi64, name = "Tensordot__:0">, <3xi32, name = "concat_1:0"> -> <3xi64, name = "Tensordot__:0">
    %out_buf = "atir.buffer"() : () -> !atir.tensor<4x5x16xf32, name = "out">
    %out = atir.Reshape %out_buf, %mm, %tail64 : <4x5x16xf32, name = "out">, <?x16xf32, name = "MatMul:0">, <3xi64, name = "Tensordot__:0"> -> <4x5x16xf32, name = "out">
    return %out : !atir.tensor<4x5x16xf32, name = "out">
  }
}


// -----

// 负例：MatMul 带 withBias。FTDMM 站点没有 bias 实参，这类形态由
// batched_matmul_add 等其他站点负责，这里不融。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_tensordot_matmul
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<4x5x32xf32, name = "x">) -> !atir.tensor<4x5x16xf32, name = "out"> {
    %perm = atir.constant "private" @"perm" -> <3xi64, name = "perm", data = dense<[0,1,2]> : tensor<3xi64>>
    %free = atir.constant "public" @"free" -> <2xi32, name = "free", data = dense<[0,1]> : tensor<2xi32>>
    %axes = atir.constant "public" @"axes" -> <1xi32, name = "axes", data = dense<[2]> : tensor<1xi32>>
    %zero1 = atir.constant "private" @"zero1" -> <1xi64, name = "zero1", data = dense<0> : tensor<1xi64>>
    %zero0 = atir.constant "private" @"zero0" -> <i32, name = "zero0", data = dense<0> : tensor<i32>>
    %dim_n = atir.constant "public" @"dim_n" -> <1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>
    %w = atir.constant "public" @"w" -> <32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>

    %t_buf = "atir.buffer"() : () -> !atir.tensor<4x5x32xf32, name = "transpose:0">
    %t = atir.Transpose %t_buf, %x, %perm {permutation = [0,1,2]} : <4x5x32xf32, name = "transpose:0">, <4x5x32xf32, name = "x">, <3xi64, name = "perm", data = dense<[0,1,2]> : tensor<3xi64>> -> <4x5x32xf32, name = "transpose:0">
    %shape = atir.Shape %x : (!atir.tensor<4x5x32xf32, name = "x">) -> !atir.tensor<3xi64, name = "Shape:0">
    %s32_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "Shape__:0">
    %s32 = atir.Cast %s32_buf, %shape : <3xi32, name = "Shape__:0">, <3xi64, name = "Shape:0"> -> <3xi32, name = "Shape__:0">
    %g_axes_buf = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "GatherV2_1:0">
    %g_axes = atir.Gather %g_axes_buf, %s32, %axes, %zero0 : <1xi32, name = "GatherV2_1:0">, <3xi32, name = "Shape__:0">, <1xi32, name = "axes", data = dense<[2]> : tensor<1xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <1xi32, name = "GatherV2_1:0">
    %g_free_buf = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "GatherV2:0">
    %g_free = atir.Gather %g_free_buf, %s32, %free, %zero0 : <2xi32, name = "GatherV2:0">, <3xi32, name = "Shape__:0">, <2xi32, name = "free", data = dense<[0,1]> : tensor<2xi32>>, <i32, name = "zero0", data = dense<0> : tensor<i32>> -> <2xi32, name = "GatherV2:0">
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
    %flat = atir.Reshape %flat_buf, %t, %cat64 : <?x32xf32, name = "Reshape:0">, <4x5x32xf32, name = "transpose:0">, <2xi64, name = "Reshape__731:0"> -> <?x32xf32, name = "Reshape:0">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %mm = "atir.MatMul"(%mm_buf, %flat, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = true}> : (!atir.tensor<?x16xf32, name = "MatMul:0">, !atir.tensor<?x32xf32, name = "Reshape:0">, !atir.tensor<32x16xf32, name = "w", data = dense<0.100000> : tensor<32x16xf32>>) -> !atir.tensor<?x16xf32, name = "MatMul:0">
    %tail_buf = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail = "atir.Concat"(%tail_buf, %g_free, %dim_n) <{axis = 0 : i32}> : (!atir.tensor<3xi32, name = "concat_1:0">, !atir.tensor<2xi32, name = "GatherV2:0">, !atir.tensor<1xi32, name = "dim_n", data = dense<16> : tensor<1xi32>>) -> !atir.tensor<3xi32, name = "concat_1:0">
    %tail64_buf = "atir.buffer"() : () -> !atir.tensor<3xi64, name = "Tensordot__:0">
    %tail64 = atir.Cast %tail64_buf, %tail : <3xi64, name = "Tensordot__:0">, <3xi32, name = "concat_1:0"> -> <3xi64, name = "Tensordot__:0">
    %out_buf = "atir.buffer"() : () -> !atir.tensor<4x5x16xf32, name = "out">
    %out = atir.Reshape %out_buf, %mm, %tail64 : <4x5x16xf32, name = "out">, <?x16xf32, name = "MatMul:0">, <3xi64, name = "Tensordot__:0"> -> <4x5x16xf32, name = "out">
    return %out : !atir.tensor<4x5x16xf32, name = "out">
  }
}
