// RUN: annc-opt --split-input-file %s -atir-config-fusion="config=%S/../../../../configs/fusion/kp_embedding_patterns.json run-builtin-after=false warn-limit=0" | FileCheck %s --check-prefix=CONFIG
// Config fusion first level for sparse select (KPFusedSparseSelect),
// Execution V2: kernel func signature (!llvm.ptr, a, b, c, gt, eq1, eq2,
// eq3) -> (x, y, w); 7 inputs = exactly the V2 input limit.

// CONFIG: func.func private @fused_kp_sparse_select_
// CONFIG-SAME: %arg0: !llvm.ptr
// CONFIG-SAME: annc.kernel
// CONFIG-SAME: abi = "annc_execution_v2"
// CONFIG: func.func @main
// CONFIG: atir.ConcatV2
// CONFIG-SAME: annc.fusion_materialized

// 正例: 完整链 (a/b/c i32, 4 个比较常量, empty [?,0] 第三输入)。
// CHECK-LABEL: func.func @main
// CHECK: atir.ConcatV2
// CHECK-SAME: annc.fusion_materialized
// CHECK-LABEL: func.func private @fused_kp_sparse_select_
// CHECK-SAME: %arg0: !llvm.ptr,
// CHECK-SAME: %arg1: !atir.tensor<?xi32, name = "a">, %arg2: !atir.tensor<?xi32, name = "b">, %arg3: !atir.tensor<?xi32, name = "c">, %arg4: !atir.tensor<i32, name = "gt_c", data = dense<0> : tensor<i32>>, %arg5: !atir.tensor<i32, name = "eq1_c", data = dense<1> : tensor<i32>>, %arg6: !atir.tensor<i32, name = "eq2_c", data = dense<2> : tensor<i32>>, %arg7: !atir.tensor<i32, name = "eq3_c", data = dense<3> : tensor<i32>>
// CHECK-SAME: annc.kernel
// CHECK-SAME: abi = "annc_execution_v2"
// CHECK-SAME: kernel_name = "fused_kp_sparse_select_
module attributes {module.state = "atir"} {
  func.func @main(
      %arg0: !atir.tensor<?xi32, name = "a">,
      %arg1: !atir.tensor<?xi32, name = "b">,
      %arg2: !atir.tensor<?xi32, name = "c">)
      -> (!atir.tensor<?x1xi32, name = "reshape_4">, !atir.tensor<?x2xf32, name = "concat_out">) {
  %0 = atir.constant "public" @"rs_shape" -> <2xi32, name = "rs_shape", data = dense<[-1, 1]> : tensor<2xi32>>
  %1 = "atir.buffer"() : () -> !atir.tensor<?x1xi32, name = "reshape_4">
  %2 = "atir.Reshape"(%1, %arg0, %0) : (!atir.tensor<?x1xi32, name = "reshape_4">, !atir.tensor<?xi32, name = "a">, !atir.tensor<2xi32, name = "rs_shape", data = dense<[-1, 1]> : tensor<2xi32>>) -> !atir.tensor<?x1xi32, name = "reshape_4">
  %3 = "atir.buffer"() : () -> !atir.tensor<?x1xi32, name = "reshape_1">
  %4 = "atir.Reshape"(%3, %arg1, %0) : (!atir.tensor<?x1xi32, name = "reshape_1">, !atir.tensor<?xi32, name = "b">, !atir.tensor<2xi32, name = "rs_shape", data = dense<[-1, 1]> : tensor<2xi32>>) -> !atir.tensor<?x1xi32, name = "reshape_1">
  %5 = "atir.buffer"() : () -> !atir.tensor<?x1xi32, name = "reshape_2">
  %6 = "atir.Reshape"(%5, %arg2, %0) : (!atir.tensor<?x1xi32, name = "reshape_2">, !atir.tensor<?xi32, name = "c">, !atir.tensor<2xi32, name = "rs_shape", data = dense<[-1, 1]> : tensor<2xi32>>) -> !atir.tensor<?x1xi32, name = "reshape_2">
  %7 = atir.constant "public" @"gt_c" -> <i32, name = "gt_c", data = dense<0> : tensor<i32>>
  %8 = atir.constant "public" @"eq1_c" -> <i32, name = "eq1_c", data = dense<1> : tensor<i32>>
  %9 = atir.constant "public" @"eq2_c" -> <i32, name = "eq2_c", data = dense<2> : tensor<i32>>
  %10 = atir.constant "public" @"eq3_c" -> <i32, name = "eq3_c", data = dense<3> : tensor<i32>>
  %11 = atir.Compare %2, %7 {comparisonDirection = "GT"} : (!atir.tensor<?x1xi32, name = "reshape_4">, !atir.tensor<i32, name = "gt_c", data = dense<0> : tensor<i32>>) -> !atir.tensor<?x1xi32, encoding = <"bool">, name = "greater">
  %12 = "atir.buffer"() : () -> !atir.tensor<?x1xf32, name = "cast">
  %13 = "atir.Cast"(%12, %11) : (!atir.tensor<?x1xf32, name = "cast">, !atir.tensor<?x1xi32, encoding = <"bool">, name = "greater">) -> !atir.tensor<?x1xf32, name = "cast">
  %14 = atir.Shape %4 : (!atir.tensor<?x1xi32, name = "reshape_1">) -> !atir.tensor<2xi32, name = "Shape">
  %15 = atir.constant "public" @"one_f" -> <f32, name = "one_f", data = dense<1.000000e+00> : tensor<f32>>
  %16 = "atir.buffer"() : () -> !atir.tensor<?x1xf32, name = "fill">
  %17 = "atir.Fill"(%16, %14, %15) : (!atir.tensor<?x1xf32, name = "fill">, !atir.tensor<2xi32, name = "Shape">, !atir.tensor<f32, name = "one_f", data = dense<1.000000e+00> : tensor<f32>>) -> !atir.tensor<?x1xf32, name = "fill">
  %18 = atir.Compare %4, %8 {comparisonDirection = "EQ"} : (!atir.tensor<?x1xi32, name = "reshape_1">, !atir.tensor<i32, name = "eq1_c", data = dense<1> : tensor<i32>>) -> !atir.tensor<?x1xi32, encoding = <"bool">, name = "equal">
  %19 = "atir.buffer"() : () -> !atir.tensor<?x1xf32, name = "select_1">
  %20 = "atir.Where"(%19, %18, %17, %13) : (!atir.tensor<?x1xf32, name = "select_1">, !atir.tensor<?x1xi32, encoding = <"bool">, name = "equal">, !atir.tensor<?x1xf32, name = "fill">, !atir.tensor<?x1xf32, name = "cast">) -> !atir.tensor<?x1xf32, name = "select_1">
  %21 = atir.Compare %4, %9 {comparisonDirection = "EQ"} : (!atir.tensor<?x1xi32, name = "reshape_1">, !atir.tensor<i32, name = "eq2_c", data = dense<2> : tensor<i32>>) -> !atir.tensor<?x1xi32, encoding = <"bool">, name = "equal_1">
  %22 = atir.Shape %4 : (!atir.tensor<?x1xi32, name = "reshape_1">) -> !atir.tensor<2xi32, name = "Shape_1">
  %23 = "atir.buffer"() : () -> !atir.tensor<?x1xf32, name = "fill_0">
  %24 = "atir.Fill"(%23, %22, %15) : (!atir.tensor<?x1xf32, name = "fill_0">, !atir.tensor<2xi32, name = "Shape_1">, !atir.tensor<f32, name = "one_f", data = dense<1.000000e+00> : tensor<f32>>) -> !atir.tensor<?x1xf32, name = "fill_0">
  %25 = "atir.buffer"() : () -> !atir.tensor<?x1xf32, name = "select_0">
  %26 = "atir.Where"(%25, %21, %24, %20) : (!atir.tensor<?x1xf32, name = "select_0">, !atir.tensor<?x1xi32, encoding = <"bool">, name = "equal_1">, !atir.tensor<?x1xf32, name = "fill_0">, !atir.tensor<?x1xf32, name = "select_1">) -> !atir.tensor<?x1xf32, name = "select_0">
  %27 = atir.Compare %6, %10 {comparisonDirection = "EQ"} : (!atir.tensor<?x1xi32, name = "reshape_2">, !atir.tensor<i32, name = "eq3_c", data = dense<3> : tensor<i32>>) -> !atir.tensor<?x1xi32, encoding = <"bool">, name = "equal_2">
  %28 = atir.Shape %6 : (!atir.tensor<?x1xi32, name = "reshape_2">) -> !atir.tensor<2xi32, name = "Shape_2">
  %29 = atir.Shape %6 : (!atir.tensor<?x1xi32, name = "reshape_2">) -> !atir.tensor<2xi32, name = "Shape_3">
  %30 = "atir.buffer"() : () -> !atir.tensor<?x1xf32, name = "fill_1">
  %31 = "atir.Fill"(%30, %28, %15) : (!atir.tensor<?x1xf32, name = "fill_1">, !atir.tensor<2xi32, name = "Shape_2">, !atir.tensor<f32, name = "one_f", data = dense<1.000000e+00> : tensor<f32>>) -> !atir.tensor<?x1xf32, name = "fill_1">
  %32 = "atir.buffer"() : () -> !atir.tensor<?x1xf32, name = "fill_2">
  %33 = "atir.Fill"(%32, %29, %15) : (!atir.tensor<?x1xf32, name = "fill_2">, !atir.tensor<2xi32, name = "Shape_3">, !atir.tensor<f32, name = "one_f", data = dense<1.000000e+00> : tensor<f32>>) -> !atir.tensor<?x1xf32, name = "fill_2">
  %34 = "atir.buffer"() : () -> !atir.tensor<?x1xf32, name = "select_2">
  %35 = "atir.Where"(%34, %27, %33, %31) : (!atir.tensor<?x1xf32, name = "select_2">, !atir.tensor<?x1xi32, encoding = <"bool">, name = "equal_2">, !atir.tensor<?x1xf32, name = "fill_2">, !atir.tensor<?x1xf32, name = "fill_1">) -> !atir.tensor<?x1xf32, name = "select_2">
  %36 = atir.constant "public" @"empty_shape" -> <2xi32, name = "empty_shape", data = dense<[-1, 0]> : tensor<2xi32>>
  %37 = atir.constant "public" @"zero_f" -> <f32, name = "zero_f", data = dense<0.000000e+00> : tensor<f32>>
  %38 = "atir.buffer"() : () -> !atir.tensor<?x0xf32, name = "empty">
  %39 = "atir.Fill"(%38, %36, %37) : (!atir.tensor<?x0xf32, name = "empty">, !atir.tensor<2xi32, name = "empty_shape", data = dense<[-1, 0]> : tensor<2xi32>>, !atir.tensor<f32, name = "zero_f", data = dense<0.000000e+00> : tensor<f32>>) -> !atir.tensor<?x0xf32, name = "empty">
  %40 = atir.constant "public" @"axis_c" -> <i32, name = "axis_c", data = dense<1> : tensor<i32>>
  %41 = "atir.buffer"() : () -> !atir.tensor<?x2xf32, name = "concat_out">
  %42 = "atir.ConcatV2"(%41, %26, %35, %39, %40) : (!atir.tensor<?x2xf32, name = "concat_out">, !atir.tensor<?x1xf32, name = "select_0">, !atir.tensor<?x1xf32, name = "select_2">, !atir.tensor<?x0xf32, name = "empty">, !atir.tensor<i32, name = "axis_c", data = dense<1> : tensor<i32>>) -> !atir.tensor<?x2xf32, name = "concat_out">
  return %2, %42 : !atir.tensor<?x1xi32, name = "reshape_4">, !atir.tensor<?x2xf32, name = "concat_out">
  }
}

// -----

// 负例 1: concat axis = 0 (拒绝, 812 只支持 axis=1)。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_kp_sparse_select_
module attributes {module.state = "atir"} {
  func.func @main(
      %arg0: !atir.tensor<?xi32, name = "a">,
      %arg1: !atir.tensor<?xi32, name = "b">,
      %arg2: !atir.tensor<?xi32, name = "c">)
      -> (!atir.tensor<?x1xi32, name = "reshape_4">, !atir.tensor<?x2xf32, name = "concat_out">) {
  %0 = atir.constant "public" @"rs_shape" -> <2xi32, name = "rs_shape", data = dense<[-1, 1]> : tensor<2xi32>>
  %1 = "atir.buffer"() : () -> !atir.tensor<?x1xi32, name = "reshape_4">
  %2 = "atir.Reshape"(%1, %arg0, %0) : (!atir.tensor<?x1xi32, name = "reshape_4">, !atir.tensor<?xi32, name = "a">, !atir.tensor<2xi32, name = "rs_shape", data = dense<[-1, 1]> : tensor<2xi32>>) -> !atir.tensor<?x1xi32, name = "reshape_4">
  %3 = "atir.buffer"() : () -> !atir.tensor<?x1xi32, name = "reshape_1">
  %4 = "atir.Reshape"(%3, %arg1, %0) : (!atir.tensor<?x1xi32, name = "reshape_1">, !atir.tensor<?xi32, name = "b">, !atir.tensor<2xi32, name = "rs_shape", data = dense<[-1, 1]> : tensor<2xi32>>) -> !atir.tensor<?x1xi32, name = "reshape_1">
  %5 = "atir.buffer"() : () -> !atir.tensor<?x1xi32, name = "reshape_2">
  %6 = "atir.Reshape"(%5, %arg2, %0) : (!atir.tensor<?x1xi32, name = "reshape_2">, !atir.tensor<?xi32, name = "c">, !atir.tensor<2xi32, name = "rs_shape", data = dense<[-1, 1]> : tensor<2xi32>>) -> !atir.tensor<?x1xi32, name = "reshape_2">
  %7 = atir.constant "public" @"gt_c" -> <i32, name = "gt_c", data = dense<0> : tensor<i32>>
  %8 = atir.constant "public" @"eq1_c" -> <i32, name = "eq1_c", data = dense<1> : tensor<i32>>
  %9 = atir.constant "public" @"eq2_c" -> <i32, name = "eq2_c", data = dense<2> : tensor<i32>>
  %10 = atir.constant "public" @"eq3_c" -> <i32, name = "eq3_c", data = dense<3> : tensor<i32>>
  %11 = atir.Compare %2, %7 {comparisonDirection = "GT"} : (!atir.tensor<?x1xi32, name = "reshape_4">, !atir.tensor<i32, name = "gt_c", data = dense<0> : tensor<i32>>) -> !atir.tensor<?x1xi32, encoding = <"bool">, name = "greater">
  %12 = "atir.buffer"() : () -> !atir.tensor<?x1xf32, name = "cast">
  %13 = "atir.Cast"(%12, %11) : (!atir.tensor<?x1xf32, name = "cast">, !atir.tensor<?x1xi32, encoding = <"bool">, name = "greater">) -> !atir.tensor<?x1xf32, name = "cast">
  %14 = atir.Shape %4 : (!atir.tensor<?x1xi32, name = "reshape_1">) -> !atir.tensor<2xi32, name = "Shape">
  %15 = atir.constant "public" @"one_f" -> <f32, name = "one_f", data = dense<1.000000e+00> : tensor<f32>>
  %16 = "atir.buffer"() : () -> !atir.tensor<?x1xf32, name = "fill">
  %17 = "atir.Fill"(%16, %14, %15) : (!atir.tensor<?x1xf32, name = "fill">, !atir.tensor<2xi32, name = "Shape">, !atir.tensor<f32, name = "one_f", data = dense<1.000000e+00> : tensor<f32>>) -> !atir.tensor<?x1xf32, name = "fill">
  %18 = atir.Compare %4, %8 {comparisonDirection = "EQ"} : (!atir.tensor<?x1xi32, name = "reshape_1">, !atir.tensor<i32, name = "eq1_c", data = dense<1> : tensor<i32>>) -> !atir.tensor<?x1xi32, encoding = <"bool">, name = "equal">
  %19 = "atir.buffer"() : () -> !atir.tensor<?x1xf32, name = "select_1">
  %20 = "atir.Where"(%19, %18, %17, %13) : (!atir.tensor<?x1xf32, name = "select_1">, !atir.tensor<?x1xi32, encoding = <"bool">, name = "equal">, !atir.tensor<?x1xf32, name = "fill">, !atir.tensor<?x1xf32, name = "cast">) -> !atir.tensor<?x1xf32, name = "select_1">
  %21 = atir.Compare %4, %9 {comparisonDirection = "EQ"} : (!atir.tensor<?x1xi32, name = "reshape_1">, !atir.tensor<i32, name = "eq2_c", data = dense<2> : tensor<i32>>) -> !atir.tensor<?x1xi32, encoding = <"bool">, name = "equal_1">
  %22 = atir.Shape %4 : (!atir.tensor<?x1xi32, name = "reshape_1">) -> !atir.tensor<2xi32, name = "Shape_1">
  %23 = "atir.buffer"() : () -> !atir.tensor<?x1xf32, name = "fill_0">
  %24 = "atir.Fill"(%23, %22, %15) : (!atir.tensor<?x1xf32, name = "fill_0">, !atir.tensor<2xi32, name = "Shape_1">, !atir.tensor<f32, name = "one_f", data = dense<1.000000e+00> : tensor<f32>>) -> !atir.tensor<?x1xf32, name = "fill_0">
  %25 = "atir.buffer"() : () -> !atir.tensor<?x1xf32, name = "select_0">
  %26 = "atir.Where"(%25, %21, %24, %20) : (!atir.tensor<?x1xf32, name = "select_0">, !atir.tensor<?x1xi32, encoding = <"bool">, name = "equal_1">, !atir.tensor<?x1xf32, name = "fill_0">, !atir.tensor<?x1xf32, name = "select_1">) -> !atir.tensor<?x1xf32, name = "select_0">
  %27 = atir.Compare %6, %10 {comparisonDirection = "EQ"} : (!atir.tensor<?x1xi32, name = "reshape_2">, !atir.tensor<i32, name = "eq3_c", data = dense<3> : tensor<i32>>) -> !atir.tensor<?x1xi32, encoding = <"bool">, name = "equal_2">
  %28 = atir.Shape %6 : (!atir.tensor<?x1xi32, name = "reshape_2">) -> !atir.tensor<2xi32, name = "Shape_2">
  %29 = atir.Shape %6 : (!atir.tensor<?x1xi32, name = "reshape_2">) -> !atir.tensor<2xi32, name = "Shape_3">
  %30 = "atir.buffer"() : () -> !atir.tensor<?x1xf32, name = "fill_1">
  %31 = "atir.Fill"(%30, %28, %15) : (!atir.tensor<?x1xf32, name = "fill_1">, !atir.tensor<2xi32, name = "Shape_2">, !atir.tensor<f32, name = "one_f", data = dense<1.000000e+00> : tensor<f32>>) -> !atir.tensor<?x1xf32, name = "fill_1">
  %32 = "atir.buffer"() : () -> !atir.tensor<?x1xf32, name = "fill_2">
  %33 = "atir.Fill"(%32, %29, %15) : (!atir.tensor<?x1xf32, name = "fill_2">, !atir.tensor<2xi32, name = "Shape_3">, !atir.tensor<f32, name = "one_f", data = dense<1.000000e+00> : tensor<f32>>) -> !atir.tensor<?x1xf32, name = "fill_2">
  %34 = "atir.buffer"() : () -> !atir.tensor<?x1xf32, name = "select_2">
  %35 = "atir.Where"(%34, %27, %33, %31) : (!atir.tensor<?x1xf32, name = "select_2">, !atir.tensor<?x1xi32, encoding = <"bool">, name = "equal_2">, !atir.tensor<?x1xf32, name = "fill_2">, !atir.tensor<?x1xf32, name = "fill_1">) -> !atir.tensor<?x1xf32, name = "select_2">
  %36 = atir.constant "public" @"empty_shape" -> <2xi32, name = "empty_shape", data = dense<[-1, 0]> : tensor<2xi32>>
  %37 = atir.constant "public" @"zero_f" -> <f32, name = "zero_f", data = dense<0.000000e+00> : tensor<f32>>
  %38 = "atir.buffer"() : () -> !atir.tensor<?x0xf32, name = "empty">
  %39 = "atir.Fill"(%38, %36, %37) : (!atir.tensor<?x0xf32, name = "empty">, !atir.tensor<2xi32, name = "empty_shape", data = dense<[-1, 0]> : tensor<2xi32>>, !atir.tensor<f32, name = "zero_f", data = dense<0.000000e+00> : tensor<f32>>) -> !atir.tensor<?x0xf32, name = "empty">
  %40 = atir.constant "public" @"axis_c" -> <i32, name = "axis_c", data = dense<0> : tensor<i32>>
  %41 = "atir.buffer"() : () -> !atir.tensor<?x2xf32, name = "concat_out">
  %42 = "atir.ConcatV2"(%41, %26, %35, %39, %40) : (!atir.tensor<?x2xf32, name = "concat_out">, !atir.tensor<?x1xf32, name = "select_0">, !atir.tensor<?x1xf32, name = "select_2">, !atir.tensor<?x0xf32, name = "empty">, !atir.tensor<i32, name = "axis_c", data = dense<0> : tensor<i32>>) -> !atir.tensor<?x2xf32, name = "concat_out">
  return %2, %42 : !atir.tensor<?x1xi32, name = "reshape_4">, !atir.tensor<?x2xf32, name = "concat_out">
  }
}
