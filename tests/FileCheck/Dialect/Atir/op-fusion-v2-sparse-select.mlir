// RUN: annc-asm %s -atir-op-fusion-v2 | FileCheck %s
// RUN: annc-asm %s -atir-op-fusion-v2 --annc-fast-codegen | FileCheck %s --check-prefix=CODEGEN
//
module {
  func.func @main(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>, %arg2: tensor<?xi32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<i32>) -> (tensor<?x1xi32>, tensor<4x2xf32>) attributes {tf.arg_names = ["a", "b", "c", "gt", "eq1", "eq2", "eq3"], tf.entry_function = {control_outputs = "", inputs = "a:0,b:0,c:0,gt:0,eq1:0,eq2:0,eq3:0", outputs = "x:0,w:0"}} {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32> loc(fused["Fill:", "fill"])
    %c = stablehlo.constant dense<[-1, 1]> : tensor<2xi32> loc(fused["Reshape:", "rs_x_shape"])
    %0 = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", device = "", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%arg1, %c) : (tensor<?xi32>, tensor<2xi32>) -> tensor<?x1xi32> loc(fused["Reshape:", "reshape_b"])
    %1 = shape.shape_of %0 : tensor<?x1xi32> -> tensor<2xindex> loc(fused["Equal:", "equal"])
    %2 = stablehlo.dynamic_broadcast_in_dim %arg4, %1, dims = [] : (tensor<i32>, tensor<2xindex>) -> tensor<?x1xi32> loc(fused["Equal:", "equal"])
    %3 = stablehlo.compare  EQ, %0, %2 : (tensor<?x1xi32>, tensor<?x1xi32>) -> tensor<?x1xi1> loc(fused["Equal:", "equal"])
    %4 = shape.shape_of %0 : tensor<?x1xi32> -> tensor<2xindex> loc(fused["Equal:", "equal_1"])
    %5 = stablehlo.dynamic_broadcast_in_dim %arg5, %4, dims = [] : (tensor<i32>, tensor<2xindex>) -> tensor<?x1xi32> loc(fused["Equal:", "equal_1"])
    %6 = stablehlo.compare  EQ, %0, %5 : (tensor<?x1xi32>, tensor<?x1xi32>) -> tensor<?x1xi1> loc(fused["Equal:", "equal_1"])
    %7 = shape.shape_of %0 : tensor<?x1xi32> -> tensor<2xindex> loc(fused["Shape:", "shape_b"])
    %8 = arith.index_cast %7 : tensor<2xindex> to tensor<2xi32> loc(fused["Shape:", "shape_b"])
    %9 = stablehlo.dynamic_broadcast_in_dim %cst, %8, dims = [] : (tensor<f32>, tensor<2xi32>) -> tensor<?x1xf32> loc(fused["Fill:", "fill"])
    %10 = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", device = "", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%arg2, %c) : (tensor<?xi32>, tensor<2xi32>) -> tensor<?x1xi32> loc(fused["Reshape:", "reshape_c"])
    %11 = shape.shape_of %10 : tensor<?x1xi32> -> tensor<2xindex> loc(fused["Equal:", "equal_2"])
    %12 = stablehlo.dynamic_broadcast_in_dim %arg6, %11, dims = [] : (tensor<i32>, tensor<2xindex>) -> tensor<?x1xi32> loc(fused["Equal:", "equal_2"])
    %13 = stablehlo.compare  EQ, %10, %12 : (tensor<?x1xi32>, tensor<?x1xi32>) -> tensor<?x1xi1> loc(fused["Equal:", "equal_2"])
    %14 = shape.shape_of %10 : tensor<?x1xi32> -> tensor<2xindex> loc(fused["Shape:", "shape_c"])
    %15 = arith.index_cast %14 : tensor<2xindex> to tensor<2xi32> loc(fused["Shape:", "shape_c"])
    %16 = stablehlo.dynamic_broadcast_in_dim %cst, %15, dims = [] : (tensor<f32>, tensor<2xi32>) -> tensor<?x1xf32> loc(fused["Fill:", "fill_1"])
    %17 = shape.shape_of %13 : tensor<?x1xi1> -> tensor<2xindex> loc(fused["Select:", "select_2"])
    %18 = shape.shape_of %16 : tensor<?x1xf32> -> tensor<2xindex> loc(fused["Select:", "select_2"])
    %19 = shape.shape_of %16 : tensor<?x1xf32> -> tensor<2xindex> loc(fused["Select:", "select_2"])
    %20 = shape.cstr_eq %18, %19 : tensor<2xindex>, tensor<2xindex> loc(fused["Select:", "select_2"])
    %21 = shape.cstr_eq %17, %18 : tensor<2xindex>, tensor<2xindex> loc(fused["Select:", "select_2"])
    %22 = shape.cstr_eq %18, %19, %17, %18 : tensor<2xindex>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex> loc(fused["Select:", "select_2"])
    %23 = shape.assuming %22 -> (tensor<?x1xf32>) {
      shape.assuming_yield %16 : tensor<?x1xf32> loc(fused["Select:", "select_2"])
    } loc(fused["Select:", "select_2"])
    %24 = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", device = "", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%arg0, %c) : (tensor<?xi32>, tensor<2xi32>) -> tensor<?x1xi32> loc(fused["Reshape:", "x"])
    %25 = shape.shape_of %24 : tensor<?x1xi32> -> tensor<2xindex> loc(fused["Greater:", "greater"])
    %26 = stablehlo.dynamic_broadcast_in_dim %arg3, %25, dims = [] : (tensor<i32>, tensor<2xindex>) -> tensor<?x1xi32> loc(fused["Greater:", "greater"])
    %27 = stablehlo.compare  GT, %24, %26 : (tensor<?x1xi32>, tensor<?x1xi32>) -> tensor<?x1xi1> loc(fused["Greater:", "greater"])
    %28 = stablehlo.convert %27 : (tensor<?x1xi1>) -> tensor<?x1xf32> loc(fused["Cast:", "cast"])
    %29 = shape.shape_of %3 : tensor<?x1xi1> -> tensor<2xindex> loc(fused["Select:", "select_1"])
    %30 = shape.shape_of %9 : tensor<?x1xf32> -> tensor<2xindex> loc(fused["Select:", "select_1"])
    %31 = shape.shape_of %28 : tensor<?x1xf32> -> tensor<2xindex> loc(fused["Select:", "select_1"])
    %32 = shape.cstr_eq %30, %31 : tensor<2xindex>, tensor<2xindex> loc(fused["Select:", "select_1"])
    %33 = shape.cstr_eq %29, %30 : tensor<2xindex>, tensor<2xindex> loc(fused["Select:", "select_1"])
    %34 = shape.cstr_eq %30, %31, %29, %30 : tensor<2xindex>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex> loc(fused["Select:", "select_1"])
    %35 = shape.assuming %34 -> (tensor<?x1xf32>) {
      %44 = stablehlo.select %3, %9, %28 : tensor<?x1xi1>, tensor<?x1xf32> loc(fused["Select:", "select_1"])
      shape.assuming_yield %44 : tensor<?x1xf32> loc(fused["Select:", "select_1"])
    } loc(fused["Select:", "select_1"])
    %36 = shape.shape_of %6 : tensor<?x1xi1> -> tensor<2xindex> loc(fused["Select:", "select_0"])
    %37 = shape.shape_of %9 : tensor<?x1xf32> -> tensor<2xindex> loc(fused["Select:", "select_0"])
    %38 = shape.shape_of %35 : tensor<?x1xf32> -> tensor<2xindex> loc(fused["Select:", "select_0"])
    %39 = shape.cstr_eq %37, %38 : tensor<2xindex>, tensor<2xindex> loc(fused["Select:", "select_0"])
    %40 = shape.cstr_eq %36, %37 : tensor<2xindex>, tensor<2xindex> loc(fused["Select:", "select_0"])
    %41 = shape.cstr_eq %37, %38, %36, %37 : tensor<2xindex>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex> loc(fused["Select:", "select_0"])
    %42 = shape.assuming %41 -> (tensor<?x1xf32>) {
      %44 = stablehlo.select %6, %9, %35 : tensor<?x1xi1>, tensor<?x1xf32> loc(fused["Select:", "select_0"])
      shape.assuming_yield %44 : tensor<?x1xf32> loc(fused["Select:", "select_0"])
    } loc(fused["Select:", "select_0"])
    %43 = stablehlo.concatenate %42, %23, dim = 1 : (tensor<?x1xf32>, tensor<?x1xf32>) -> tensor<4x2xf32> loc(fused["ConcatV2:", "w"])
    return %24, %43 : tensor<?x1xi32>, tensor<4x2xf32> loc("output")
  }
}

// CHECK: call @KPFusedSparseSelect
// CHECK: func.func private @KPFusedSparseSelect
// CHECK-SAME: annc.kernel
// CHECK-SAME: abi = "annc_context_v3"
// CHECK-SAME: args = [{dtype = "i32", rank = 1 : i64, role = "dynamic", shape = [-9223372036854775808], tf_name = "a"}, {dtype = "i32", rank = 1 : i64, role = "dynamic", shape = [-9223372036854775808], tf_name = "b"}, {dtype = "i32", rank = 1 : i64, role = "dynamic", shape = [-9223372036854775808], tf_name = "c"}, {dtype = "i32", rank = 0 : i64, role = "dynamic", shape = [], tf_name = "gt"}, {dtype = "i32", rank = 0 : i64, role = "dynamic", shape = [], tf_name = "eq1"}, {dtype = "i32", rank = 0 : i64, role = "dynamic", shape = [], tf_name = "eq2"}, {dtype = "i32", rank = 0 : i64, role = "dynamic", shape = [], tf_name = "eq3"}]
// CHECK-SAME: fallback_function = "original_subgraph"
// CHECK-SAME: fusion.pattern = "kp_sparse_select"
// CHECK-SAME: kernel_name = "KPFusedSparseSelect"
// CHECK-SAME: outputs = [{dtype = "i32", rank = 2 : i64, role = "output", shape = [-9223372036854775808, 1], tf_name = "x:0"}, {dtype = "f32", rank = 2 : i64, role = "output", shape = [-9223372036854775808, 1], tf_name = "select_0:0"}, {dtype = "f32", rank = 2 : i64, role = "output", shape = [4, 2], tf_name = "w:0"}]

// CODEGEN: atir.Customize
// CODEGEN-SAME: custom.op_name = "KPFusedSparseSelect"
// CODEGEN: return
