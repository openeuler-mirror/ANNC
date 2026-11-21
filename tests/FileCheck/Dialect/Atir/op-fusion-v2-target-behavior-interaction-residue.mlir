// RUN: annc-asm %s -atir-op-fusion-v2 | FileCheck %s
// RUN: annc-asm %s -atir-op-fusion-v2 --annc-fast-codegen | FileCheck %s --check-prefix=CODEGEN
//
module {
  func.func @main(%arg0: tensor<?x?x4xf32> loc(unknown), %arg1: tensor<4x5xf32> loc(unknown), %arg2: tensor<5xf32> loc(unknown), %arg3: tensor<?x?x5xf32> loc(unknown)) -> tensor<?x?x20xf32> attributes {tf.arg_names = ["batch_input", "weight", "bias", "tile_input"], tf.entry_function = {control_outputs = "", inputs = "batch_input:0,weight:0,bias:0,tile_input:0", outputs = "w:0"}} {
    %0 = shape.const_shape [5] : tensor<1xindex> loc(unknown)
    %1 = shape.const_shape [4, 5] : !shape.shape loc(unknown)
    %c-2 = arith.constant -2 : index loc(unknown)
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32> loc(fused["Const:", "half"])
    %2 = shape.shape_of %arg0 : tensor<?x?x4xf32> -> tensor<3xindex> loc(fused["BatchMatMulV2:", "bmm"])
    %head, %tail = "shape.split_at"(%2, %c-2) : (tensor<3xindex>, index) -> (!shape.shape, !shape.shape) loc(fused["BatchMatMulV2:", "bmm"])
    %3 = shape.concat %head, %tail : !shape.shape, !shape.shape -> !shape.shape loc(fused["BatchMatMulV2:", "bmm"])
    %4 = shape.to_extent_tensor %3 : !shape.shape -> tensor<3xindex> loc(fused["BatchMatMulV2:", "bmm"])
    %5 = stablehlo.dynamic_broadcast_in_dim %arg0, %4, dims = [0, 1, 2] : (tensor<?x?x4xf32>, tensor<3xindex>) -> tensor<?x?x4xf32> loc(fused["BatchMatMulV2:", "bmm"])
    %6 = shape.concat %head, %1 : !shape.shape, !shape.shape -> !shape.shape loc(fused["BatchMatMulV2:", "bmm"])
    %7 = shape.to_extent_tensor %6 : !shape.shape -> tensor<3xindex> loc(fused["BatchMatMulV2:", "bmm"])
    %8 = stablehlo.dynamic_broadcast_in_dim %arg1, %7, dims = [1, 2] : (tensor<4x5xf32>, tensor<3xindex>) -> tensor<?x4x5xf32> loc(fused["BatchMatMulV2:", "bmm"])
    %9 = stablehlo.dot_general %5, %8, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<?x?x4xf32>, tensor<?x4x5xf32>) -> tensor<?x?x5xf32> loc(fused["BatchMatMulV2:", "bmm"])
    %10 = shape.shape_of %9 : tensor<?x?x5xf32> -> tensor<3xindex> loc(fused["AddV2:", "add21"])
    %11 = shape.broadcast %10, %0 : tensor<3xindex>, tensor<1xindex> -> tensor<3xindex> loc(fused["AddV2:", "add21"])
    %12 = stablehlo.dynamic_broadcast_in_dim %9, %11, dims = [0, 1, 2] : (tensor<?x?x5xf32>, tensor<3xindex>) -> tensor<?x?x5xf32> loc(fused["AddV2:", "add21"])
    %13 = stablehlo.dynamic_broadcast_in_dim %arg2, %11, dims = [2] : (tensor<5xf32>, tensor<3xindex>) -> tensor<?x?x5xf32> loc(fused["AddV2:", "add21"])
    %14 = stablehlo.add %12, %13 : tensor<?x?x5xf32> loc(fused["AddV2:", "add21"])
    %15 = stablehlo.abs %14 : tensor<?x?x5xf32> loc(fused["Abs:", "abs"])
    %16 = shape.shape_of %15 : tensor<?x?x5xf32> -> tensor<3xindex> loc(fused["AddV2:", "add22"])
    %17 = shape.shape_of %14 : tensor<?x?x5xf32> -> tensor<3xindex> loc(fused["AddV2:", "add22"])
    %18 = shape.cstr_broadcastable %16, %17 : tensor<3xindex>, tensor<3xindex> loc(fused["AddV2:", "add22"])
    %19 = shape.assuming %18 -> (tensor<?x?x5xf32>) {
      %30 = shape.shape_of %15 : tensor<?x?x5xf32> -> tensor<3xindex> loc(fused["AddV2:", "add22"])
      %31 = shape.shape_of %14 : tensor<?x?x5xf32> -> tensor<3xindex> loc(fused["AddV2:", "add22"])
      %32 = shape.broadcast %30, %31 : tensor<3xindex>, tensor<3xindex> -> tensor<3xindex> loc(fused["AddV2:", "add22"])
      %33 = stablehlo.dynamic_broadcast_in_dim %15, %32, dims = [0, 1, 2] : (tensor<?x?x5xf32>, tensor<3xindex>) -> tensor<?x?x5xf32> loc(fused["AddV2:", "add22"])
      %34 = stablehlo.dynamic_broadcast_in_dim %14, %32, dims = [0, 1, 2] : (tensor<?x?x5xf32>, tensor<3xindex>) -> tensor<?x?x5xf32> loc(fused["AddV2:", "add22"])
      %35 = stablehlo.add %33, %34 : tensor<?x?x5xf32> loc(fused["AddV2:", "add22"])
      shape.assuming_yield %35 : tensor<?x?x5xf32> loc(fused["AddV2:", "add22"])
    } loc(fused["AddV2:", "add22"])
    %rd_shape = shape.shape_of %19 : tensor<?x?x5xf32> -> tensor<3xindex> loc(fused["Mul:", "realdiv"])
    %half_bcast = stablehlo.dynamic_broadcast_in_dim %cst, %rd_shape, dims = [] : (tensor<f32>, tensor<3xindex>) -> tensor<?x?x5xf32> loc(fused["Mul:", "realdiv"])
    %20 = stablehlo.multiply %19, %half_bcast : tensor<?x?x5xf32> loc(fused["Mul:", "realdiv"])
    %21 = shape.shape_of %20 : tensor<?x?x5xf32> -> tensor<3xindex> loc(fused["Mul:", "mul"])
    %22 = shape.shape_of %arg3 : tensor<?x?x5xf32> -> tensor<3xindex> loc(fused["Mul:", "mul"])
    %23 = shape.cstr_broadcastable %21, %22 : tensor<3xindex>, tensor<3xindex> loc(fused["Mul:", "mul"])
    %24 = shape.assuming %23 -> (tensor<?x?x5xf32>) {
      %30 = shape.shape_of %20 : tensor<?x?x5xf32> -> tensor<3xindex> loc(fused["Mul:", "mul"])
      %31 = shape.shape_of %arg3 : tensor<?x?x5xf32> -> tensor<3xindex> loc(fused["Mul:", "mul"])
      %32 = shape.broadcast %30, %31 : tensor<3xindex>, tensor<3xindex> -> tensor<3xindex> loc(fused["Mul:", "mul"])
      %33 = stablehlo.dynamic_broadcast_in_dim %20, %32, dims = [0, 1, 2] : (tensor<?x?x5xf32>, tensor<3xindex>) -> tensor<?x?x5xf32> loc(fused["Mul:", "mul"])
      %34 = stablehlo.dynamic_broadcast_in_dim %arg3, %32, dims = [0, 1, 2] : (tensor<?x?x5xf32>, tensor<3xindex>) -> tensor<?x?x5xf32> loc(fused["Mul:", "mul"])
      %35 = stablehlo.multiply %33, %34 : tensor<?x?x5xf32> loc(fused["Mul:", "mul"])
      shape.assuming_yield %35 : tensor<?x?x5xf32> loc(fused["Mul:", "mul"])
    } loc(fused["Mul:", "mul"])
    %25 = shape.shape_of %20 : tensor<?x?x5xf32> -> tensor<3xindex> loc(fused["Sub:", "sub"])
    %26 = shape.shape_of %arg3 : tensor<?x?x5xf32> -> tensor<3xindex> loc(fused["Sub:", "sub"])
    %27 = shape.cstr_broadcastable %25, %26 : tensor<3xindex>, tensor<3xindex> loc(fused["Sub:", "sub"])
    %28 = shape.assuming %27 -> (tensor<?x?x5xf32>) {
      %30 = shape.shape_of %20 : tensor<?x?x5xf32> -> tensor<3xindex> loc(fused["Sub:", "sub"])
      %31 = shape.shape_of %arg3 : tensor<?x?x5xf32> -> tensor<3xindex> loc(fused["Sub:", "sub"])
      %32 = shape.broadcast %30, %31 : tensor<3xindex>, tensor<3xindex> -> tensor<3xindex> loc(fused["Sub:", "sub"])
      %33 = stablehlo.dynamic_broadcast_in_dim %20, %32, dims = [0, 1, 2] : (tensor<?x?x5xf32>, tensor<3xindex>) -> tensor<?x?x5xf32> loc(fused["Sub:", "sub"])
      %34 = stablehlo.dynamic_broadcast_in_dim %arg3, %32, dims = [0, 1, 2] : (tensor<?x?x5xf32>, tensor<3xindex>) -> tensor<?x?x5xf32> loc(fused["Sub:", "sub"])
      %35 = stablehlo.subtract %33, %34 : tensor<?x?x5xf32> loc(fused["Sub:", "sub"])
      shape.assuming_yield %35 : tensor<?x?x5xf32> loc(fused["Sub:", "sub"])
    } loc(fused["Sub:", "sub"])
    %axis_c = stablehlo.constant dense<2> : tensor<i32> loc("axisc")
    %29 = atir.unsupported {opType = "ConcatV2"} (%20, %arg3, %28, %24, %axis_c) : (tensor<?x?x5xf32>, tensor<?x?x5xf32>, tensor<?x?x5xf32>, tensor<?x?x5xf32>, tensor<i32>) -> tensor<?x?x20xf32> loc(fused["ConcatV2:", "w"])
    return %29 : tensor<?x?x20xf32> loc(unknown)
  }
}

// CHECK: call @KPFusedTargetBehaviorInteraction
// CHECK: func.func private @KPFusedTargetBehaviorInteraction
// CHECK-SAME: annc.kernel
// CHECK-SAME: abi = "annc_context_v3"
// CHECK-SAME: args = [{dtype = "f32", rank = 3 : i64, role = "dynamic", shape = [-9223372036854775808, -9223372036854775808, 4], tf_name = "batch_input"}, {dtype = "f32", rank = 2 : i64, role = "dynamic", shape = [4, 5], tf_name = "weight"}, {dtype = "f32", rank = 1 : i64, role = "dynamic", shape = [5], tf_name = "bias"}, {dtype = "f32", rank = 3 : i64, role = "dynamic", shape = [-9223372036854775808, -9223372036854775808, 5], tf_name = "tile_input"}]
// CHECK-SAME: fallback_function = "original_subgraph"
// CHECK-SAME: fusion.pattern = "kp_target_behavior_interaction"
// CHECK-SAME: kernel_name = "KPFusedTargetBehaviorInteraction"
// CHECK-SAME: outputs = [{dtype = "f32", rank = 3 : i64, role = "output", shape = [-9223372036854775808, -9223372036854775808, 20], tf_name = "w:0"}]

// CODEGEN: atir.Customize
// CODEGEN-SAME: custom.op_name = "KPFusedTargetBehaviorInteraction"
// CODEGEN: return
