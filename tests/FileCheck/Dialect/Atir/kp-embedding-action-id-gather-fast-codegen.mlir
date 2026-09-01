// RUN: annc-asm --atir-fast-codegen %s | FileCheck %s
// KP-style embedding action-id gather fusion, second level: inside the
// kernel func extracted by OpFusion (args = indices1/params/indices2/
// pack_dim/pack/output), the Gather->Gather->Reshape(Pack(-1)) +
// Fill(Pack(pack_dim, pack)) subgraph folds into a single
// KPFusedEmbeddingActionIdGather CustomizeOp.

// CHECK-LABEL: func.func private @fused_kp_embedding_action_id_gather
// CHECK: atir.Customize
// CHECK-SAME: custom.op_name = "KPFusedEmbeddingActionIdGatherR2x2"
// CHECK-NOT: atir.Gather
// CHECK-NOT: atir.ConcatV2
module attributes {module.state = "atir"} {
  func.func private @fused_kp_embedding_action_id_gather(
      %arg0: !atir.tensor<2x1xi64, name = "indices1">,
      %arg1: !atir.tensor<10x4xf32, name = "params">,
      %arg2: !atir.tensor<1x2xi32, name = "indices2">,
      %arg3: !atir.tensor<i32, name = "pack_dim">,
      %arg4: !atir.tensor<i32, name = "pack">,
      %arg5: !atir.tensor<2x5xf32, name = "output">) attributes {annc.kernel} {
    // Inner gather: params[indices1] along axis 0 -> [2,1,4].
    %0 = "atir.buffer"() : () -> !atir.tensor<2x1x4xf32, name = "inner_gather">
    %1 = atir.constant "public" @"axis_0" -> <i32, name = "axis_0", data = dense<0> : tensor<i32>>
    %2 = atir.Gather %0, %arg1, %arg0, %1 : <2x1x4xf32, name = "inner_gather">, <10x4xf32, name = "params">, <2x1xi64, name = "indices1">, <i32, name = "axis_0", data = dense<0> : tensor<i32>> -> <2x1x4xf32, name = "inner_gather">
    // Outer gather: inner[indices2] along axis 0 -> [1,2,1,4].
    %3 = "atir.buffer"() : () -> !atir.tensor<1x2x1x4xf32, name = "outer_gather">
    %4 = atir.constant "public" @"axis_1" -> <i32, name = "axis_1", data = dense<0> : tensor<i32>>
    %5 = atir.Gather %3, %2, %arg2, %4 : <1x2x1x4xf32, name = "outer_gather">, <2x1x4xf32, name = "inner_gather">, <1x2xi32, name = "indices2">, <i32, name = "axis_1", data = dense<0> : tensor<i32>> -> <1x2x1x4xf32, name = "outer_gather">
    // Reshape shape: Pack(pack_dim, -1) -> [2,4].
    %6 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "pack_shape">
    %7 = atir.constant "public" @"minus_one" -> <i32, name = "minus_one", data = dense<-1> : tensor<i32>>
    %8 = atir.Pack %6, (%arg3, %7) axis = 0 : <2xi32, name = "pack_shape">, !atir.tensor<i32, name = "pack_dim">, !atir.tensor<i32, name = "minus_one", data = dense<-1> : tensor<i32>> -> <2xi32, name = "pack_shape">
    %9 = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "reshaped">
    %10 = atir.Reshape %9, %5, %8 : <2x4xf32, name = "reshaped">, <1x2x1x4xf32, name = "outer_gather">, <2xi32, name = "pack_shape"> -> <2x4xf32, name = "reshaped">
    // Fill shape: Pack(pack_dim, pack); value = 0.
    %11 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "pack_dims">
    %12 = atir.Pack %11, (%arg3, %arg4) axis = 0 : <2xi32, name = "pack_dims">, !atir.tensor<i32, name = "pack_dim">, !atir.tensor<i32, name = "pack"> -> <2xi32, name = "pack_dims">
    %13 = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "fill_out">
    %14 = atir.constant "public" @"zero" -> <f32, name = "zero", data = dense<0.0> : tensor<f32>>
    %15 = atir.Fill %13, %12, %14 : <2x1xf32, name = "fill_out">, <2xi32, name = "pack_dims">, <f32, name = "zero", data = dense<0.0> : tensor<f32>> -> <2x1xf32, name = "fill_out">
    // Anchor: ConcatV2([reshaped, fill_out], axis=-1) -> [2,5].
    %16 = atir.constant "public" @"concat_axis" -> <i32, name = "concat_axis", data = dense<-1> : tensor<i32>>
    %17 = atir.ConcatV2 %arg5, %10, %15, %16 : (!atir.tensor<2x5xf32, name = "output">, !atir.tensor<2x4xf32, name = "reshaped">, !atir.tensor<2x1xf32, name = "fill_out">, !atir.tensor<i32, name = "concat_axis", data = dense<-1> : tensor<i32>>) -> (!atir.tensor<2x5xf32, name = "output">)
    return
  }
}

// -----

// Positive case: (i32,i32) dtype combo, 1-D indices (hmv concat_5/6 shape) —
// resolves to the R1x1 name; the registry picks the i32/i32 specialization
// via the T1/T2 type constraints.
// CHECK-LABEL: func.func private @i32i32_r1x1_gather
// CHECK: atir.Customize
// CHECK-SAME: custom.op_name = "KPFusedEmbeddingActionIdGatherR1x1"
// CHECK-NOT: atir.Gather
// CHECK-NOT: atir.ConcatV2
module attributes {module.state = "atir"} {
  func.func private @i32i32_r1x1_gather(
      %arg0: !atir.tensor<2xi32, name = "indices1">,
      %arg1: !atir.tensor<10x4xf32, name = "params">,
      %arg2: !atir.tensor<2xi32, name = "indices2">,
      %arg3: !atir.tensor<i32, name = "pack_dim">,
      %arg4: !atir.tensor<i32, name = "pack">,
      %arg5: !atir.tensor<2x5xf32, name = "output">) attributes {annc.kernel} {
    %0 = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "inner_gather">
    %1 = atir.constant "public" @"axis_0" -> <i32, name = "axis_0", data = dense<0> : tensor<i32>>
    %2 = atir.Gather %0, %arg1, %arg0, %1 : <2x4xf32, name = "inner_gather">, <10x4xf32, name = "params">, <2xi32, name = "indices1">, <i32, name = "axis_0", data = dense<0> : tensor<i32>> -> <2x4xf32, name = "inner_gather">
    %3 = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "outer_gather">
    %4 = atir.constant "public" @"axis_1" -> <i32, name = "axis_1", data = dense<0> : tensor<i32>>
    %5 = atir.Gather %3, %2, %arg2, %4 : <2x4xf32, name = "outer_gather">, <2x4xf32, name = "inner_gather">, <2xi32, name = "indices2">, <i32, name = "axis_1", data = dense<0> : tensor<i32>> -> <2x4xf32, name = "outer_gather">
    %6 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "pack_shape">
    %7 = atir.constant "public" @"minus_one" -> <i32, name = "minus_one", data = dense<-1> : tensor<i32>>
    %8 = atir.Pack %6, (%arg3, %7) axis = 0 : <2xi32, name = "pack_shape">, !atir.tensor<i32, name = "pack_dim">, !atir.tensor<i32, name = "minus_one", data = dense<-1> : tensor<i32>> -> <2xi32, name = "pack_shape">
    %9 = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "reshaped">
    %10 = atir.Reshape %9, %5, %8 : <2x4xf32, name = "reshaped">, <2x4xf32, name = "outer_gather">, <2xi32, name = "pack_shape"> -> <2x4xf32, name = "reshaped">
    %11 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "pack_dims">
    %12 = atir.Pack %11, (%arg3, %arg4) axis = 0 : <2xi32, name = "pack_dims">, !atir.tensor<i32, name = "pack_dim">, !atir.tensor<i32, name = "pack"> -> <2xi32, name = "pack_dims">
    %13 = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "fill_out">
    %14 = atir.constant "public" @"zero" -> <f32, name = "zero", data = dense<0.0> : tensor<f32>>
    %15 = atir.Fill %13, %12, %14 : <2x1xf32, name = "fill_out">, <2xi32, name = "pack_dims">, <f32, name = "zero", data = dense<0.0> : tensor<f32>> -> <2x1xf32, name = "fill_out">
    %16 = atir.constant "public" @"concat_axis" -> <i32, name = "concat_axis", data = dense<-1> : tensor<i32>>
    %17 = atir.ConcatV2 %arg5, %10, %15, %16 : (!atir.tensor<2x5xf32, name = "output">, !atir.tensor<2x4xf32, name = "reshaped">, !atir.tensor<2x1xf32, name = "fill_out">, !atir.tensor<i32, name = "concat_axis", data = dense<-1> : tensor<i32>>) -> (!atir.tensor<2x5xf32, name = "output">)
    return
  }
}

// -----

// Negative case: inner gather uses axis=1 — the pattern must reject it.
// CHECK-LABEL: func.func private @axis_one_gather
// CHECK-NOT: atir.Customize
// CHECK: atir.ConcatV2
module attributes {module.state = "atir"} {
  func.func private @axis_one_gather(
      %arg0: !atir.tensor<2x1xi64, name = "indices1_t">,
      %arg1: !atir.tensor<10x4xf32, name = "params_t">,
      %arg2: !atir.tensor<1x2xi32, name = "indices2_t">,
      %arg3: !atir.tensor<i32, name = "pack_dim_t">,
      %arg4: !atir.tensor<i32, name = "pack_t">,
      %arg5: !atir.tensor<2x5xf32, name = "output_t">) attributes {annc.kernel} {
    %0 = "atir.buffer"() : () -> !atir.tensor<2x1x4xf32, name = "inner_gather_t">
    %1 = atir.constant "public" @"axis_bad" -> <i32, name = "axis_bad", data = dense<1> : tensor<i32>>
    %2 = atir.Gather %0, %arg1, %arg0, %1 : <2x1x4xf32, name = "inner_gather_t">, <10x4xf32, name = "params_t">, <2x1xi64, name = "indices1_t">, <i32, name = "axis_bad", data = dense<1> : tensor<i32>> -> <2x1x4xf32, name = "inner_gather_t">
    %3 = "atir.buffer"() : () -> !atir.tensor<1x2x1x4xf32, name = "outer_gather_t">
    %4 = atir.constant "public" @"axis_ok" -> <i32, name = "axis_ok", data = dense<0> : tensor<i32>>
    %5 = atir.Gather %3, %2, %arg2, %4 : <1x2x1x4xf32, name = "outer_gather_t">, <2x1x4xf32, name = "inner_gather_t">, <1x2xi32, name = "indices2_t">, <i32, name = "axis_ok", data = dense<0> : tensor<i32>> -> <1x2x1x4xf32, name = "outer_gather_t">
    %6 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "pack_shape_t">
    %7 = atir.constant "public" @"minus_one_t" -> <i32, name = "minus_one_t", data = dense<-1> : tensor<i32>>
    %8 = atir.Pack %6, (%arg3, %7) axis = 0 : <2xi32, name = "pack_shape_t">, !atir.tensor<i32, name = "pack_dim_t">, !atir.tensor<i32, name = "minus_one_t", data = dense<-1> : tensor<i32>> -> <2xi32, name = "pack_shape_t">
    %9 = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "reshaped_t">
    %10 = atir.Reshape %9, %5, %8 : <2x4xf32, name = "reshaped_t">, <1x2x1x4xf32, name = "outer_gather_t">, <2xi32, name = "pack_shape_t"> -> <2x4xf32, name = "reshaped_t">
    %11 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "pack_dims_t">
    %12 = atir.Pack %11, (%arg3, %arg4) axis = 0 : <2xi32, name = "pack_dims_t">, !atir.tensor<i32, name = "pack_dim_t">, !atir.tensor<i32, name = "pack_t"> -> <2xi32, name = "pack_dims_t">
    %13 = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "fill_out_t">
    %14 = atir.constant "public" @"zero_t" -> <f32, name = "zero_t", data = dense<0.0> : tensor<f32>>
    %15 = atir.Fill %13, %12, %14 : <2x1xf32, name = "fill_out_t">, <2xi32, name = "pack_dims_t">, <f32, name = "zero_t", data = dense<0.0> : tensor<f32>> -> <2x1xf32, name = "fill_out_t">
    %16 = atir.constant "public" @"concat_axis_t" -> <i32, name = "concat_axis_t", data = dense<-1> : tensor<i32>>
    %17 = atir.ConcatV2 %arg5, %10, %15, %16 : (!atir.tensor<2x5xf32, name = "output_t">, !atir.tensor<2x4xf32, name = "reshaped_t">, !atir.tensor<2x1xf32, name = "fill_out_t">, !atir.tensor<i32, name = "concat_axis_t", data = dense<-1> : tensor<i32>>) -> (!atir.tensor<2x5xf32, name = "output_t">)
    return
  }
}
