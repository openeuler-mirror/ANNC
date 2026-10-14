// RUN: annc-opt --split-input-file %s -atir-config-fusion="config=%S/../../../../configs/fusion/kp_embedding_patterns.json run-builtin-after=false warn-limit=0" | FileCheck %s --check-prefix=CONFIG
// Config fusion first level for the embedding action-id gather: the
// Gather->Gather->Reshape + Fill subgraph in main is extracted into a
// private kernel func with annc.kernel + fusion.metadata. The legacy output
// buffer ABI keeps the main graph for GraphDef rewriting.

// CONFIG: func.func private @fused_kp_embedding_action_id_gather_
// CONFIG-SAME: annc.kernel
// CONFIG-SAME: fusion.pattern = "kp_embedding_action_id_gather"
// CONFIG: func.func @main
// CONFIG: atir.ConcatV2
// CONFIG-SAME: annc.fusion_materialized

// CHECK-LABEL: func.func @main
// CHECK: atir.ConcatV2
// CHECK-SAME: annc.fusion_materialized
// CHECK: func.func private @fused_kp_embedding_action_id_gather_
// CHECK-SAME: annc.kernel
// CHECK-SAME: fusion.metadata
// CHECK-SAME: role = "dynamic"
// CHECK-SAME: role = "fixed"
// CHECK-SAME: kernel_name = "fused_kp_embedding_action_id_gather_
// CHECK-SAME: outputs = [{dtype = "f32"
// CHECK-SAME: llvm.emit_c_interface
func.func @main(
    %arg0: !atir.tensor<2x1xi64, name = "indices1">,
    %arg1: !atir.tensor<10x4xf32, name = "params">,
    %arg2: !atir.tensor<1x2xi32, name = "indices2">,
    %arg3: !atir.tensor<i32, name = "pack_dim">,
    %arg4: !atir.tensor<i32, name = "pack">)
    -> !atir.tensor<2x5xf32, name = "output"> {
  %0 = "atir.buffer"() : () -> !atir.tensor<2x1x4xf32, name = "inner_gather">
  %1 = atir.constant "public" @"axis_0" -> <i32, name = "axis_0", data = dense<0> : tensor<i32>>
  %2 = atir.Gather %0, %arg1, %arg0, %1 : <2x1x4xf32, name = "inner_gather">, <10x4xf32, name = "params">, <2x1xi64, name = "indices1">, <i32, name = "axis_0", data = dense<0> : tensor<i32>> -> <2x1x4xf32, name = "inner_gather">
  %3 = "atir.buffer"() : () -> !atir.tensor<1x2x1x4xf32, name = "outer_gather">
  %4 = atir.constant "public" @"axis_1" -> <i32, name = "axis_1", data = dense<0> : tensor<i32>>
  %5 = atir.Gather %3, %2, %arg2, %4 : <1x2x1x4xf32, name = "outer_gather">, <2x1x4xf32, name = "inner_gather">, <1x2xi32, name = "indices2">, <i32, name = "axis_1", data = dense<0> : tensor<i32>> -> <1x2x1x4xf32, name = "outer_gather">
  %6 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "pack_shape">
  %7 = atir.constant "public" @"minus_one" -> <i32, name = "minus_one", data = dense<-1> : tensor<i32>>
  %8 = atir.Pack %6, (%arg3, %7) axis = 0 : <2xi32, name = "pack_shape">, !atir.tensor<i32, name = "pack_dim">, !atir.tensor<i32, name = "minus_one", data = dense<-1> : tensor<i32>> -> <2xi32, name = "pack_shape">
  %9 = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "reshaped">
  %10 = atir.Reshape %9, %5, %8 : <2x4xf32, name = "reshaped">, <1x2x1x4xf32, name = "outer_gather">, <2xi32, name = "pack_shape"> -> <2x4xf32, name = "reshaped">
  %11 = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "pack_dims">
  %12 = atir.Pack %11, (%arg3, %arg4) axis = 0 : <2xi32, name = "pack_dims">, !atir.tensor<i32, name = "pack_dim">, !atir.tensor<i32, name = "pack"> -> <2xi32, name = "pack_dims">
  %13 = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "fill_out">
  %14 = atir.constant "public" @"zero" -> <f32, name = "zero", data = dense<0.0> : tensor<f32>>
  %15 = atir.Fill %13, %12, %14 : <2x1xf32, name = "fill_out">, <2xi32, name = "pack_dims">, <f32, name = "zero", data = dense<0.0> : tensor<f32>> -> <2x1xf32, name = "fill_out">
  %16 = "atir.buffer"() : () -> !atir.tensor<2x5xf32, name = "output">
  %17 = atir.constant "public" @"concat_axis" -> <i32, name = "concat_axis", data = dense<-1> : tensor<i32>>
  %18 = atir.ConcatV2 %16, %10, %15, %17 : (!atir.tensor<2x5xf32, name = "output">, !atir.tensor<2x4xf32, name = "reshaped">, !atir.tensor<2x1xf32, name = "fill_out">, !atir.tensor<i32, name = "concat_axis", data = dense<-1> : tensor<i32>>) -> (!atir.tensor<2x5xf32, name = "output">)
  return %18 : !atir.tensor<2x5xf32, name = "output">
}

// -----

// Negative case: inner gather uses axis=1 — no fusion, subgraph preserved.
// CHECK-LABEL: func.func @axis_one_gather
// CHECK-NOT: call @fused_kp_embedding_action_id_gather_
// CHECK: atir.ConcatV2
func.func @axis_one_gather(
    %arg0: !atir.tensor<2x1xi64, name = "indices1_t">,
    %arg1: !atir.tensor<10x4xf32, name = "params_t">,
    %arg2: !atir.tensor<1x2xi32, name = "indices2_t">,
    %arg3: !atir.tensor<i32, name = "pack_dim_t">,
    %arg4: !atir.tensor<i32, name = "pack_t">)
    -> !atir.tensor<2x5xf32, name = "output_t"> {
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
  %16 = "atir.buffer"() : () -> !atir.tensor<2x5xf32, name = "output_t">
  %17 = atir.constant "public" @"concat_axis_t" -> <i32, name = "concat_axis_t", data = dense<-1> : tensor<i32>>
  %18 = atir.ConcatV2 %16, %10, %15, %17 : (!atir.tensor<2x5xf32, name = "output_t">, !atir.tensor<2x4xf32, name = "reshaped_t">, !atir.tensor<2x1xf32, name = "fill_out_t">, !atir.tensor<i32, name = "concat_axis_t", data = dense<-1> : tensor<i32>>) -> (!atir.tensor<2x5xf32, name = "output_t">)
  return %18 : !atir.tensor<2x5xf32, name = "output_t">
}
