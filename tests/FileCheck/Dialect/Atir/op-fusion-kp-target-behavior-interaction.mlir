// RUN: annc-opt --split-input-file %s -atir-config-fusion="config=%S/../../../../configs/fusion/kp_embedding_patterns.json run-builtin-after=false warn-limit=0" | FileCheck %s --check-prefix=CONFIG
// Config fusion first level for target-behavior interaction: the
// BatchMatMul->Add->Abs->Add->Mul(0.5) + Tile + Sub + Mul subgraph in main
// is extracted into a private kernel func with annc.kernel +
// fusion.metadata while preserving the legacy output-buffer ABI.

// CONFIG: func.func private @fused_kp_target_behavior_interaction_
// CONFIG-SAME: annc.kernel
// CONFIG-SAME: fusion.pattern = "kp_target_behavior_interaction"
// CONFIG: func.func @main
// CONFIG: atir.ConcatV2
// CONFIG-SAME: annc.fusion_materialized

// CHECK-LABEL: func.func @main
// CHECK: atir.ConcatV2
// CHECK-SAME: annc.fusion_materialized
// CHECK: func.func private @fused_kp_target_behavior_interaction_
// CHECK-SAME: annc.kernel
// CHECK-SAME: fusion.metadata
// CHECK-SAME: role = "dynamic"
// CHECK-SAME: role = "fixed"
// CHECK-SAME: kernel_name = "fused_kp_target_behavior_interaction_
// CHECK-SAME: outputs = [{dtype = "f32"
// CHECK-SAME: llvm.emit_c_interface
func.func @main(
    %arg0: !atir.tensor<2x3x4xf32, name = "batch_input">,
    %arg1: !atir.tensor<4x5xf32, name = "weight">,
    %arg2: !atir.tensor<5xf32, name = "bias">,
    %arg3: !atir.tensor<2x3x5xf32, name = "tile_input">)
    -> !atir.tensor<2x3x20xf32, name = "output"> {
  %0 = "atir.buffer"() : () -> !atir.tensor<2x3x5xf32, name = "bmm_out">
  %1 = "atir.BatchMatMul"(%0, %arg0, %arg1) <{transposeA = false, transposeB = false}> : (!atir.tensor<2x3x5xf32, name = "bmm_out">, !atir.tensor<2x3x4xf32, name = "batch_input">, !atir.tensor<4x5xf32, name = "weight">) -> !atir.tensor<2x3x5xf32, name = "bmm_out">
  %2 = "atir.buffer"() : () -> !atir.tensor<2x3x5xf32, name = "add21_out">
  %3 = "atir.Add"(%2, %1, %arg2) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> : (!atir.tensor<2x3x5xf32, name = "add21_out">, !atir.tensor<2x3x5xf32, name = "bmm_out">, !atir.tensor<5xf32, name = "bias">) -> !atir.tensor<2x3x5xf32, name = "add21_out">
  %4 = "atir.buffer"() : () -> !atir.tensor<2x3x5xf32, name = "abs_out">
  %5 = "atir.Abs"(%4, %3) : (!atir.tensor<2x3x5xf32, name = "abs_out">, !atir.tensor<2x3x5xf32, name = "add21_out">) -> !atir.tensor<2x3x5xf32, name = "abs_out">
  %6 = "atir.buffer"() : () -> !atir.tensor<2x3x5xf32, name = "add22_out">
  %7 = "atir.Add"(%6, %5, %3) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> : (!atir.tensor<2x3x5xf32, name = "add22_out">, !atir.tensor<2x3x5xf32, name = "abs_out">, !atir.tensor<2x3x5xf32, name = "add21_out">) -> !atir.tensor<2x3x5xf32, name = "add22_out">
  %8 = "atir.buffer"() : () -> !atir.tensor<2x3x5xf32, name = "realdiv_out">
  %9 = atir.constant "public" @"half" -> <f32, name = "half", data = dense<5.000000e-01> : tensor<f32>>
  %10 = "atir.Mul"(%8, %7, %9) : (!atir.tensor<2x3x5xf32, name = "realdiv_out">, !atir.tensor<2x3x5xf32, name = "add22_out">, !atir.tensor<f32, name = "half", data = dense<5.000000e-01> : tensor<f32>>) -> !atir.tensor<2x3x5xf32, name = "realdiv_out">
  %11 = "atir.buffer"() : () -> !atir.tensor<2x3x5xf32, name = "tile_out">
  %12 = atir.constant "public" @"multiples" -> <3xi32, name = "multiples", data = dense<[1, 1, 1]> : tensor<3xi32>>
  %13 = "atir.Tile"(%11, %arg3, %12) : (!atir.tensor<2x3x5xf32, name = "tile_out">, !atir.tensor<2x3x5xf32, name = "tile_input">, !atir.tensor<3xi32, name = "multiples", data = dense<[1, 1, 1]> : tensor<3xi32>>) -> !atir.tensor<2x3x5xf32, name = "tile_out">
  %14 = "atir.buffer"() : () -> !atir.tensor<2x3x5xf32, name = "sub_out">
  %15 = "atir.Sub"(%14, %10, %13) : (!atir.tensor<2x3x5xf32, name = "sub_out">, !atir.tensor<2x3x5xf32, name = "realdiv_out">, !atir.tensor<2x3x5xf32, name = "tile_out">) -> !atir.tensor<2x3x5xf32, name = "sub_out">
  %16 = "atir.buffer"() : () -> !atir.tensor<2x3x5xf32, name = "mul_out">
  %17 = "atir.Mul"(%16, %10, %13) : (!atir.tensor<2x3x5xf32, name = "mul_out">, !atir.tensor<2x3x5xf32, name = "realdiv_out">, !atir.tensor<2x3x5xf32, name = "tile_out">) -> !atir.tensor<2x3x5xf32, name = "mul_out">
  %18 = "atir.buffer"() : () -> !atir.tensor<2x3x20xf32, name = "output">
  %19 = atir.constant "public" @"concat_axis" -> <i32, name = "concat_axis", data = dense<-1> : tensor<i32>>
  %20 = "atir.ConcatV2"(%18, %10, %13, %15, %17, %19) : (!atir.tensor<2x3x20xf32, name = "output">, !atir.tensor<2x3x5xf32, name = "realdiv_out">, !atir.tensor<2x3x5xf32, name = "tile_out">, !atir.tensor<2x3x5xf32, name = "sub_out">, !atir.tensor<2x3x5xf32, name = "mul_out">, !atir.tensor<i32, name = "concat_axis", data = dense<-1> : tensor<i32>>) -> !atir.tensor<2x3x20xf32, name = "output">
  return %20 : !atir.tensor<2x3x20xf32, name = "output">
}

// -----

// Negative case: Mul constant 0.25 — no fusion, subgraph preserved.
// CHECK-LABEL: func.func @bad_half
// CHECK-NOT: call @fused_kp_target_behavior_interaction_
// CHECK: atir.ConcatV2
func.func @bad_half(
    %arg0: !atir.tensor<2x3x4xf32, name = "batch_input_t">,
    %arg1: !atir.tensor<4x5xf32, name = "weight_t">,
    %arg2: !atir.tensor<5xf32, name = "bias_t">,
    %arg3: !atir.tensor<2x3x5xf32, name = "tile_input_t">)
    -> !atir.tensor<2x3x20xf32, name = "output_t"> {
  %0 = "atir.buffer"() : () -> !atir.tensor<2x3x5xf32, name = "bmm_out_t">
  %1 = "atir.BatchMatMul"(%0, %arg0, %arg1) <{transposeA = false, transposeB = false}> : (!atir.tensor<2x3x5xf32, name = "bmm_out_t">, !atir.tensor<2x3x4xf32, name = "batch_input_t">, !atir.tensor<4x5xf32, name = "weight_t">) -> !atir.tensor<2x3x5xf32, name = "bmm_out_t">
  %2 = "atir.buffer"() : () -> !atir.tensor<2x3x5xf32, name = "add21_out_t">
  %3 = "atir.Add"(%2, %1, %arg2) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> : (!atir.tensor<2x3x5xf32, name = "add21_out_t">, !atir.tensor<2x3x5xf32, name = "bmm_out_t">, !atir.tensor<5xf32, name = "bias_t">) -> !atir.tensor<2x3x5xf32, name = "add21_out_t">
  %4 = "atir.buffer"() : () -> !atir.tensor<2x3x5xf32, name = "abs_out_t">
  %5 = "atir.Abs"(%4, %3) : (!atir.tensor<2x3x5xf32, name = "abs_out_t">, !atir.tensor<2x3x5xf32, name = "add21_out_t">) -> !atir.tensor<2x3x5xf32, name = "abs_out_t">
  %6 = "atir.buffer"() : () -> !atir.tensor<2x3x5xf32, name = "add22_out_t">
  %7 = "atir.Add"(%6, %5, %3) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> : (!atir.tensor<2x3x5xf32, name = "add22_out_t">, !atir.tensor<2x3x5xf32, name = "abs_out_t">, !atir.tensor<2x3x5xf32, name = "add21_out_t">) -> !atir.tensor<2x3x5xf32, name = "add22_out_t">
  %8 = "atir.buffer"() : () -> !atir.tensor<2x3x5xf32, name = "realdiv_out_t">
  %9 = atir.constant "public" @"quarter" -> <f32, name = "quarter", data = dense<2.500000e-01> : tensor<f32>>
  %10 = "atir.Mul"(%8, %7, %9) : (!atir.tensor<2x3x5xf32, name = "realdiv_out_t">, !atir.tensor<2x3x5xf32, name = "add22_out_t">, !atir.tensor<f32, name = "quarter", data = dense<2.500000e-01> : tensor<f32>>) -> !atir.tensor<2x3x5xf32, name = "realdiv_out_t">
  %11 = "atir.buffer"() : () -> !atir.tensor<2x3x5xf32, name = "tile_out_t">
  %12 = atir.constant "public" @"multiples_t" -> <3xi32, name = "multiples_t", data = dense<[1, 1, 1]> : tensor<3xi32>>
  %13 = "atir.Tile"(%11, %arg3, %12) : (!atir.tensor<2x3x5xf32, name = "tile_out_t">, !atir.tensor<2x3x5xf32, name = "tile_input_t">, !atir.tensor<3xi32, name = "multiples_t", data = dense<[1, 1, 1]> : tensor<3xi32>>) -> !atir.tensor<2x3x5xf32, name = "tile_out_t">
  %14 = "atir.buffer"() : () -> !atir.tensor<2x3x5xf32, name = "sub_out_t">
  %15 = "atir.Sub"(%14, %10, %13) : (!atir.tensor<2x3x5xf32, name = "sub_out_t">, !atir.tensor<2x3x5xf32, name = "realdiv_out_t">, !atir.tensor<2x3x5xf32, name = "tile_out_t">) -> !atir.tensor<2x3x5xf32, name = "sub_out_t">
  %16 = "atir.buffer"() : () -> !atir.tensor<2x3x5xf32, name = "mul_out_t">
  %17 = "atir.Mul"(%16, %10, %13) : (!atir.tensor<2x3x5xf32, name = "mul_out_t">, !atir.tensor<2x3x5xf32, name = "realdiv_out_t">, !atir.tensor<2x3x5xf32, name = "tile_out_t">) -> !atir.tensor<2x3x5xf32, name = "mul_out_t">
  %18 = "atir.buffer"() : () -> !atir.tensor<2x3x20xf32, name = "output_t">
  %19 = atir.constant "public" @"concat_axis_t" -> <i32, name = "concat_axis_t", data = dense<-1> : tensor<i32>>
  %20 = "atir.ConcatV2"(%18, %10, %13, %15, %17, %19) : (!atir.tensor<2x3x20xf32, name = "output_t">, !atir.tensor<2x3x5xf32, name = "realdiv_out_t">, !atir.tensor<2x3x5xf32, name = "tile_out_t">, !atir.tensor<2x3x5xf32, name = "sub_out_t">, !atir.tensor<2x3x5xf32, name = "mul_out_t">, !atir.tensor<i32, name = "concat_axis_t", data = dense<-1> : tensor<i32>>) -> !atir.tensor<2x3x20xf32, name = "output_t">
  return %20 : !atir.tensor<2x3x20xf32, name = "output_t">
}
