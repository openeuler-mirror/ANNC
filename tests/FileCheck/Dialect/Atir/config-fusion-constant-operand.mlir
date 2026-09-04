// RUN: annc-opt --split-input-file %s -atir-config-fusion="config=%S/config-fusion-constant-operand.json run-builtin-after=false" | FileCheck %s

// CHECK-LABEL: func.func private @fused_mul_half_
// CHECK: fusion.pattern = "mul_half"
// CHECK-LABEL: func.func @main
// CHECK: call @fused_mul_half_
// CHECK-NOT: "atir.Mul"
// CHECK: return
func.func @main(
    %x: !atir.tensor<2x3xf32, name = "x">,
    %out: !atir.tensor<2x3xf32, name = "y">)
    -> !atir.tensor<2x3xf32, name = "y"> {
  %scale = atir.constant "public" @"scale" -> <f32, name = "scale", data = dense<5.000000e-01> : tensor<f32>>
  %y = "atir.Mul"(%out, %x, %scale) : (
      !atir.tensor<2x3xf32, name = "y">,
      !atir.tensor<2x3xf32, name = "x">,
      !atir.tensor<f32, name = "scale", data = dense<5.000000e-01> : tensor<f32>>) ->
      !atir.tensor<2x3xf32, name = "y">
  return %y : !atir.tensor<2x3xf32, name = "y">
}

// -----

// CHECK-LABEL: func.func @main
// CHECK-NOT: call @fused_mul_half_
// CHECK: "atir.Mul"
func.func @main(
    %x: !atir.tensor<2x3xf32, name = "x_bad">,
    %out: !atir.tensor<2x3xf32, name = "y_bad">)
    -> !atir.tensor<2x3xf32, name = "y_bad"> {
  %scale = atir.constant "public" @"scale_bad" -> <f32, name = "scale_bad", data = dense<2.500000e-01> : tensor<f32>>
  %y = "atir.Mul"(%out, %x, %scale) : (
      !atir.tensor<2x3xf32, name = "y_bad">,
      !atir.tensor<2x3xf32, name = "x_bad">,
      !atir.tensor<f32, name = "scale_bad", data = dense<2.500000e-01> : tensor<f32>>) ->
      !atir.tensor<2x3xf32, name = "y_bad">
  return %y : !atir.tensor<2x3xf32, name = "y_bad">
}
