// RUN: annc-asm --split-input-file %s -atir-fast-codegen | FileCheck %s
// RUN: annc-asm --split-input-file %s -atir-fast-codegen='enable-kdnn=true' | FileCheck %s --check-prefix=KDNN
// FastCodegen: MatmulToCustomCallRewrite pattern，--enable-kdnn 给 module 加 annc.enable_kdnn 标记

// MatMul -> Customize
// CHECK-LABEL: func @main
// KDNN-LABEL: func @main
// KDNN: annc.enable_kdnn = true
module @main attributes {module.state = "atir"} {
  func.func @main(
      %c: !atir.tensor<4x4xf32, name = "out">,
      %a: !atir.tensor<4x8xf32, name = "lhs">,
      %b: !atir.tensor<8x4xf32, name = "rhs">) -> !atir.tensor<4x4xf32, name = "out"> {
    // CHECK: atir.Customize
    // CHECK-SAME: custom.op_name = "MatMul"
    // CHECK-NOT: "atir.MatMul"
    %0 = "atir.MatMul"(%c, %a, %b) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.0 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<4x4xf32, name = "out">, !atir.tensor<4x8xf32, name = "lhs">, !atir.tensor<8x4xf32, name = "rhs">) -> !atir.tensor<4x4xf32, name = "out">
    return %0 : !atir.tensor<4x4xf32, name = "out">
  }
}

// -----

// 没有 MatMul 的函数不受影响
// CHECK-LABEL: func @no_matmul
module @no_mm attributes {module.state = "atir"} {
  func.func @no_matmul(
      %c: !atir.tensor<4x4xf32, name = "out">) -> !atir.tensor<4x4xf32, name = "out"> {
    // CHECK: return
    // CHECK-NOT: atir.Customize
    return %c : !atir.tensor<4x4xf32, name = "out">
  }
}