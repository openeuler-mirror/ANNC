// RUN: annc-asm --split-input-file %s -atir-fast-codegen | FileCheck %s
// RUN: annc-asm --split-input-file %s -atir-fast-codegen='enable-kdnn=true' | FileCheck %s --check-prefix=KDNN
// RUN: annc-asm --split-input-file %s -atir-fast-codegen='enable-custom-ops=MatMul' | FileCheck %s --check-prefix=ENABLED
// RUN: annc-asm --split-input-file %s -atir-fast-codegen='disable-custom-ops=MatMul' | FileCheck %s --check-prefix=DISABLE
// RUN: annc-asm --split-input-file %s -atir-fast-codegen='enable-kdnn=true enable-custom-ops=MatMulAdd' | FileCheck %s --check-prefix=ALLOW
// RUN: annc-asm --split-input-file %s -atir-fast-codegen='enable-custom-ops=MatMul disable-custom-ops=MatMul' | FileCheck %s --check-prefix=DISABLE
// RUN: env ANNC_FAST_CODEGEN_ENABLE_CUSTOM_OPS=MatMul annc-asm --split-input-file %s -atir-fast-codegen | FileCheck %s --check-prefix=ENABLED
// RUN: env ANNC_FAST_CODEGEN_ENABLE_CUSTOM_OPS=MatMulAdd annc-asm --split-input-file %s -atir-fast-codegen='enable-kdnn=true' | FileCheck %s --check-prefix=ALLOW
// RUN: env ANNC_FAST_CODEGEN_DISABLE_CUSTOM_OPS=Ignored,MatMul annc-asm --split-input-file %s -atir-fast-codegen | FileCheck %s --check-prefix=DISABLE
// RUN: env ANNC_FAST_CODEGEN_ENABLE_CUSTOM_OPS=MatMul annc-asm --split-input-file %s -atir-fast-codegen='enable-kdnn=true enable-custom-ops=MatMulAdd' | FileCheck %s --check-prefix=ALLOW
// RUN: env ANNC_FAST_CODEGEN_DISABLE_CUSTOM_OPS=MatMul annc-asm --split-input-file %s -atir-fast-codegen='enable-custom-ops=MatMul' | FileCheck %s --check-prefix=DISABLE
// RUN: annc-asm --split-input-file %s -atir-fast-codegen='enable-kdnn=true enable-custom-ops=MatMulAdd' | FileCheck %s --check-prefix=FUSED-ALLOW
// RUN: annc-asm --split-input-file %s -atir-fast-codegen='enable-kdnn=true enable-custom-ops=MatMulAddRelu' | FileCheck %s --check-prefix=RELU-ALLOW
// FastCodegen: MatmulToCustomCallRewrite pattern，--enable-kdnn 给 module 加 annc.enable_kdnn 标记

// Ordinary GEMM stays on the generated path. An allowlist must not resolve
// an already migrated builtin through its removed legacy ABI.
// CHECK-LABEL: func @main
// KDNN-LABEL: module @main attributes
// KDNN-SAME: annc.enable_kdnn = true
// KDNN: func.func @main
// KDNN: "atir.MatMul"
// KDNN-NOT: atir.Customize
// KDNN: return
// ENABLED-LABEL: func @main
module @main attributes {module.state = "atir"} {
  func.func @main(
      %c: !atir.tensor<4x4xf32, name = "out">,
      %a: !atir.tensor<4x8xf32, name = "lhs">,
      %b: !atir.tensor<8x4xf32, name = "rhs">) -> !atir.tensor<4x4xf32, name = "out"> {
    // CHECK: "atir.MatMul"
    // CHECK-NOT: atir.Customize
    // CHECK: return
    // ENABLED: "atir.MatMul"
    // ENABLED-NOT: atir.Customize
    // ENABLED: return
    // DISABLE: "atir.MatMul"
    // DISABLE-NOT: atir.Customize
    // DISABLE: return
    // ALLOW: "atir.MatMul"
    // ALLOW-NOT: atir.Customize
    // ALLOW: return
    %0 = "atir.MatMul"(%c, %a, %b) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.0 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<4x4xf32, name = "out">, !atir.tensor<4x8xf32, name = "lhs">, !atir.tensor<8x4xf32, name = "rhs">) -> !atir.tensor<4x4xf32, name = "out">
    return %0 : !atir.tensor<4x4xf32, name = "out">
  }
}

// -----

// CHECK-LABEL: func @matmul_add
// CHECK: "atir.MatMul"
// CHECK: "atir.Add"
// CHECK-NOT: atir.Customize
// CHECK: return
// KDNN-LABEL: module @fused attributes
// KDNN-SAME: annc.enable_kdnn = true
// KDNN: func.func @matmul_add
// KDNN: "atir.MatMul"
// KDNN: "atir.Add"
// KDNN-NOT: atir.Customize
// KDNN: return
// ALLOW-LABEL: func @matmul_add
// ALLOW: "atir.MatMul"
// ALLOW: "atir.Add"
// ALLOW-NOT: atir.Customize
// ALLOW: return
// FUSED-ALLOW-LABEL: func @matmul_add
// FUSED-ALLOW: "atir.MatMul"
// FUSED-ALLOW: "atir.Add"
// FUSED-ALLOW-NOT: atir.Customize
// FUSED-ALLOW: return
module @fused attributes {module.state = "atir"} {
  func.func @matmul_add(
      %c: !atir.tensor<4x4xf32, name = "out">,
      %a: !atir.tensor<4x8xf32, name = "lhs">,
      %b: !atir.tensor<8x4xf32, name = "rhs">,
      %bias: !atir.tensor<4xf32, name = "bias">)
      -> !atir.tensor<4x4xf32, name = "out"> {
    %tmp = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
    %matmul = "atir.MatMul"(%tmp, %a, %b) <{
      left_transpose = false,
      output_transpose = false,
      right_transpose = false,
      withBias = false
    }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32, name = "lhs">,
          !atir.tensor<8x4xf32, name = "rhs">)
        -> !atir.tensor<4x4xf32>
    %add = "atir.Add"(%c, %matmul, %bias) <{
      do_relu = false,
      relu_limit = -1.0 : f32
    }> : (!atir.tensor<4x4xf32, name = "out">, !atir.tensor<4x4xf32>,
          !atir.tensor<4xf32, name = "bias">)
        -> !atir.tensor<4x4xf32, name = "out">
    return %add : !atir.tensor<4x4xf32, name = "out">
  }
}

// -----

// CHECK-LABEL: func @matmul_add_relu
// CHECK: "atir.MatMul"
// CHECK: "atir.Add"
// CHECK: "atir.Relu"
// CHECK-NOT: atir.Customize
// CHECK: return
// KDNN-LABEL: module @fused_relu attributes
// KDNN-SAME: annc.enable_kdnn = true
// KDNN: func.func @matmul_add_relu
// KDNN: "atir.MatMul"
// KDNN: "atir.Add"
// KDNN: "atir.Relu"
// KDNN-NOT: atir.Customize
// KDNN: return
// RELU-ALLOW-LABEL: func @matmul_add_relu
// RELU-ALLOW: "atir.MatMul"
// RELU-ALLOW: "atir.Add"
// RELU-ALLOW: "atir.Relu"
// RELU-ALLOW-NOT: atir.Customize
// RELU-ALLOW: return
module @fused_relu attributes {module.state = "atir"} {
  func.func @matmul_add_relu(
      %c: !atir.tensor<4x4xf32, name = "out">,
      %a: !atir.tensor<4x8xf32, name = "lhs">,
      %b: !atir.tensor<8x4xf32, name = "rhs">,
      %bias: !atir.tensor<4xf32, name = "bias">)
      -> !atir.tensor<4x4xf32, name = "out"> {
    %matmul_buffer = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
    %matmul = "atir.MatMul"(%matmul_buffer, %a, %b) <{
      left_transpose = false,
      output_transpose = false,
      right_transpose = false,
      withBias = false
    }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32, name = "lhs">,
          !atir.tensor<8x4xf32, name = "rhs">)
        -> !atir.tensor<4x4xf32>
    %add_buffer = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
    %add = "atir.Add"(%add_buffer, %matmul, %bias) <{
      do_relu = false,
      relu_limit = -1.0 : f32
    }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>,
          !atir.tensor<4xf32, name = "bias">) -> !atir.tensor<4x4xf32>
    %relu = "atir.Relu"(%c, %add) <{relu_limit = -1.0 : f32}> :
        (!atir.tensor<4x4xf32, name = "out">, !atir.tensor<4x4xf32>)
        -> !atir.tensor<4x4xf32, name = "out">
    return %relu : !atir.tensor<4x4xf32, name = "out">
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
