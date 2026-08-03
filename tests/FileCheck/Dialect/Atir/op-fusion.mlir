// RUN: annc-opt --split-input-file %s -atir-op-fusion | FileCheck %s
// OpFusion: 将 MatMulOp 替换为 call @fused_* + private kernel 函数
// MatMul operand 顺序: (C=输出buffer, lhs, rhs)

// main 函数内部有 matmul，触发 fusion
// CHECK-LABEL: func @main
func.func @main(
    %a: !atir.tensor<64x128xf32>,
    %b: !atir.tensor<128x64xf32>,
    %c: !atir.tensor<64x64xf32>) -> !atir.tensor<64x64xf32> {
  %0 = "atir.MatMul"(%c, %a, %b) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.0 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<64x64xf32>, !atir.tensor<64x128xf32>, !atir.tensor<128x64xf32>) -> !atir.tensor<64x64xf32>
  // CHECK: call @fused_matmul
  // CHECK: func.func private @fused_matmul
  // CHECK: fusion.metadata
  // CHECK: args = [
  // CHECK: outputs = [
  // CHECK-NOT: tf.inputs
  // CHECK-NOT: tf.input_shapes
  // CHECK-NOT: tf.output
  // CHECK-NOT: Nconstants
  // CHECK-NOT: Nfixed
  // CHECK-NOT: Ndynamic
  // CHECK-NOT: num_outputs
  // CHECK-NOT: input_ranks
  // CHECK-NOT: output_ranks
  // CHECK-NOT: output_shapes
  return %0 : !atir.tensor<64x64xf32>
}

// -----

// 函数不叫 main，不触发
// CHECK-LABEL: func @other
func.func @other(
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>,
    %c: !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32> {
  %0 = "atir.MatMul"(%c, %a, %b) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.0 : f32, right_transpose = false, withBias = false}> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>, !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  // CHECK: "atir.MatMul"
  // CHECK-NOT: call @fused_matmul
  return %0 : !atir.tensor<4x4xf32>
}

// -----

// main 函数无 MatMul，不触发
// CHECK-LABEL: func @main
func.func @main(
    %c: !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32> {
  // CHECK: return
  // CHECK-NOT: call @fused_
  return %c : !atir.tensor<4x4xf32>
}
