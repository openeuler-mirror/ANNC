// RUN: annc-asm --split-input-file %s -atir-gemm-epilogue-fusion -aarch64-atir-gemm-epilogue-commit -convert-atir-to-linalg | FileCheck %s --implicit-check-not=annc.gemm.epilogue.candidate
// ConvertAtirToLinalg: atir op -> linalg

// CHECK-LABEL: func @matmul_to_linalg
// CHECK: %[[BUFFER:.*]] = memref.alloca() : memref<4x4xf32>
// CHECK: bufferization.to_tensor %[[BUFFER]] restrict writable
// CHECK: linalg.matmul
// CHECK-NOT: "atir.MatMul"
func.func @matmul_to_linalg(
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32> {
  %c = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %0 = "atir.MatMul"(%c, %a, %b) <{left_transpose = false, output_transpose = false, right_transpose = false, withBias = false}> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>, !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  return %0 : !atir.tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @empty
func.func @empty() {
  return
}

// -----

// CHECK-LABEL: func @buffer_to_alloca
// CHECK: %[[BUFFER:.*]] = memref.alloca() : memref<4x4xf32>
// CHECK: bufferization.to_tensor %[[BUFFER]] restrict writable
// CHECK-NOT: "atir.buffer"
func.func @buffer_to_alloca() {
  %0 = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  return
}

// -----

// Commit must precede Linalg conversion. The ordered add/relu epilogue stays
// in metadata for the AArch64 backend; the generic body only accumulates A*B.
// CHECK-LABEL: func @matmul_bias_relu_to_linalg(
// CHECK-SAME: %[[C:[^:]+]]: memref<4x4xf32>, %[[A:[^:]+]]: memref<4x8xf32>, %[[B:[^:]+]]: memref<8x4xf32>, %[[BIAS:[^:]+]]: memref<4xf32>)
// CHECK-DAG: %[[BIAS_T:.*]] = bufferization.to_tensor %[[BIAS]] restrict writable
// CHECK-DAG: %[[B_T:.*]] = bufferization.to_tensor %[[B]] restrict writable
// CHECK-DAG: %[[A_T:.*]] = bufferization.to_tensor %[[A]] restrict writable
// CHECK-DAG: %[[C_T:.*]] = bufferization.to_tensor %[[C]] restrict writable
// CHECK-NOT: linalg.matmul
// CHECK-NOT: arith.maxnumf
// CHECK: %[[GEMM:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[A_T]], %[[B_T]], %[[BIAS_T]] : tensor<4x8xf32>, tensor<8x4xf32>, tensor<4xf32>) outs(%[[C_T]] : tensor<4x4xf32>)
// CHECK-SAME: annc.gemm, annc.gemm.epilogue = {output_role = "main_d", steps = [{broadcast = "n", input = 1 : i64, opcode = "add"}, {broadcast = "scalar", input = 0 : i64, limit = -1.000000e+00 : f32, opcode = "relu"}], terminal_view = "identity", version = 1 : i64}
// CHECK-NEXT: ^bb0(%[[LHS:[^:]+]]: f32, %[[RHS:[^:]+]]: f32, %[[BIAS_ELT:[^:]+]]: f32, %[[ACC:[^:]+]]: f32):
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.mulf %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[ACC]], %[[PRODUCT]] : f32
// CHECK-NEXT: linalg.yield %[[SUM]] : f32
// CHECK-NEXT: } -> tensor<4x4xf32>
// CHECK-NEXT: %[[RESULT:.*]] = bufferization.to_memref %[[GEMM]] : tensor<4x4xf32> to memref<4x4xf32>
// CHECK-NEXT: return %[[RESULT]] : memref<4x4xf32>
func.func @matmul_bias_relu_to_linalg(
    %c: !atir.tensor<4x4xf32>, %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>, %bias: !atir.tensor<4xf32>)
    -> !atir.tensor<4x4xf32> {
  %matmul_buffer = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %matmul = "atir.MatMul"(%matmul_buffer, %a, %b) <{
    left_transpose = false, output_transpose = false, right_transpose = false,
    withBias = false
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  %add_buffer = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %add = "atir.Add"(%add_buffer, %matmul, %bias) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>, !atir.tensor<4xf32>)
      -> !atir.tensor<4x4xf32>
  %relu = "atir.Relu"(%c, %add) <{relu_limit = -1.0 : f32}> :
      (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32>
  return %relu : !atir.tensor<4x4xf32>
}
