// RUN: annc-asm --split-input-file %s -atir-gemm-epilogue-fusion | FileCheck %s

// Case 1: Fuse MatMul, bias Add and Relu.
// Expect: emit ordered bias_add and relu steps and remove unused buffers.
// CHECK-LABEL: func.func @matmul_bias_relu(
// CHECK-SAME: %[[A:.*]]: !atir.tensor<4x8xf32>, %[[B:.*]]: !atir.tensor<8x4xf32>, %[[C:.*]]: !atir.tensor<4x4xf32>, %[[BIAS:.*]]: !atir.tensor<4xf32>)
// CHECK-NOT: "atir.buffer"
// CHECK-NOT: "atir.Add"
// CHECK-NOT: "atir.Relu"
// CHECK: "atir.MatMul"(%[[C]], %[[A]], %[[B]], %[[BIAS]])
// CHECK-SAME: annc.gemm.epilogue = [{broadcast = "n", input = 0 : i64, kind = "bias_add"}, {kind = "relu", limit = -1.000000e+00 : f32}]
// CHECK: return
func.func @matmul_bias_relu(
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>,
    %c: !atir.tensor<4x4xf32>,
    %bias: !atir.tensor<4xf32>) -> !atir.tensor<4x4xf32> {
  %matmul_buffer = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %matmul = "atir.MatMul"(%matmul_buffer, %a, %b) <{left_transpose = false,
    output_transpose = false,right_transpose = false, withBias = false
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  %add_buffer = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %add = "atir.Add"(%add_buffer, %matmul, %bias) <{
    do_relu = false,
    relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>,
        !atir.tensor<4xf32>) -> !atir.tensor<4x4xf32>
  %relu = "atir.Relu"(%c, %add) <{relu_limit = -1.0 : f32}> :
      (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>) ->
      !atir.tensor<4x4xf32>
  return %relu : !atir.tensor<4x4xf32>
}

// -----

// Case 2: Fuse MatMul and bias Add without Relu.
// Expect: emit only bias_add and remove the Add and unused buffer.
// CHECK-LABEL: func.func @matmul_bias(
// CHECK-NOT: "atir.buffer"
// CHECK-NOT: "atir.Add"
// CHECK: "atir.MatMul"
// CHECK-SAME: annc.gemm.epilogue = [{broadcast = "n", input = 0 : i64, kind = "bias_add"}]
// CHECK: return
func.func @matmul_bias(
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>,
    %c: !atir.tensor<4x4xf32>,
    %bias: !atir.tensor<4xf32>) -> !atir.tensor<4x4xf32> {
  %tmp = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{left_transpose = false,
    output_transpose = false,right_transpose = false, withBias = false
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  %add = "atir.Add"(%c, %matmul, %bias) <{
    do_relu = false,
    relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>,
        !atir.tensor<4xf32>) -> !atir.tensor<4x4xf32>
  return %add : !atir.tensor<4x4xf32>
}

// -----

// Case 3: The bias is defined after MatMul.
// Expect: place the fused MatMul after the bias definition.
// CHECK-LABEL: func.func @late_bias_definition(
// CHECK: "atir.buffer"
// CHECK: "atir.MatMul"
// CHECK-NOT: "atir.Add"
func.func @late_bias_definition(
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>,
    %c: !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32> {
  %tmp = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false,
    output_transpose = false,
    right_transpose = false,
    withBias = false
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  %bias = "atir.buffer"() : () -> !atir.tensor<4xf32>
  %add = "atir.Add"(%c, %matmul, %bias) <{
    do_relu = false,
    relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>,
        !atir.tensor<4xf32>) -> !atir.tensor<4x4xf32>
  return %add : !atir.tensor<4x4xf32>
}

// -----

// Case 4: Preserve a MatMul with a legacy activation.
// Expect: keep MatMul and Add without consuming the existing activation.
// CHECK-LABEL: func.func @reject_legacy_matmul_postop(
// CHECK: "atir.MatMul"
// CHECK: do_relu = true
// CHECK: "atir.Add"
func.func @reject_legacy_matmul_postop(
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>,
    %c: !atir.tensor<4x4xf32>,
    %bias: !atir.tensor<4xf32>) -> !atir.tensor<4x4xf32> {
  %tmp = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    do_relu = true,
    left_transpose = false,
    output_transpose = false,
    relu_limit = -1.0 : f32,
    right_transpose = false,
    withBias = false
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  %add = "atir.Add"(%c, %matmul, %bias) <{
    do_relu = false,
    relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>,
        !atir.tensor<4xf32>) -> !atir.tensor<4x4xf32>
  return %add : !atir.tensor<4x4xf32>
}

// -----

// Case 5: Preserve an Add with a rank-2 addend.
// Expect: keep the matrix addition instead of treating it as an N-axis bias.
// CHECK-LABEL: func.func @reject_non_bias_addend(
// CHECK: "atir.MatMul"
// CHECK: "atir.Add"
func.func @reject_non_bias_addend(
    %a: !atir.tensor<4x8xf32>,
    %b: !atir.tensor<8x4xf32>,
    %c: !atir.tensor<4x4xf32>,
    %addend: !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32> {
  %tmp = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{left_transpose = false,
    output_transpose = false,right_transpose = false, withBias = false
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  %add = "atir.Add"(%c, %matmul, %addend) <{
    do_relu = false,
    relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>,
        !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32>
  return %add : !atir.tensor<4x4xf32>
}

// -----

// Case 6: Preserve a MatMul with output transposition.
// Expect: keep the transposed MatMul and its Add.
// CHECK-LABEL: func.func @reject_transposed_matmul(
// CHECK: "atir.MatMul"
// CHECK: "atir.Add"
func.func @reject_transposed_matmul(
    %a: !atir.tensor<2x3xf32>,
    %b: !atir.tensor<3x4xf32>,
    %c: !atir.tensor<4x2xf32>,
    %bias: !atir.tensor<2xf32>) -> !atir.tensor<4x2xf32> {
  %tmp = "atir.buffer"() : () -> !atir.tensor<4x2xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false,
    output_transpose = true,
    right_transpose = false,
    withBias = false
  }> : (!atir.tensor<4x2xf32>, !atir.tensor<2x3xf32>,
        !atir.tensor<3x4xf32>) -> !atir.tensor<4x2xf32>
  %add = "atir.Add"(%c, %matmul, %bias) <{
    do_relu = false,
    relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x2xf32>, !atir.tensor<4x2xf32>,
        !atir.tensor<2xf32>) -> !atir.tensor<4x2xf32>
  return %add : !atir.tensor<4x2xf32>
}

// -----

// Case 7: Preserve a MatMul result shared by two eligible Adds.
// Expect: keep both Adds connected to the original MatMul and return both
// results without introducing an epilogue. Both destinations are compatible
// function arguments, so the extra MatMul use is the only fusion blocker.
// CHECK-LABEL: func.func @shared_matmul_result(
// CHECK-SAME: %[[A:[^:]+]]: !atir.tensor<3x6xf32>, %[[B:[^:]+]]: !atir.tensor<6x3xf32>, %[[OUT0:[^:]+]]: !atir.tensor<3x3xf32>, %[[OUT1:[^:]+]]: !atir.tensor<3x3xf32>, %[[BIAS:[^:]+]]: !atir.tensor<3xf32>)
// CHECK-NEXT: %[[BUFFER:.*]] = "atir.buffer"() : () -> !atir.tensor<3x3xf32>
// CHECK-NEXT: %[[MATMUL:.*]] = "atir.MatMul"(%[[BUFFER]], %[[A]], %[[B]])
// CHECK-SAME: <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> :
// CHECK-NEXT: %[[ADD0:.*]] = "atir.Add"(%[[OUT0]], %[[MATMUL]], %[[BIAS]]) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> :
// CHECK-NEXT: %[[ADD1:.*]] = "atir.Add"(%[[OUT1]], %[[MATMUL]], %[[BIAS]]) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> :
// CHECK-NEXT: return %[[ADD0]], %[[ADD1]] : !atir.tensor<3x3xf32>, !atir.tensor<3x3xf32>
// CHECK-NEXT: }
func.func @shared_matmul_result(
    %a: !atir.tensor<3x6xf32>,
    %b: !atir.tensor<6x3xf32>,
    %out0: !atir.tensor<3x3xf32>,
    %out1: !atir.tensor<3x3xf32>,
    %bias: !atir.tensor<3xf32>)
    -> (!atir.tensor<3x3xf32>, !atir.tensor<3x3xf32>) {
  %buf = "atir.buffer"() : () -> !atir.tensor<3x3xf32>
  %matmul = "atir.MatMul"(%buf, %a, %b) <{withBias = false}> :
      (!atir.tensor<3x3xf32>, !atir.tensor<3x6xf32>, !atir.tensor<6x3xf32>)
      -> !atir.tensor<3x3xf32>
  %add0 = "atir.Add"(%out0, %matmul, %bias) :
      (!atir.tensor<3x3xf32>, !atir.tensor<3x3xf32>, !atir.tensor<3xf32>)
      -> !atir.tensor<3x3xf32>
  %add1 = "atir.Add"(%out1, %matmul, %bias) :
      (!atir.tensor<3x3xf32>, !atir.tensor<3x3xf32>, !atir.tensor<3xf32>)
      -> !atir.tensor<3x3xf32>
  return %add0, %add1 : !atir.tensor<3x3xf32>, !atir.tensor<3x3xf32>
}

// -----

// Case 8: Fuse only the bias when Add has multiple users.
// Expect: write the fused result to the Add output argument, redirect both
// users, keep Relu separate, and remove the unused MatMul buffer.
// CHECK-LABEL: func.func @fuse_shared_add_result(
// CHECK-SAME: %[[A:[^:]+]]: !atir.tensor<2x3xf32>, %[[B:[^:]+]]: !atir.tensor<3x4xf32>, %[[ADD_OUT:[^:]+]]: !atir.tensor<2x4xf32>, %[[RELU_OUT:[^:]+]]: !atir.tensor<2x4xf32>, %[[BIAS:[^:]+]]: !atir.tensor<4xf32>)
// CHECK-NEXT: %[[FUSED:.*]] = "atir.MatMul"(%[[ADD_OUT]], %[[A]], %[[B]], %[[BIAS]])
// CHECK-SAME: {annc.gemm.epilogue = [{broadcast = "n", input = 0 : i64, kind = "bias_add"}]} :
// CHECK-NEXT: %[[RELU:.*]] = "atir.Relu"(%[[RELU_OUT]], %[[FUSED]]) <{relu_limit = -1.000000e+00 : f32}> :
// CHECK-NEXT: return %[[FUSED]], %[[RELU]] : !atir.tensor<2x4xf32>, !atir.tensor<2x4xf32>
// CHECK-NEXT: }
func.func @fuse_shared_add_result(
    %a: !atir.tensor<2x3xf32>,
    %b: !atir.tensor<3x4xf32>,
    %add_out: !atir.tensor<2x4xf32>,
    %relu_out: !atir.tensor<2x4xf32>,
    %bias: !atir.tensor<4xf32>)
    -> (!atir.tensor<2x4xf32>, !atir.tensor<2x4xf32>) {
  %buffer = "atir.buffer"() : () -> !atir.tensor<2x4xf32>
  %matmul = "atir.MatMul"(%buffer, %a, %b) <{withBias = false}> :
      (!atir.tensor<2x4xf32>, !atir.tensor<2x3xf32>, !atir.tensor<3x4xf32>)
      -> !atir.tensor<2x4xf32>
  %add = "atir.Add"(%add_out, %matmul, %bias) :
      (!atir.tensor<2x4xf32>, !atir.tensor<2x4xf32>, !atir.tensor<4xf32>)
      -> !atir.tensor<2x4xf32>
  %relu = "atir.Relu"(%relu_out, %add) :
      (!atir.tensor<2x4xf32>, !atir.tensor<2x4xf32>) -> !atir.tensor<2x4xf32>
  return %add, %relu : !atir.tensor<2x4xf32>, !atir.tensor<2x4xf32>
}

// -----

// Case 9: Preserve an Add with its own bounded Relu.
// Expect: keep both operations and the original connections; preserve the
// Add activation and its limit instead of dropping them during fusion.
// CHECK-LABEL: func.func @preserve_add_activation(
// CHECK-SAME: %[[A:[^:]+]]: !atir.tensor<2x3xf32>, %[[B:[^:]+]]: !atir.tensor<3x4xf32>, %[[OUT:[^:]+]]: !atir.tensor<2x4xf32>, %[[BIAS:[^:]+]]: !atir.tensor<4xf32>)
// CHECK-NEXT: %[[BUFFER:.*]] = "atir.buffer"() : () -> !atir.tensor<2x4xf32>
// CHECK-NEXT: %[[MATMUL:.*]] = "atir.MatMul"(%[[BUFFER]], %[[A]], %[[B]])
// CHECK-SAME: <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> :
// CHECK-NEXT: %[[ADD:.*]] = "atir.Add"(%[[OUT]], %[[MATMUL]], %[[BIAS]]) <{do_relu = true, relu_limit = 6.000000e+00 : f32}> :
// CHECK-NEXT: return %[[ADD]] : !atir.tensor<2x4xf32>
// CHECK-NEXT: }
func.func @preserve_add_activation(
    %a: !atir.tensor<2x3xf32>,
    %b: !atir.tensor<3x4xf32>,
    %out: !atir.tensor<2x4xf32>,
    %bias: !atir.tensor<4xf32>) -> !atir.tensor<2x4xf32> {
  %buffer = "atir.buffer"() : () -> !atir.tensor<2x4xf32>
  %matmul = "atir.MatMul"(%buffer, %a, %b) <{withBias = false}> :
      (!atir.tensor<2x4xf32>, !atir.tensor<2x3xf32>, !atir.tensor<3x4xf32>)
      -> !atir.tensor<2x4xf32>
  %add = "atir.Add"(%out, %matmul, %bias) <{
    do_relu = true, relu_limit = 6.0 : f32
  }> : (!atir.tensor<2x4xf32>, !atir.tensor<2x4xf32>, !atir.tensor<4xf32>)
      -> !atir.tensor<2x4xf32>
  return %add : !atir.tensor<2x4xf32>
}

// -----

// Case 10: Fuse only the first of two consecutive bias additions.
// Expect: keep the first bias in the epilogue and the second bias in a
// separate Add; greedy rewriting must not overwrite the existing epilogue.
// CHECK-LABEL: func.func @preserve_second_bias_add(
// CHECK-SAME: %[[A:[^:]+]]: !atir.tensor<2x3xf32>, %[[B:[^:]+]]: !atir.tensor<3x4xf32>, %[[FIRST_OUT:[^:]+]]: !atir.tensor<2x4xf32>, %[[FINAL_OUT:[^:]+]]: !atir.tensor<2x4xf32>, %[[BIAS1:[^:]+]]: !atir.tensor<4xf32>, %[[BIAS2:[^:]+]]: !atir.tensor<4xf32>)
// CHECK-NEXT: %[[FUSED:.*]] = "atir.MatMul"(%[[FIRST_OUT]], %[[A]], %[[B]], %[[BIAS1]])
// CHECK-SAME: {annc.gemm.epilogue = [{broadcast = "n", input = 0 : i64, kind = "bias_add"}]} :
// CHECK-NEXT: %[[ADD:.*]] = "atir.Add"(%[[FINAL_OUT]], %[[FUSED]], %[[BIAS2]]) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> :
// CHECK-NEXT: return %[[ADD]] : !atir.tensor<2x4xf32>
// CHECK-NEXT: }
func.func @preserve_second_bias_add(
    %a: !atir.tensor<2x3xf32>,
    %b: !atir.tensor<3x4xf32>,
    %first_out: !atir.tensor<2x4xf32>,
    %final_out: !atir.tensor<2x4xf32>,
    %bias1: !atir.tensor<4xf32>,
    %bias2: !atir.tensor<4xf32>) -> !atir.tensor<2x4xf32> {
  %buffer = "atir.buffer"() : () -> !atir.tensor<2x4xf32>
  %matmul = "atir.MatMul"(%buffer, %a, %b) <{withBias = false}> :
      (!atir.tensor<2x4xf32>, !atir.tensor<2x3xf32>, !atir.tensor<3x4xf32>)
      -> !atir.tensor<2x4xf32>
  %add1 = "atir.Add"(%first_out, %matmul, %bias1) :
      (!atir.tensor<2x4xf32>, !atir.tensor<2x4xf32>, !atir.tensor<4xf32>)
      -> !atir.tensor<2x4xf32>
  %add2 = "atir.Add"(%final_out, %add1, %bias2) :
      (!atir.tensor<2x4xf32>, !atir.tensor<2x4xf32>, !atir.tensor<4xf32>)
      -> !atir.tensor<2x4xf32>
  return %add2 : !atir.tensor<2x4xf32>
}

// -----

// Case 11: Preserve an intermediate buffer with a remaining Shape user.
// Expect: fuse MatMul, Add and Relu; delete only the unused MatMul buffer,
// keeping the Add buffer and the Shape connection alive.
// CHECK-LABEL: func.func @preserve_shared_intermediate_buffer(
// CHECK-SAME: %[[A:[^:]+]]: !atir.tensor<2x3xf32>, %[[B:[^:]+]]: !atir.tensor<3x4xf32>, %[[OUT:[^:]+]]: !atir.tensor<2x4xf32>, %[[BIAS:[^:]+]]: !atir.tensor<4xf32>)
// CHECK-NEXT: %[[SHARED_BUFFER:.*]] = "atir.buffer"() : () -> !atir.tensor<2x4xf32>
// CHECK-NEXT: %[[FUSED:.*]] = "atir.MatMul"(%[[OUT]], %[[A]], %[[B]], %[[BIAS]])
// CHECK-SAME: {annc.gemm.epilogue = [{broadcast = "n", input = 0 : i64, kind = "bias_add"}, {kind = "relu", limit = -1.000000e+00 : f32}]} :
// CHECK-NEXT: %[[SHAPE:.*]] = atir.Shape %[[SHARED_BUFFER]] : (!atir.tensor<2x4xf32>) -> !atir.tensor<2xi32>
// CHECK-NEXT: return %[[FUSED]], %[[SHAPE]] : !atir.tensor<2x4xf32>, !atir.tensor<2xi32>
// CHECK-NEXT: }
func.func @preserve_shared_intermediate_buffer(
    %a: !atir.tensor<2x3xf32>,
    %b: !atir.tensor<3x4xf32>,
    %out: !atir.tensor<2x4xf32>,
    %bias: !atir.tensor<4xf32>)
    -> (!atir.tensor<2x4xf32>, !atir.tensor<2xi32>) {
  %matmul_buffer = "atir.buffer"() : () -> !atir.tensor<2x4xf32>
  %matmul = "atir.MatMul"(%matmul_buffer, %a, %b) <{withBias = false}> :
      (!atir.tensor<2x4xf32>, !atir.tensor<2x3xf32>, !atir.tensor<3x4xf32>)
      -> !atir.tensor<2x4xf32>
  %shared_buffer = "atir.buffer"() : () -> !atir.tensor<2x4xf32>
  %add = "atir.Add"(%shared_buffer, %matmul, %bias) :
      (!atir.tensor<2x4xf32>, !atir.tensor<2x4xf32>, !atir.tensor<4xf32>)
      -> !atir.tensor<2x4xf32>
  %relu = "atir.Relu"(%out, %add) :
      (!atir.tensor<2x4xf32>, !atir.tensor<2x4xf32>) -> !atir.tensor<2x4xf32>
  %shape = atir.Shape %shared_buffer : (!atir.tensor<2x4xf32>) -> !atir.tensor<2xi32>
  return %relu, %shape : !atir.tensor<2x4xf32>, !atir.tensor<2xi32>
}

// -----

// Case 12: MatMul and Add share the same intermediate buffer.
// Expect: fuse the full chain into the output argument and remove the shared
// buffer once, without a double erase or a dangling use.
// CHECK-LABEL: func.func @erase_shared_intermediate_buffer(
// CHECK-SAME: %[[A:[^:]+]]: !atir.tensor<2x3xf32>, %[[B:[^:]+]]: !atir.tensor<3x4xf32>, %[[OUT:[^:]+]]: !atir.tensor<2x4xf32>, %[[BIAS:[^:]+]]: !atir.tensor<4xf32>)
// CHECK-NEXT: %[[FUSED:.*]] = "atir.MatMul"(%[[OUT]], %[[A]], %[[B]], %[[BIAS]])
// CHECK-SAME: {annc.gemm.epilogue = [{broadcast = "n", input = 0 : i64, kind = "bias_add"}, {kind = "relu", limit = -1.000000e+00 : f32}]} :
// CHECK-NEXT: return %[[FUSED]] : !atir.tensor<2x4xf32>
// CHECK-NEXT: }
func.func @erase_shared_intermediate_buffer(
    %a: !atir.tensor<2x3xf32>,
    %b: !atir.tensor<3x4xf32>,
    %out: !atir.tensor<2x4xf32>,
    %bias: !atir.tensor<4xf32>) -> !atir.tensor<2x4xf32> {
  %shared_buffer = "atir.buffer"() : () -> !atir.tensor<2x4xf32>
  %matmul = "atir.MatMul"(%shared_buffer, %a, %b) <{withBias = false}> :
      (!atir.tensor<2x4xf32>, !atir.tensor<2x3xf32>, !atir.tensor<3x4xf32>)
      -> !atir.tensor<2x4xf32>
  %add = "atir.Add"(%shared_buffer, %matmul, %bias) :
      (!atir.tensor<2x4xf32>, !atir.tensor<2x4xf32>, !atir.tensor<4xf32>)
      -> !atir.tensor<2x4xf32>
  %relu = "atir.Relu"(%out, %add) :
      (!atir.tensor<2x4xf32>, !atir.tensor<2x4xf32>) -> !atir.tensor<2x4xf32>
  return %relu : !atir.tensor<2x4xf32>
}

// -----

// Case 13: Preserve a chain whose final destination is a local buffer.
// Expect: keep MatMul, Add and both buffers; fusion requires the final
// destination to be a compatible function argument.
// CHECK-LABEL: func.func @preserve_local_destination(
// CHECK-SAME: %[[A:[^:]+]]: !atir.tensor<2x3xf32>, %[[B:[^:]+]]: !atir.tensor<3x4xf32>, %[[BIAS:[^:]+]]: !atir.tensor<4xf32>)
// CHECK-NEXT: %[[MATMUL_BUFFER:.*]] = "atir.buffer"() : () -> !atir.tensor<2x4xf32>
// CHECK-NEXT: %[[MATMUL:.*]] = "atir.MatMul"(%[[MATMUL_BUFFER]], %[[A]], %[[B]])
// CHECK-SAME: <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> :
// CHECK-NEXT: %[[ADD_BUFFER:.*]] = "atir.buffer"() : () -> !atir.tensor<2x4xf32>
// CHECK-NEXT: %[[ADD:.*]] = "atir.Add"(%[[ADD_BUFFER]], %[[MATMUL]], %[[BIAS]]) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> :
// CHECK-NEXT: return %[[ADD]] : !atir.tensor<2x4xf32>
// CHECK-NEXT: }
func.func @preserve_local_destination(
    %a: !atir.tensor<2x3xf32>,
    %b: !atir.tensor<3x4xf32>,
    %bias: !atir.tensor<4xf32>) -> !atir.tensor<2x4xf32> {
  %matmul_buffer = "atir.buffer"() : () -> !atir.tensor<2x4xf32>
  %matmul = "atir.MatMul"(%matmul_buffer, %a, %b) <{withBias = false}> :
      (!atir.tensor<2x4xf32>, !atir.tensor<2x3xf32>, !atir.tensor<3x4xf32>)
      -> !atir.tensor<2x4xf32>
  %add_buffer = "atir.buffer"() : () -> !atir.tensor<2x4xf32>
  %add = "atir.Add"(%add_buffer, %matmul, %bias) :
      (!atir.tensor<2x4xf32>, !atir.tensor<2x4xf32>, !atir.tensor<4xf32>)
      -> !atir.tensor<2x4xf32>
  return %add : !atir.tensor<2x4xf32>
}

// -----

// Case 14: Fuse an Add with bias before the MatMul result.
// Expect: recognize the swapped inputs and keep the fused operand order as
// destination, lhs, rhs, bias; remove the Add and its unused input buffer.
// CHECK-LABEL: func.func @fuse_swapped_bias_inputs(
// CHECK-SAME: %[[A:[^:]+]]: !atir.tensor<2x3xf32>, %[[B:[^:]+]]: !atir.tensor<3x4xf32>, %[[OUT:[^:]+]]: !atir.tensor<2x4xf32>, %[[BIAS:[^:]+]]: !atir.tensor<4xf32>)
// CHECK-NEXT: %[[FUSED:.*]] = "atir.MatMul"(%[[OUT]], %[[A]], %[[B]], %[[BIAS]])
// CHECK-SAME: {annc.gemm.epilogue = [{broadcast = "n", input = 0 : i64, kind = "bias_add"}]} :
// CHECK-NEXT: return %[[FUSED]] : !atir.tensor<2x4xf32>
// CHECK-NEXT: }
func.func @fuse_swapped_bias_inputs(
    %a: !atir.tensor<2x3xf32>,
    %b: !atir.tensor<3x4xf32>,
    %out: !atir.tensor<2x4xf32>,
    %bias: !atir.tensor<4xf32>) -> !atir.tensor<2x4xf32> {
  %buffer = "atir.buffer"() : () -> !atir.tensor<2x4xf32>
  %matmul = "atir.MatMul"(%buffer, %a, %b) <{withBias = false}> :
      (!atir.tensor<2x4xf32>, !atir.tensor<2x3xf32>, !atir.tensor<3x4xf32>)
      -> !atir.tensor<2x4xf32>
  %add = "atir.Add"(%out, %bias, %matmul) :
      (!atir.tensor<2x4xf32>, !atir.tensor<4xf32>, !atir.tensor<2x4xf32>)
      -> !atir.tensor<2x4xf32>
  return %add : !atir.tensor<2x4xf32>
}
