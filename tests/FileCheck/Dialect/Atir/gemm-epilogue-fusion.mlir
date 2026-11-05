// RUN: annc-asm --split-input-file %s -atir-gemm-epilogue-fusion | FileCheck %s --check-prefix=CANDIDATE --implicit-check-not='annc.gemm.epilogue ='
// RUN: annc-asm --split-input-file %s -atir-gemm-epilogue-fusion -aarch64-atir-gemm-epilogue-commit | FileCheck %s --implicit-check-not=annc.gemm.epilogue.candidate

// Case 1: Fuse MatMul, bias Add and Relu.
// Discovery only annotates the candidate; Commit rewrites the chain. Dead
// buffers remain for later cleanup, not for either epilogue pass to erase.
// CANDIDATE-LABEL: func.func @matmul_bias_relu(
// CANDIDATE-NEXT: %[[TMP:.*]] = "atir.buffer"
// CANDIDATE-NEXT: %[[MM:.*]] = "atir.MatMul"(%[[TMP]],
// CANDIDATE-SAME: annc.gemm.epilogue.candidate = {output_role = "main_d", steps = [{broadcast = "n", input = 1 : i64, opcode = "add"}, {broadcast = "scalar", input = 0 : i64, limit = -1.000000e+00 : f32, opcode = "relu"}], terminal_view = "identity", version = 1 : i64}
// CANDIDATE-NEXT: %[[ADD_TMP:.*]] = "atir.buffer"
// CANDIDATE-NEXT: %[[ADD:.*]] = "atir.Add"(%[[ADD_TMP]], %[[MM]],
// CANDIDATE-NEXT: %[[RELU:.*]] = "atir.Relu"({{.*}}, %[[ADD]])
// CANDIDATE-NEXT: return %[[RELU]]
// CHECK-LABEL: func.func @matmul_bias_relu(
// CHECK-SAME: %[[A:.*]]: !atir.tensor<4x8xf32>, %[[B:.*]]: !atir.tensor<8x4xf32>, %[[C:.*]]: !atir.tensor<4x4xf32>, %[[BIAS:.*]]: !atir.tensor<4xf32>)
// CHECK-NEXT: "atir.buffer"
// CHECK-NEXT: "atir.buffer"
// CHECK-NEXT: %[[FUSED:.*]] = "atir.MatMul"(%[[C]], %[[A]], %[[B]], %[[BIAS]])
// CHECK-SAME: annc.gemm.epilogue = {output_role = "main_d", steps = [{broadcast = "n", input = 1 : i64, opcode = "add"}, {broadcast = "scalar", input = 0 : i64, limit = -1.000000e+00 : f32, opcode = "relu"}], terminal_view = "identity", version = 1 : i64}
// CHECK-NEXT: return %[[FUSED]]
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
// Expect: commit one N-axis add step and remove the Add.
// CHECK-LABEL: func.func @matmul_bias(
// CHECK-SAME: %[[A:[^:]+]]: !atir.tensor<4x8xf32>, %[[B:[^:]+]]: !atir.tensor<8x4xf32>, %[[C:[^:]+]]: !atir.tensor<4x4xf32>, %[[BIAS:[^:]+]]: !atir.tensor<4xf32>)
// CHECK-NEXT: "atir.buffer"
// CHECK-NEXT: %[[FUSED:.*]] = "atir.MatMul"(%[[C]], %[[A]], %[[B]], %[[BIAS]])
// CHECK-SAME: annc.gemm.epilogue = {output_role = "main_d", steps = [{broadcast = "n", input = 1 : i64, opcode = "add"}], terminal_view = "identity", version = 1 : i64}
// CHECK-NEXT: return %[[FUSED]]
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
// CHECK-SAME: %[[A:[^:]+]]: !atir.tensor<4x8xf32>, %[[B:[^:]+]]: !atir.tensor<8x4xf32>, %[[C:[^:]+]]: !atir.tensor<4x4xf32>)
// CHECK-NEXT: "atir.buffer"
// CHECK-NEXT: %[[BIAS:.*]] = "atir.buffer"() : () -> !atir.tensor<4xf32>
// CHECK-NEXT: %[[FUSED:.*]] = "atir.MatMul"(%[[C]], %[[A]], %[[B]], %[[BIAS]])
// CHECK-SAME: annc.gemm.epilogue = {output_role = "main_d", steps = [{broadcast = "n", input = 1 : i64, opcode = "add"}], terminal_view = "identity", version = 1 : i64}
// CHECK-NEXT: return %[[FUSED]]
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

// Case 5: Fuse an Add with a rank-2 addend.
// Expect: classify the addend as a matrix, not an N-axis bias.
// CHECK-LABEL: func.func @fuse_matrix_addend(
// CHECK-SAME: %[[A:[^:]+]]: !atir.tensor<4x8xf32>, %[[B:[^:]+]]: !atir.tensor<8x4xf32>, %[[C:[^:]+]]: !atir.tensor<4x4xf32>, %[[ADDEND:[^:]+]]: !atir.tensor<4x4xf32>)
// CHECK-NEXT: "atir.buffer"
// CHECK-NEXT: %[[FUSED:.*]] = "atir.MatMul"(%[[C]], %[[A]], %[[B]], %[[ADDEND]])
// CHECK-SAME: annc.gemm.epilogue = {output_role = "main_d", steps = [{broadcast = "matrix", input = 1 : i64, opcode = "add"}], terminal_view = "identity", version = 1 : i64}
// CHECK-NEXT: return %[[FUSED]]
func.func @fuse_matrix_addend(
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

// Case 8: Preserve the chain when Add has multiple users.
// Expect: reject the candidate rather than changing either user's input.
// CANDIDATE-LABEL: func.func @preserve_shared_add_result(
// CANDIDATE: "atir.MatMul"
// CANDIDATE-NOT: annc.gemm.epilogue.candidate
// CANDIDATE: "atir.Add"
// CANDIDATE: "atir.Relu"
// CANDIDATE: return
// CHECK-LABEL: func.func @preserve_shared_add_result(
// CHECK-SAME: %[[A:[^:]+]]: !atir.tensor<2x3xf32>, %[[B:[^:]+]]: !atir.tensor<3x4xf32>, %[[ADD_OUT:[^:]+]]: !atir.tensor<2x4xf32>, %[[RELU_OUT:[^:]+]]: !atir.tensor<2x4xf32>, %[[BIAS:[^:]+]]: !atir.tensor<4xf32>)
// CHECK-NEXT: %[[BUFFER:.*]] = "atir.buffer"() : () -> !atir.tensor<2x4xf32>
// CHECK-NEXT: %[[MATMUL:.*]] = "atir.MatMul"(%[[BUFFER]], %[[A]], %[[B]])
// CHECK-SAME: <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> :
// CHECK-NEXT: %[[ADD:.*]] = "atir.Add"(%[[ADD_OUT]], %[[MATMUL]], %[[BIAS]]) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> :
// CHECK-NEXT: %[[RELU:.*]] = "atir.Relu"(%[[RELU_OUT]], %[[ADD]]) <{relu_limit = -1.000000e+00 : f32}> :
// CHECK-NEXT: return %[[ADD]], %[[RELU]] : !atir.tensor<2x4xf32>, !atir.tensor<2x4xf32>
// CHECK-NEXT: }
func.func @preserve_shared_add_result(
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

// Case 10: Fuse two consecutive bias additions.
// Expect: preserve both ordered steps and both bias operands.
// CHECK-LABEL: func.func @fuse_two_bias_adds(
// CHECK-SAME: %[[A:[^:]+]]: !atir.tensor<2x3xf32>, %[[B:[^:]+]]: !atir.tensor<3x4xf32>, %[[FIRST_OUT:[^:]+]]: !atir.tensor<2x4xf32>, %[[FINAL_OUT:[^:]+]]: !atir.tensor<2x4xf32>, %[[BIAS1:[^:]+]]: !atir.tensor<4xf32>, %[[BIAS2:[^:]+]]: !atir.tensor<4xf32>)
// CHECK-NEXT: "atir.buffer"
// CHECK-NEXT: %[[FUSED:.*]] = "atir.MatMul"(%[[FINAL_OUT]], %[[A]], %[[B]], %[[BIAS1]], %[[BIAS2]])
// CHECK-SAME: {annc.gemm.epilogue = {output_role = "main_d", steps = [{broadcast = "n", input = 1 : i64, opcode = "add"}, {broadcast = "n", input = 1 : i64, opcode = "add"}], terminal_view = "identity", version = 1 : i64}} :
// CHECK-NEXT: return %[[FUSED]] : !atir.tensor<2x4xf32>
// CHECK-NEXT: }
func.func @fuse_two_bias_adds(
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
// Expect: fuse MatMul, Add and Relu while keeping the shared Add buffer and
// its Shape connection alive. Dead buffer cleanup is a separate stage.
// CHECK-LABEL: func.func @preserve_shared_intermediate_buffer(
// CHECK-SAME: %[[A:[^:]+]]: !atir.tensor<2x3xf32>, %[[B:[^:]+]]: !atir.tensor<3x4xf32>, %[[OUT:[^:]+]]: !atir.tensor<2x4xf32>, %[[BIAS:[^:]+]]: !atir.tensor<4xf32>)
// CHECK-NEXT: "atir.buffer"
// CHECK-NEXT: %[[SHARED_BUFFER:.*]] = "atir.buffer"() : () -> !atir.tensor<2x4xf32>
// CHECK-NEXT: %[[FUSED:.*]] = "atir.MatMul"(%[[OUT]], %[[A]], %[[B]], %[[BIAS]])
// CHECK-SAME: {annc.gemm.epilogue = {output_role = "main_d", steps = [{broadcast = "n", input = 1 : i64, opcode = "add"}, {broadcast = "scalar", input = 0 : i64, limit = -1.000000e+00 : f32, opcode = "relu"}], terminal_view = "identity", version = 1 : i64}} :
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
// Expect: fuse the full chain into the output argument, leaving the old
// shared buffer for later cleanup without a dangling use.
// CHECK-LABEL: func.func @fuse_shared_intermediate_buffer(
// CHECK-SAME: %[[A:[^:]+]]: !atir.tensor<2x3xf32>, %[[B:[^:]+]]: !atir.tensor<3x4xf32>, %[[OUT:[^:]+]]: !atir.tensor<2x4xf32>, %[[BIAS:[^:]+]]: !atir.tensor<4xf32>)
// CHECK-NEXT: "atir.buffer"
// CHECK-NEXT: %[[FUSED:.*]] = "atir.MatMul"(%[[OUT]], %[[A]], %[[B]], %[[BIAS]])
// CHECK-SAME: {annc.gemm.epilogue = {output_role = "main_d", steps = [{broadcast = "n", input = 1 : i64, opcode = "add"}, {broadcast = "scalar", input = 0 : i64, limit = -1.000000e+00 : f32, opcode = "relu"}], terminal_view = "identity", version = 1 : i64}} :
// CHECK-NEXT: return %[[FUSED]] : !atir.tensor<2x4xf32>
// CHECK-NEXT: }
func.func @fuse_shared_intermediate_buffer(
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

// Case 13: Fuse a chain whose final destination is a local buffer.
// Expect: use the Add buffer as the fused MatMul destination.
// CHECK-LABEL: func.func @fuse_local_destination(
// CHECK-SAME: %[[A:[^:]+]]: !atir.tensor<2x3xf32>, %[[B:[^:]+]]: !atir.tensor<3x4xf32>, %[[BIAS:[^:]+]]: !atir.tensor<4xf32>)
// CHECK-NEXT: %[[MATMUL_BUFFER:.*]] = "atir.buffer"() : () -> !atir.tensor<2x4xf32>
// CHECK-NEXT: %[[ADD_BUFFER:.*]] = "atir.buffer"() : () -> !atir.tensor<2x4xf32>
// CHECK-NEXT: %[[FUSED:.*]] = "atir.MatMul"(%[[ADD_BUFFER]], %[[A]], %[[B]], %[[BIAS]])
// CHECK-SAME: {annc.gemm.epilogue = {output_role = "main_d", steps = [{broadcast = "n", input = 1 : i64, opcode = "add"}], terminal_view = "identity", version = 1 : i64}} :
// CHECK-NEXT: return %[[FUSED]] : !atir.tensor<2x4xf32>
// CHECK-NEXT: }
func.func @fuse_local_destination(
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
// destination, lhs, rhs, bias; input=0 records the bias's original position.
// CHECK-LABEL: func.func @fuse_swapped_bias_inputs(
// CHECK-SAME: %[[A:[^:]+]]: !atir.tensor<2x3xf32>, %[[B:[^:]+]]: !atir.tensor<3x4xf32>, %[[OUT:[^:]+]]: !atir.tensor<2x4xf32>, %[[BIAS:[^:]+]]: !atir.tensor<4xf32>)
// CHECK-NEXT: "atir.buffer"
// CHECK-NEXT: %[[FUSED:.*]] = "atir.MatMul"(%[[OUT]], %[[A]], %[[B]], %[[BIAS]])
// CHECK-SAME: {annc.gemm.epilogue = {output_role = "main_d", steps = [{broadcast = "n", input = 0 : i64, opcode = "add"}], terminal_view = "identity", version = 1 : i64}} :
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
