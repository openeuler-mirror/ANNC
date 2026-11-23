// RUN: env ANNC_ENABLE_LOWERING_CONTEXT_ABI=1 not annc-asm %s --convert-context-kernel-abi --convert-atir-to-linalg --verify-each 2>&1 | FileCheck %s
// CHECK: context Customize entry must contain only input unpacking, constants, one CustomizeOp, and return
module @main attributes {module.state = "atir"} {
  func.func private @fused_tensordot_matmul(
      %input_tensor: tensor<4x5x32xf32>,
      %weight_tensor: tensor<32x16xf32>)
      -> !atir.tensor<4x5x16xf32, name = "output">
      attributes {
        annc.kernel,
        fusion.metadata = {
          abi = "annc_context_v3",
          args = [
            {dtype = "f32", rank = 3 : i64, role = "dynamic", shape = [4, 5, 32], tf_name = "input"},
            {dtype = "f32", rank = 2 : i64, role = "fixed", shape = [32, 16], tf_name = "weight"}
          ],
          dynamic_dims = [],
          fusion.pattern = "tensordot_matmul",
          kernel_arg_order = [],
          kernel_name = "fused_tensordot_matmul",
          outputs = [{dtype = "f32", rank = 3 : i64, role = "output", shape = [4, 5, 16], tf_name = "output"}]
        }
      } {
    %input = builtin.unrealized_conversion_cast %input_tensor : tensor<4x5x32xf32> to !atir.tensor<4x5x32xf32, name = "input">
    %weight = builtin.unrealized_conversion_cast %weight_tensor : tensor<32x16xf32> to !atir.tensor<32x16xf32, name = "weight">
    %tmp = "atir.buffer"() : () -> !atir.tensor<4x5x32xf32, name = "input">
    %computed = "atir.Add"(%tmp, %input, %input) <{do_relu = false, relu_limit = -1.0 : f32}> : (!atir.tensor<4x5x32xf32, name = "input">, !atir.tensor<4x5x32xf32, name = "input">, !atir.tensor<4x5x32xf32, name = "input">) -> !atir.tensor<4x5x32xf32, name = "input">
    %result = "atir.Customize"(%computed, %weight) {
      opType = "TensordotMatMul",
      metadata = {
        custom.op_name = "TensordotMatMul",
        custom.type_vars = ["T"],
        custom.arg_names = ["input", "weight"],
        custom.arg_kinds = ["memref", "memref"],
        custom.arg_ranks = [3, 2],
        custom.arg_type_vars = ["T", "T"],
        custom.result_names = ["output"],
        custom.result_ranks = [3],
        custom.result_type_vars = ["T"]
      }
    } : (!atir.tensor<4x5x32xf32, name = "input">,
         !atir.tensor<32x16xf32, name = "weight">)
      -> !atir.tensor<4x5x16xf32, name = "output">
    return %result : !atir.tensor<4x5x16xf32, name = "output">
  }
}
