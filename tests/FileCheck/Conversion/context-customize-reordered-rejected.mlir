// RUN: env ANNC_ENABLE_LOWERING_CONTEXT_ABI=1 not annc-asm %s --convert-context-kernel-abi --convert-atir-to-linalg --verify-each 2>&1 | FileCheck %s
// CHECK: context Customize operands must match entry input slots or constants

module @main attributes {module.state = "atir"} {
  func.func private @fused_tensordot_matmul(
      %input_tensor: tensor<4x5x32xf32>,
      %weight_tensor: tensor<32x16xf32>)
      -> !atir.tensor<4x5x16xf32, name = "output">
      attributes {annc.kernel, fusion.metadata = {abi = "annc_context_v3"}} {
    %input = builtin.unrealized_conversion_cast %input_tensor : tensor<4x5x32xf32> to !atir.tensor<4x5x32xf32, name = "input">
    %weight = builtin.unrealized_conversion_cast %weight_tensor : tensor<32x16xf32> to !atir.tensor<32x16xf32, name = "weight">
    %result = "atir.Customize"(%weight, %input) {
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
    } : (!atir.tensor<32x16xf32, name = "weight">,
         !atir.tensor<4x5x32xf32, name = "input">)
      -> !atir.tensor<4x5x16xf32, name = "output">
    return %result : !atir.tensor<4x5x16xf32, name = "output">
  }
}
