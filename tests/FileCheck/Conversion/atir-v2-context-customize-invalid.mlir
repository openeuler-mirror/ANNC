// RUN: split-file %s %t
// RUN: env ANNC_ENABLE_LOWERING_CONTEXT_ABI=1 not annc-asm %t/mixed.mlir --convert-context-kernel-abi --convert-atir-v2-to-linalg --verify-each 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: env ANNC_ENABLE_LOWERING_CONTEXT_ABI=1 not annc-asm %t/reordered.mlir --convert-context-kernel-abi --convert-atir-v2-to-linalg --verify-each 2>&1 | FileCheck %s --check-prefix=REORDERED

// MIXED: context Customize entry must contain only input unpacking, constants, one CustomizeOp, and return
// REORDERED: context Customize operands must match entry input slots or constants

//--- mixed.mlir
func.func @mixed(%input: tensor<4x5x32xf32>, %weight: tensor<32x16xf32>)
    -> tensor<4x5x16xf32>
    attributes {annc.kernel, fusion.metadata = {abi = "annc_context_v3"}} {
  %computed = arith.addf %input, %input : tensor<4x5x32xf32>
  %result = "atir.Customize"(%computed, %weight) {opType = "TensordotMatMul"}
      : (tensor<4x5x32xf32>, tensor<32x16xf32>) -> tensor<4x5x16xf32>
  return %result : tensor<4x5x16xf32>
}

//--- reordered.mlir
func.func @reordered(%input: tensor<4x5x32xf32>, %weight: tensor<32x16xf32>)
    -> tensor<4x5x16xf32>
    attributes {annc.kernel, fusion.metadata = {abi = "annc_context_v3"}} {
  %result = "atir.Customize"(%weight, %input) {opType = "TensordotMatMul"}
      : (tensor<32x16xf32>, tensor<4x5x32xf32>) -> tensor<4x5x16xf32>
  return %result : tensor<4x5x16xf32>
}
