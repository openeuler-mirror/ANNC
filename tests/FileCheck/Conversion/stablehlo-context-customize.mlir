// RUN: annc-stablehlo-sink --help >/dev/null && env ANNC_ENABLE_LOWERING_CONTEXT_ABI=1 annc-asm %s --convert-context-kernel-abi --convert-atir-v2-to-linalg --verify-each | FileCheck %s --implicit-check-not=stablehlo.constant --implicit-check-not=atir.Customize --implicit-check-not='!atir.tensor'

func.func @stablehlo_constant(%input: tensor<4x5x32xf32>) -> tensor<4x5x16xf32>
    attributes {annc.kernel, fusion.metadata = {abi = "annc_context_v3"}} {
  %weight = stablehlo.constant dense<3.0> : tensor<32x16xf32>
  %result = "atir.Customize"(%input, %weight) {
    opType = "TensordotMatMul",
    metadata = {
      custom.op_name = "TensordotMatMul", custom.type_vars = ["T"],
      custom.arg_names = ["input", "weight"],
      custom.arg_kinds = ["memref", "memref"],
      custom.arg_ranks = [3, 2], custom.arg_type_vars = ["T", "T"],
      custom.result_names = ["output"], custom.result_ranks = [3],
      custom.result_type_vars = ["T"]
    }
  } : (tensor<4x5x32xf32>, tensor<32x16xf32>) -> tensor<4x5x16xf32>
  return %result : tensor<4x5x16xf32>
}

// CHECK: llvm.mlir.global private constant @stablehlo_constant_context_input_1_data
// CHECK-SAME: dense<3.000000e+00> : tensor<512xf32>
// CHECK-LABEL: func.func @stablehlo_constant(
// CHECK-SAME: %[[CTX:.*]]: !llvm.ptr) -> i32
// CHECK: llvm.mlir.addressof @stablehlo_constant_context_input_1_data
// CHECK: %[[LOCAL:.*]] = llvm.alloca {{.*}} x !llvm.struct<(i32, i32, ptr, i32, ptr, i32, ptr, i32, ptr, i32, ptr)>
// CHECK: llvm.store {{.*}}, %[[LOCAL]]
// CHECK: %[[STATUS:.*]] = call @ANNCKernel_ContextV3_TensordotMatMul_float_auto(%[[LOCAL]])
// CHECK-NEXT: return %[[STATUS]] : i32
