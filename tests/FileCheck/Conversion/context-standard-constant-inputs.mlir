// RUN: annc-asm %s --lower-context-kernel-entries --convert-atir-to-linalg | FileCheck %s
// Prepared constants override only the kernel's view of the original context.
// The packed-B accessor must expose the same prepared weight global.
// CHECK-DAG: llvm.mlir.global private constant @[[W:.*_context_input_1_data]]
// CHECK-DAG: llvm.mlir.global private constant @[[B:.*_context_input_2_data]]
// CHECK-LABEL: func.func @prepared
// CHECK-SAME: annc.effective_weight_global = "[[W]]"
// CHECK: llvm.mlir.addressof @[[W]]
// CHECK: llvm.mlir.addressof @[[B]]
// CHECK: call @ANNCKernel_ContextV3_BatchedMatMulAdd_float_auto
// CHECK-LABEL: func.func @__annc_effective_weight_prepared
// CHECK: llvm.mlir.addressof @[[W]]
// CHECK: llvm.ptrtoint
// CHECK-LABEL: func.func @__annc_effective_weight_size_prepared
// CHECK: arith.constant 12 : i64
#schema = {custom.arg_kinds = ["memref", "memref", "memref"], custom.arg_names = ["data", "weight", "bias"], custom.arg_ranks = [3, 3, 3], custom.arg_type_vars = ["T", "T", "T"], custom.op_name = "BatchedMatMulAdd", custom.result_names = ["output"], custom.result_ranks = [3], custom.result_type_vars = ["T"], custom.type_vars = ["T"]}
module {
  func.func @prepared(%x: tensor<1x?x3xf32>, %w: tensor<1x3x4xf32>, %b: tensor<1x1x4xf32>) -> tensor<1x?x4xf32>
      attributes {annc.kernel, fusion.pattern = "batched_matmul_add", fusion.metadata = {
        abi = "annc_context_v3", context_i64_attrs = [],
        args = [{dtype = "f32", rank = 3 : i64, role = "dynamic", shape = [1, -1, 3], tf_name = "x"}, {dtype = "f32", rank = 3 : i64, role = "fixed", shape = [1, 3, 4], tf_name = "w"}, {dtype = "f32", rank = 3 : i64, role = "fixed", shape = [1, 1, 4], tf_name = "b"}],
        outputs = [{dtype = "f32", rank = 3 : i64, role = "output", shape = [1, -1, 4], tf_name = "y"}]}} {
    %prepared_w = arith.constant {annc.effective_weight} dense<0.125> : tensor<1x3x4xf32>
    %prepared_b = arith.constant dense<2.0> : tensor<1x1x4xf32>
    %y = call @builtin(%x, %prepared_w, %prepared_b) {annc.builtin = "BatchedMatMulAdd", metadata = #schema} : (tensor<1x?x3xf32>, tensor<1x3x4xf32>, tensor<1x1x4xf32>) -> tensor<1x?x4xf32>
    return %y : tensor<1x?x4xf32>
  }
  func.func private @builtin(tensor<1x?x3xf32>, tensor<1x3x4xf32>, tensor<1x1x4xf32>) -> tensor<1x?x4xf32> attributes {annc.builtin = "BatchedMatMulAdd", metadata = #schema}
}
