// RUN: annc-asm %s -lower-context-kernel-entries -convert-atir-to-linalg | FileCheck %s
// RUN: annc-asm %s -lower-context-kernel-entries -convert-atir-to-linalg | mlir-opt -convert-func-to-llvm -reconcile-unrealized-casts | FileCheck %s --check-prefix=LLVM

// Folded initializers must survive lowering as private constants.
// CHECK-DAG: llvm.mlir.global private constant @folded_context_input_1_data
// CHECK-DAG: llvm.mlir.global private constant @folded_context_input_2_data
// CHECK-LABEL: func.func private @forwarded(
// CHECK-SAME: %[[CTX:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: call @ANNCKernel_ContextV3_BatchedMatMulAdd_float_auto(%[[CTX]])
// CHECK-LABEL: func.func private @folded(
// CHECK-SAME: %[[ORIGINAL:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// The complete C layout (including the independent resource array/count) is
// copied; only field 2 (inputs) is replaced below, preserving resources.
// CHECK: %[[COPY:[a-zA-Z0-9_]+]] = llvm.load %[[ORIGINAL]] : !llvm.ptr -> !llvm.struct<(i32, i32, ptr, i32, ptr, i32, ptr, i32, ptr, i32, ptr)>
// CHECK: llvm.alloca
// CHECK: llvm.mlir.addressof @folded_context_input_1_data
// CHECK: llvm.mlir.addressof @folded_context_input_2_data
// CHECK: %[[UPDATED:[a-zA-Z0-9_]+]] = llvm.insertvalue {{.*}}, %[[COPY]][2]
// CHECK: %[[LOCAL:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: llvm.store %[[UPDATED]], %[[LOCAL]]
// CHECK: call @ANNCKernel_ContextV3_BatchedMatMulAdd_float_auto(%[[LOCAL]])
// CHECK-LABEL: func.func private @masked(
// CHECK-SAME: %[[MASKCTX:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: call @ANNCKernel_ContextV3_MaskedMul1_float_auto(%[[MASKCTX]])
// LLVM-DAG: llvm.func @_mlir_ciface_forwarded({{.*}}!llvm.ptr{{.*}}) -> i32
// LLVM-DAG: llvm.func @_mlir_ciface_folded({{.*}}!llvm.ptr{{.*}}) -> i32
// LLVM-DAG: llvm.func @_mlir_ciface_masked({{.*}}!llvm.ptr{{.*}}) -> i32

module {
  func.func private @forwarded(%x: !atir.tensor<1x2x3xf32>,
      %w: !atir.tensor<1x3x4xf32>, %b: !atir.tensor<1x1x4xf32>)
      -> !atir.tensor<1x2x4xf32> attributes {
        annc.kernel, llvm.emit_c_interface,
        fusion.metadata = {abi = "annc_context_v3"}
      } {
    %y = atir.Customize(%x, %w, %b) {opType = "BatchedMatMulAdd", metadata = {custom.arg_kinds = ["memref", "memref", "memref"], custom.arg_names = ["data", "weight", "bias"], custom.arg_ranks = [3, 3, 3], custom.arg_type_vars = ["T", "T", "T"], custom.op_name = "BatchedMatMulAdd", custom.result_names = ["output"], custom.result_ranks = [3], custom.result_type_vars = ["T"], custom.type_vars = ["T"]}} : (!atir.tensor<1x2x3xf32>, !atir.tensor<1x3x4xf32>, !atir.tensor<1x1x4xf32>) -> !atir.tensor<1x2x4xf32>
    return %y : !atir.tensor<1x2x4xf32>
  }
  func.func private @folded(%x: !atir.tensor<1x2x3xf32>,
      %w: !atir.tensor<1x3x4xf32>, %b: !atir.tensor<1x1x4xf32>)
      -> !atir.tensor<1x2x4xf32> attributes {
        annc.kernel, llvm.emit_c_interface,
        fusion.metadata = {abi = "annc_context_v3"}
      } {
    %fw = atir.constant "private" @folded_w -> <1x3x4xf32, data = dense<2.0> : tensor<1x3x4xf32>>
    %fb = atir.constant "private" @folded_b -> <1x1x4xf32, data = dense<3.0> : tensor<1x1x4xf32>>
    %y = atir.Customize(%x, %fw, %fb) {opType = "BatchedMatMulAdd", metadata = {custom.arg_kinds = ["memref", "memref", "memref"], custom.arg_names = ["data", "weight", "bias"], custom.arg_ranks = [3, 3, 3], custom.arg_type_vars = ["T", "T", "T"], custom.op_name = "BatchedMatMulAdd", custom.result_names = ["output"], custom.result_ranks = [3], custom.result_type_vars = ["T"], custom.type_vars = ["T"]}} : (!atir.tensor<1x2x3xf32>, !atir.tensor<1x3x4xf32, data = dense<2.0> : tensor<1x3x4xf32>>, !atir.tensor<1x1x4xf32, data = dense<3.0> : tensor<1x1x4xf32>>) -> !atir.tensor<1x2x4xf32>
    return %y : !atir.tensor<1x2x4xf32>
  }
  func.func private @masked(%x: !atir.tensor<1x2x3xf32>,
      %ids: !atir.tensor<1x2xi64>) -> !atir.tensor<1x2x3xf32> attributes {
        annc.kernel, llvm.emit_c_interface,
        fusion.metadata = {abi = "annc_context_v3",
          context_i64_attrs = [1, 1, 0, 0, 0, 0, 1065353216, 0]}
      } {
    %y = atir.Customize(%x, %ids) {opType = "MaskedMul1", metadata = {custom.arg_kinds = ["memref", "memref", "i64_attr", "i64_attr", "i64_attr", "i64_attr", "i64_attr", "i64_attr", "i64_attr", "i64_attr"], custom.arg_names = ["arg0", "ids", "dataArgs", "rawMuls", "fedMuls", "argMap", "sumMask", "interiorKind", "divisorBits", "padding"], custom.arg_ranks = [3, 2, -1, -1, -1, -1, -1, -1, -1, -1], custom.arg_type_vars = ["T", "TMask", "", "", "", "", "", "", "", ""], custom.attr_values = [1, 1, 0, 0, 0, 0, 1065353216, 0], custom.op_name = "MaskedMul1", custom.result_names = ["output0"], custom.result_ranks = [3], custom.result_type_vars = ["T"], custom.type_vars = ["T", "TMask"]}} : (!atir.tensor<1x2x3xf32>, !atir.tensor<1x2xi64>) -> !atir.tensor<1x2x3xf32>
    return %y : !atir.tensor<1x2x3xf32>
  }
}
