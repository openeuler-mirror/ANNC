// RUN: annc-asm %s -atir-fast-codegen -lower-context-kernel-entries | FileCheck %s
// RUN: annc-asm %s -atir-fast-codegen | FileCheck %s --check-prefix=SCHEMA
// RUN: env ANNC_GEMM=2 annc-asm %s -atir-fast-codegen | FileCheck %s --check-prefix=SCHEMA
// RUN: annc-asm %s -atir-fast-codegen -lower-context-kernel-entries -convert-atir-to-linalg | FileCheck %s
// RUN: annc-asm %s -atir-fast-codegen -lower-context-kernel-entries -convert-atir-to-linalg | mlir-opt -convert-func-to-llvm -reconcile-unrealized-casts | FileCheck %s --check-prefix=LLVM

// SCHEMA: atir.Customize
// SCHEMA-SAME: custom.arg_names = ["lhs", "rhs"]
// SCHEMA-SAME: custom.result_names = ["output"]
// CHECK: func.func private @ANNCKernel_ContextV3_MatMul_float_auto(!llvm.ptr) -> i32 attributes {llvm.emit_c_interface}
// CHECK-LABEL: func.func private @context_matmul(
// CHECK-SAME: %[[CTX:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK-NOT: memref
// CHECK: %[[STATUS:[a-zA-Z0-9_]+]] = call @ANNCKernel_ContextV3_MatMul_float_auto(%[[CTX]]) : (!llvm.ptr) -> i32
// CHECK: return %[[STATUS]] : i32
// LLVM: llvm.func @_mlir_ciface_ANNCKernel_ContextV3_MatMul_float_auto(!llvm.ptr) -> i32
// LLVM-LABEL: llvm.func @_mlir_ciface_context_matmul(
// LLVM-SAME: %{{[a-zA-Z0-9_]+}}: !llvm.ptr) -> i32

module {
  func.func private @context_matmul(
      %lhs: !atir.tensor<3x8xf32>, %rhs: !atir.tensor<8x4xf32>)
      -> !atir.tensor<3x4xf32> attributes {
        annc.kernel, llvm.emit_c_interface,
        fusion.metadata = {abi = "annc_context_v3"}
      } {
    %buffer = "atir.buffer"() : () -> !atir.tensor<3x4xf32>
    %result = "atir.MatMul"(%buffer, %lhs, %rhs) <{
      do_relu = false, left_transpose = false, output_transpose = false,
      relu_limit = -1.0 : f32, right_transpose = false, withBias = false
    }> : (!atir.tensor<3x4xf32>, !atir.tensor<3x8xf32>,
          !atir.tensor<8x4xf32>) -> !atir.tensor<3x4xf32>
    return %result : !atir.tensor<3x4xf32>
  }
}
