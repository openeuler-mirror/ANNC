// RUN: env ANNC_ENABLE_LOWERING_CONTEXT_ABI=1 annc-asm %s --convert-context-kernel-abi --verify-each | FileCheck %s
// RUN: env -u ANNC_ENABLE_LOWERING_CONTEXT_ABI annc-asm %s --convert-context-kernel-abi --verify-each | FileCheck %s --check-prefix=DISABLED
// DISABLED-NOT: annc_context_get_input_memref
// DISABLED-LABEL: func.func private @entry(
// DISABLED-SAME: %{{.*}}: !atir.tensor<2x3xf32
// DISABLED: "atir.buffer"

// CHECK: func.func private @annc_context_get_input_memref(!llvm.ptr, i32) -> !llvm.ptr
// CHECK-LABEL: func.func private @entry(
// CHECK-SAME: %[[CONTEXT:.*]]: !llvm.ptr) -> !atir.tensor<2x3xf32
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: call @annc_context_get_input_memref(%[[CONTEXT]], %[[ZERO]]) : (!llvm.ptr, i32) -> !llvm.ptr
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<{{.*}}> to memref<2x3xf32>
// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: call @annc_context_get_input_memref(%[[CONTEXT]], %[[ONE]]) : (!llvm.ptr, i32) -> !llvm.ptr
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<{{.*}}> to memref<2x3xf32>
// CHECK: bufferization.to_tensor
// CHECK-NOT: strided<
// CHECK-NOT: llvm.getelementptr
// CHECK: "atir.buffer"
// CHECK: "atir.Add"
// CHECK-NOT: memref.copy
// CHECK: return %{{.*}} : !atir.tensor<2x3xf32
// RUN: env ANNC_ENABLE_LOWERING_CONTEXT_ABI=1 annc-asm %s --convert-context-kernel-abi --convert-atir-to-linalg --verify-each | FileCheck %s --check-prefix=NEW-BUFFER
// NEW-BUFFER-LABEL: func.func private @entry(
// NEW-BUFFER: tensor.empty() : tensor<2x3xf32>
// NEW-BUFFER-NOT: memref.alloca
module {
  func.func private @entry(
      %lhs: !atir.tensor<2x3xf32, name = "lhs" >,
      %rhs: !atir.tensor<2x3xf32, name = "rhs" >)
      -> !atir.tensor<2x3xf32, name = "out">
      attributes {
        annc.kernel,
        fusion.metadata = {
          abi = "annc_context_v3",
          args = [
            {dtype = "f32", rank = 2 : i64, role = "dynamic", shape = [2, 3], tf_name = "lhs"},
            {dtype = "f32", rank = 2 : i64, role = "dynamic", shape = [2, 3], tf_name = "rhs"}
          ],
          kernel_arg_order = [],
          outputs = [
            {dtype = "f32", rank = 2 : i64, role = "output", shape = [2, 3], tf_name = "out"}
          ]
        }
      } {
    %buffer = "atir.buffer"() : () -> !atir.tensor<2x3xf32, name = "out">
    %sum = "atir.Add"(%buffer, %lhs, %rhs) <{
      do_relu = false,
      relu_limit = -1.0 : f32
    }> : (!atir.tensor<2x3xf32, name = "out" >,
          !atir.tensor<2x3xf32, name = "lhs" >,
          !atir.tensor<2x3xf32, name = "rhs" >)
        -> !atir.tensor<2x3xf32, name = "out">
    return %sum : !atir.tensor<2x3xf32, name = "out">
  }
}
