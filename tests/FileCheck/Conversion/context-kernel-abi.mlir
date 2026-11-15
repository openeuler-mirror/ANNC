// RUN: env ANNC_ENABLE_LOWERING_CONTEXT_ABI=1 annc-asm %s --convert-context-kernel-abi --verify-each | FileCheck %s
// RUN: env -u ANNC_ENABLE_LOWERING_CONTEXT_ABI annc-asm %s --convert-context-kernel-abi --verify-each | FileCheck %s --check-prefix=DISABLED

// DISABLED-NOT: annc_context_get_input_memref
// DISABLED-LABEL: func.func private @entry(
// DISABLED-SAME: %{{.*}}: tensor<2x3xf32>, %{{.*}}: tensor<2x3xf32>) -> tensor<2x3xf32>
// DISABLED: arith.addf
// DISABLED-LABEL: func.func private @dynamic_scalar(
// DISABLED-SAME: tensor<?x3xi64>
// DISABLED-SAME: tensor<i32>
// DISABLED-NOT: annc_context_get_input_memref

// CHECK: func.func private @annc_context_get_input_memref(!llvm.ptr, i32) -> !llvm.ptr
// CHECK-LABEL: func.func private @entry(
// CHECK-SAME: %[[CONTEXT:.*]]: !llvm.ptr) -> tensor<2x3xf32>
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[LHS_PTR:.*]] = call @annc_context_get_input_memref(%[[CONTEXT]], %[[ZERO]]) : (!llvm.ptr, i32) -> !llvm.ptr
// CHECK-NEXT: %[[LHS_DESC:.*]] = llvm.load %[[LHS_PTR]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[LHS_MEMREF:.*]] = builtin.unrealized_conversion_cast %[[LHS_DESC]] : !llvm.struct<{{.*}}> to memref<2x3xf32>
// CHECK-NEXT: %[[LHS:.*]] = bufferization.to_tensor %[[LHS_MEMREF]] restrict writable : memref<2x3xf32> to tensor<2x3xf32>
// CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[RHS_PTR:.*]] = call @annc_context_get_input_memref(%[[CONTEXT]], %[[ONE]]) : (!llvm.ptr, i32) -> !llvm.ptr
// CHECK-NEXT: %[[RHS_DESC:.*]] = llvm.load %[[RHS_PTR]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[RHS_MEMREF:.*]] = builtin.unrealized_conversion_cast %[[RHS_DESC]] : !llvm.struct<{{.*}}> to memref<2x3xf32>
// CHECK-NEXT: %[[RHS:.*]] = bufferization.to_tensor %[[RHS_MEMREF]] restrict writable : memref<2x3xf32> to tensor<2x3xf32>
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[LHS]], %[[RHS]] : tensor<2x3xf32>
// CHECK-NEXT: return %[[SUM]] : tensor<2x3xf32>

// CHECK-LABEL: func.func private @dynamic_scalar(
// CHECK-SAME: %[[CONTEXT2:.*]]: !llvm.ptr) -> (tensor<?x3xi64>, tensor<i32>)
// CHECK: %[[SLOT0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %{{.*}} = call @annc_context_get_input_memref(%[[CONTEXT2]], %[[SLOT0]]) : (!llvm.ptr, i32) -> !llvm.ptr
// CHECK: %[[DYNAMIC_MEMREF:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<{{.*}}> to memref<?x3xi64>
// CHECK-NEXT: %[[DYNAMIC:.*]] = bufferization.to_tensor %[[DYNAMIC_MEMREF]] restrict writable : memref<?x3xi64> to tensor<?x3xi64>
// CHECK-NEXT: %[[SLOT1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %{{.*}} = call @annc_context_get_input_memref(%[[CONTEXT2]], %[[SLOT1]]) : (!llvm.ptr, i32) -> !llvm.ptr
// CHECK: %[[SCALAR_MEMREF:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<{{.*}}> to memref<i32>
// CHECK-NEXT: %[[SCALAR:.*]] = bufferization.to_tensor %[[SCALAR_MEMREF]] restrict writable : memref<i32> to tensor<i32>
// CHECK-NEXT: return %[[DYNAMIC]], %[[SCALAR]] : tensor<?x3xi64>, tensor<i32>

// CHECK-LABEL: func.func @ordinary(
// CHECK-SAME: %[[VALUE:.*]]: i64) -> i64
// CHECK-NEXT: return %[[VALUE]] : i64
// CHECK-LABEL: func.func @ciface(
// CHECK-SAME: %[[BUFFER:.*]]: memref<2x3xf32>) -> memref<2x3xf32>
// CHECK-NEXT: return %[[BUFFER]] : memref<2x3xf32>

module {
  func.func private @entry(%lhs: tensor<2x3xf32>, %rhs: tensor<2x3xf32>)
      -> tensor<2x3xf32> attributes {
        annc.kernel,
        fusion.metadata = {abi = "annc_context_v3", kernel_arg_order = []}
      } {
    %sum = arith.addf %lhs, %rhs : tensor<2x3xf32>
    return %sum : tensor<2x3xf32>
  }

  func.func private @dynamic_scalar(%dynamic: tensor<?x3xi64>, %scalar: tensor<i32>)
      -> (tensor<?x3xi64>, tensor<i32>) attributes {
        annc.kernel, fusion.metadata = {abi = "annc_context_v3"}
      } {
    return %dynamic, %scalar : tensor<?x3xi64>, tensor<i32>
  }

  func.func @ordinary(%value: i64) -> i64 {
    return %value : i64
  }

  func.func @ciface(%buffer: memref<2x3xf32>) -> memref<2x3xf32> attributes {
    annc.kernel, fusion.metadata = {abi = "mlir_ciface"}
  } {
    return %buffer : memref<2x3xf32>
  }
}
