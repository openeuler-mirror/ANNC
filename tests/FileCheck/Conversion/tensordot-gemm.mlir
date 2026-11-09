// RUN: annc-asm %s -convert-atir-to-linalg -o %t
// RUN: FileCheck %s --input-file=%t --check-prefix=VIEW
// RUN: FileCheck %s --input-file=%t --check-prefix=STORAGE
// RUN: annc-asm %s -annc-aarch64-gemm-pipeline='config-path=%S/../Target/aarch64/Inputs/gemm-tuning-hip09-neon-test.json' | FileCheck %s --check-prefix=GEMM

// Both reshapes are descriptor views of the caller's buffers. GEMM writes
// directly to the rank-3 output, without an allocation or copy.
// VIEW-LABEL: func.func @tensordot(
// VIEW-SAME: %[[X:.*]]: memref<2x3x8xf32>, %[[W:.*]]: memref<8x4xf32>, %[[Y:.*]]: memref<2x3x4xf32>
// VIEW-DAG: %[[FLAT_X:.*]] = memref.collapse_shape %[[X]] {{.*}} : memref<2x3x8xf32> into memref<6x8xf32>
// VIEW-DAG: %[[FLAT_Y:.*]] = memref.collapse_shape %[[Y]] {{.*}} : memref<2x3x4xf32> into memref<6x4xf32>
// VIEW: linalg.matmul {annc.aarch64.rhs_name = "weight"} ins(%[[FLAT_X]], %[[W]] : memref<6x8xf32>, memref<8x4xf32>) outs(%[[FLAT_Y]] : memref<6x4xf32>)
// VIEW: return
// STORAGE-LABEL: func.func @tensordot(
// STORAGE-NOT: memref.alloc
// STORAGE-NOT: memref.copy
// GEMM-LABEL: func.func @tensordot(
// GEMM-NOT: linalg.matmul
// GEMM-NOT: atir.Dot
// GEMM: llvm.call @annc_aarch64_neon_kernel_rm_
func.func @tensordot(%x: !atir.tensor<2x3x8xf32>,
                     %w: !atir.tensor<8x4xf32, name = "weight">,
                     %y: !atir.tensor<2x3x4xf32>) {
  %dot = atir.Dot %y, %x, %w : <2x3x4xf32>, <2x3x8xf32>,
      <8x4xf32, name = "weight"> -> <2x3x4xf32>
  return
}
