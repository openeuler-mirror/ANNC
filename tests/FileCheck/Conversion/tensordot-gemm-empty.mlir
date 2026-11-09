// RUN: echo '{"kernel_name":"empty_tensordot","arguments":[{"index":0,"shape":[0,3,8]},{"index":1,"shape":[8,4]},{"index":2,"shape":[0,3,4]}]}' > %t.json
// RUN: annc-asm %s -atir-specialize-shapes='shape-spec=%t.json' -annc-aarch64-gemm-pipeline | FileCheck %s --implicit-check-not=llvm.call --implicit-check-not=memref.alloc --implicit-check-not=memref.copy --implicit-check-not=atir.Dot --implicit-check-not=linalg.matmul

// Specialization must accept an empty leading dimension. The output is
// already allocated by the host; no GEMM or temporary buffer is needed.
// CHECK-LABEL: func.func @empty_tensordot(
// CHECK-SAME: memref<0x3x8xf32>
// CHECK-SAME: memref<8x4xf32>
// CHECK-SAME: memref<0x3x4xf32>
// CHECK: return
func.func @empty_tensordot(%x: !atir.tensor<?x?x8xf32>,
                          %w: !atir.tensor<8x4xf32>,
                          %y: !atir.tensor<?x?x4xf32>) attributes {
    annc.kernel, fusion.pattern = "tensordot_matmul"} {
  %dot = atir.Dot %y, %x, %w : <?x?x4xf32>, <?x?x8xf32>,
      <8x4xf32> -> <?x?x4xf32>
  return
}
