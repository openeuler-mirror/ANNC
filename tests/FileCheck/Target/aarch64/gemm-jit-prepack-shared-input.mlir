// RUN: echo '[1]' > %t.json
// RUN: annc-asm %s -aarch64-resolve-gemm-plan -aarch64-select-gemm-strategy='config-path=%S/Inputs/gemm-tuning-hip09-neon-test.json enable-prepack=true' -aarch64-gemm-thread-planning -aarch64-gemm-prepack-rhs='jit-inputs=%t.json packed-c=%t.c' -aarch64-finalize-gemm-plan | FileCheck %s
// RUN: FileCheck %s --check-prefix=PACK < %t.c

// CHECK-LABEL: func.func @shared_input(
// CHECK: rhs_packing = "packed"
// CHECK-NOT: rhs_data_input
// PACK-NOT: static void pack_0
// PACK: annc_jit_packed_rhs_v1
// PACK-SAME: *count=0
func.func @shared_input(%a: memref<32x73xf32>, %b: memref<73x67xf32>,
                        %c: memref<32x67xf32>) {
  %zero = arith.constant 0 : index
  %x = memref.load %b[%zero, %zero] : memref<73x67xf32>
  memref.store %x, %c[%zero, %zero] : memref<32x67xf32>
  linalg.matmul {annc.aarch64.rhs_name = "weight"}
      ins(%a, %b : memref<32x73xf32>, memref<73x67xf32>)
      outs(%c : memref<32x67xf32>)
  return
}
