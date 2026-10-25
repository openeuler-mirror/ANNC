// RUN: echo '[1]' > %t.json
// RUN: annc-asm %s -aarch64-resolve-gemm-plan -aarch64-select-gemm-strategy='config-path=%S/Inputs/gemm-tuning-hip09-neon-test.json enable-prepack=true intra-thread-count=8' -aarch64-gemm-thread-planning -aarch64-gemm-prepack-rhs='jit-inputs=%t.json packed-c=%t.c' -aarch64-finalize-gemm-plan -aarch64-gemm-thread-tiling -aarch64-gemm-cache-blocking -aarch64-gemm-kernel-tiling -aarch64-gemm-leaf-materialization | FileCheck %s
// RUN: FileCheck %s --check-prefix=PACK < %t.c

// RUN: echo '[1,1]' > %t.bad.json
// RUN: ! annc-asm %s -aarch64-gemm-prepack-rhs='jit-inputs=%t.bad.json packed-c=%t.bad.c' > %t.error 2>&1
// RUN: FileCheck %s --check-prefix=INVALID < %t.error
// RUN: echo '{"argument":1}' > %t.bad.json
// RUN: ! annc-asm %s -aarch64-gemm-prepack-rhs='jit-inputs=%t.bad.json packed-c=%t.bad.c' > %t.error 2>&1
// RUN: FileCheck %s --check-prefix=ARRAY < %t.error
// INVALID: invalid/duplicate JIT immutable argument
// ARRAY: JIT immutable input manifest must be an array

// CHECK-NOT: memref.global
// CHECK-LABEL: func.func @jit_prepack(
// CHECK: call @annc_threadpool_parallel_for_gemm
// CHECK-LABEL: func.func private @__annc_gemm_thread_task_
// CHECK-NOT: memref.get_global
// CHECK-NOT: call @__annc_aarch64_gemm_pack_b_leaf
// CHECK: call @__annc_aarch64_gemm_microkernel_leaf
// PACK: static void pack_0
// PACK: annc_aarch64_neon_packb_f32
// PACK: {1,1,73,67,8,16,0,4,4964,64}
// PACK: annc_jit_packed_rhs_v1
// PACK-SAME: *count=1
func.func @jit_prepack(%a: memref<32x73xf32>, %b: memref<73x67xf32>,
                       %c: memref<32x67xf32>) {
  linalg.matmul {annc.aarch64.rhs_name = "weight"}
      ins(%a, %b : memref<32x73xf32>, memref<73x67xf32>)
      outs(%c : memref<32x67xf32>)
  return
}
