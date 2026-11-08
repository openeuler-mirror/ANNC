// RUN: annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-hip09-neon-test.json" -aarch64-gemm-thread-planning | FileCheck %s --check-prefix=PLAN
// RUN: annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-hip09-neon-test.json" -aarch64-gemm-thread-planning -aarch64-gemm-cache-blocking -aarch64-gemm-kernel-tiling | FileCheck %s --check-prefix=TILING
// RUN: annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-hip09-neon-test.json" -aarch64-gemm-thread-planning -aarch64-gemm-cache-blocking -aarch64-gemm-kernel-tiling -aarch64-gemm-leaf-materialization -aarch64-gemm-microkernel-lowering -aarch64-gemm-abi-lowering -aarch64-verify-gemm-schedule | FileCheck %s --check-prefix=LOWERED

// PLAN-LABEL: func.func @vecmat
// PLAN-SAME: annc.aarch64.gemm_strategies = {s0 = {
// PLAN-SAME: execution_kind = "vector-matrix"
// PLAN-SAME: kernel_family = "annc-neon-vecmat-f32-v1"
// PLAN-SAME: rhs_packing = "direct"
// PLAN: linalg.matmul
// PLAN-SAME: annc.aarch64.gemm_candidate = {

// PLAN-LABEL: func.func @matvec_priority
// PLAN-SAME: annc.aarch64.gemm_strategies = {s0 = {
// PLAN-SAME: execution_kind = "matrix-vector"
// PLAN-SAME: kernel_family = "annc-neon-matvec-f32-v1"
// PLAN: linalg.matmul
// PLAN-SAME: annc.aarch64.gemm_candidate = {

// TILING-NOT: memref.alloca
// TILING-LABEL: func.func @vecmat
// TILING: linalg.matmul
// TILING-SAME: annc.aarch64.kc_mode = "overwrite"

// LOWERED-NOT: annc_aarch64_neon_packb_f32
// LOWERED-NOT: @__annc_aarch64_gemm_pack_b_leaf
// LOWERED-DAG: llvm.func @annc_aarch64_neon_vecmat_n16_kg_r0_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32)
// LOWERED-DAG: llvm.func @annc_aarch64_neon_vecmat_n16_kg_r0_acc_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32)
// LOWERED: llvm.call @annc_aarch64_neon_vecmat_n16_kg_r0_f32
// LOWERED: llvm.call @annc_aarch64_neon_vecmat_n16_kg_r0_acc_f32
// LOWERED-NOT: linalg.matmul

func.func @vecmat(%c: memref<1x400xf32>, %a: memref<1x400xf32>,
                   %b: memref<400x400xf32>) {
  linalg.matmul ins(%a, %b : memref<1x400xf32>, memref<400x400xf32>)
    outs(%c : memref<1x400xf32>)
  return
}

// Large M*N*K problem with N==1 and M==1: the matrix-vector family wins over
// the vector-matrix family once the problem exceeds the small-shape limit.
func.func @matvec_priority(%c: memref<1x1xf32>, %a: memref<1x5000xf32>,
                            %b: memref<5000x1xf32>) {
  linalg.matmul ins(%a, %b : memref<1x5000xf32>, memref<5000x1xf32>)
    outs(%c : memref<1x1xf32>)
  return
}
