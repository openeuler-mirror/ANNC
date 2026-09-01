// RUN: annc-asm %s -aarch64-resolve-gemm-plan -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-hip09-neon-test.json" -aarch64-autotune-gemm-plan -aarch64-finalize-gemm-plan | FileCheck %s --check-prefix=PLAN
// RUN: annc-asm %s -aarch64-resolve-gemm-plan -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-hip09-neon-test.json" -aarch64-autotune-gemm-plan -aarch64-finalize-gemm-plan -aarch64-gemm-cache-blocking -aarch64-gemm-kernel-tiling | FileCheck %s --check-prefix=TILING
// RUN: annc-asm %s -aarch64-resolve-gemm-plan -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-hip09-neon-test.json" -aarch64-autotune-gemm-plan -aarch64-finalize-gemm-plan -aarch64-gemm-cache-blocking -aarch64-gemm-kernel-tiling -aarch64-gemm-leaf-materialization -aarch64-gemm-microkernel-lowering -aarch64-gemm-abi-lowering -aarch64-verify-gemm-schedule | FileCheck %s --check-prefix=LOWERED

// PLAN-LABEL: func.func @gemv_ab
// PLAN: linalg.matmul
// PLAN-SAME: execution_kind = "gemv-ab"
// PLAN-SAME: kernel_family = "annc-neon-gemv-ab-f32-v1"
// PLAN-SAME: mr = 4 : i64
// PLAN-SAME: panel_lanes = 1 : i64
// PLAN-SAME: rhs_pack_source = "none"
// PLAN-SAME: rhs_packing = "direct"

// TILING-NOT: memref.alloca
// TILING: linalg.matmul
// TILING-SAME: annc.aarch64.kc_mode = "overwrite"

// LOWERED-NOT: annc_aarch64_neon_packb_f32
// LOWERED-NOT: @__annc_aarch64_gemm_pack_b_leaf
// LOWERED-DAG: llvm.func @annc_aarch64_neon_gemv_ab_mr4_k8_f32
// LOWERED-DAG: llvm.func @annc_aarch64_neon_gemv_ab_mr2_k8_acc_f32
// LOWERED: llvm.call @annc_aarch64_neon_gemv_ab_mr4_k8_f32
// LOWERED: llvm.call @annc_aarch64_neon_gemv_ab_mr2_k8_acc_f32
// LOWERED-NOT: linalg.matmul

func.func @gemv_ab(%c: memref<34x1xf32>, %a: memref<34x800xf32>,
                   %b: memref<800x1xf32>) {
  linalg.matmul ins(%a, %b : memref<34x800xf32>, memref<800x1xf32>)
    outs(%c : memref<34x1xf32>)
  return
}
