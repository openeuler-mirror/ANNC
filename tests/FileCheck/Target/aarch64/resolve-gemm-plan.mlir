// RUN: annc-asm %s -aarch64-resolve-gemm-plan | FileCheck %s --check-prefix=PROBLEM
// RUN: annc-asm %s -aarch64-resolve-gemm-plan -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-test.json" -aarch64-autotune-gemm-plan | FileCheck %s --check-prefix=CANDIDATE
// RUN: annc-asm %s -aarch64-resolve-gemm-plan -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-test.json" -aarch64-autotune-gemm-plan -aarch64-finalize-gemm-plan | FileCheck %s --check-prefix=PLAN

// PROBLEM-LABEL: func.func @external_destination(
// PROBLEM: linalg.matmul {
// PROBLEM-SAME: annc.aarch64.gemm_problem = {
// PROBLEM-SAME: k = 8 : i64
// PROBLEM-SAME: lda = 8 : i64
// PROBLEM-SAME: ldb = 20 : i64
// PROBLEM-SAME: ldc = 20 : i64
// PROBLEM-SAME: m = 12 : i64
// PROBLEM-SAME: n = 20 : i64
// PROBLEM-SAME: version = 1 : i64

// CANDIDATE-LABEL: func.func @external_destination(
// CANDIDATE: linalg.matmul {
// CANDIDATE-SAME: annc.aarch64.gemm_candidate = {
// CANDIDATE-SAME: data_type = "f32"
// CANDIDATE-SAME: isa = "sve"
// CANDIDATE-SAME: kc = 8 : i64
// CANDIDATE-SAME: kernel_family = "annc-sve-f32-v1"
// CANDIDATE-SAME: mc = 12 : i64
// CANDIDATE-SAME: mr = 3 : i64
// CANDIDATE-SAME: nc = 16 : i64
// CANDIDATE-SAME: panel_lanes = 2 : i64
// CANDIDATE-SAME: target_arch = "kp950"
// CANDIDATE-SAME: thread_count = 1 : i64
// CANDIDATE-SAME: thread_partition = "static-2d"
// CANDIDATE-SAME: annc.aarch64.gemm_problem = {

// PLAN-LABEL: func.func @external_destination(
// PLAN: linalg.matmul {
// PLAN-NOT: annc.aarch64.gemm_candidate
// PLAN-NOT: annc.aarch64.gemm_problem
// PLAN-SAME: annc.aarch64.gemm_plan = {
// PLAN-SAME: data_type = "f32"
// PLAN-SAME: first_kc_mode = "overwrite"
// PLAN-SAME: isa = "sve"
// PLAN-SAME: k = 8 : i64
// PLAN-SAME: kc = 8 : i64
// PLAN-SAME: kernel_family = "annc-sve-f32-v1"
// PLAN-SAME: m = 12 : i64
// PLAN-SAME: macro_order = "mkn"
// PLAN-SAME: mc = 12 : i64
// PLAN-SAME: micro_order = "mn"
// PLAN-SAME: mr = 3 : i64
// PLAN-SAME: n = 20 : i64
// PLAN-SAME: nc = 16 : i64
// PLAN-SAME: next_kc_mode = "accumulate"
// PLAN-SAME: pack_b_execution = "full-then-compute"
// PLAN-SAME: pack_b_schema = "annc-sve-packed-b-v2"
// PLAN-SAME: panel_lanes = 2 : i64
// PLAN-SAME: rhs_packing = "packed"
// PLAN-SAME: target_arch = "kp950"
// PLAN-SAME: thread_count = 1 : i64
// PLAN-SAME: thread_partition = "static-2d"
// PLAN-SAME: vector_length_bytes = 32 : i64
// PLAN-SAME: version = 1 : i64
func.func @external_destination(
    %c: memref<12x20xf32>,
    %a: memref<12x8xf32>,
    %b: memref<8x20xf32>) {
  linalg.matmul
      ins(%a, %b : memref<12x8xf32>, memref<8x20xf32>)
      outs(%c : memref<12x20xf32>)
  return
}
