// RUN: annc-asm %s -aarch64-extract-gemm-problem | FileCheck %s --check-prefix=PROBLEM
// RUN: annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-test.json" -aarch64-gemm-thread-planning | FileCheck %s --check-prefix=CANDIDATE
// RUN: annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-test.json" -aarch64-gemm-thread-planning -aarch64-finalize-gemm-plan | FileCheck %s --check-prefix=PLAN
// RUN: annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-hip09-neon-test.json" -aarch64-gemm-thread-planning | FileCheck %s --check-prefix=HIP09_CANDIDATE
// RUN: annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-hip09-neon-test.json" -aarch64-gemm-thread-planning -aarch64-finalize-gemm-plan | FileCheck %s --check-prefix=HIP09_PLAN
// RUN: annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-test.json" -aarch64-gemm-thread-planning | FileCheck %s --check-prefix=LARGE_CANDIDATE
// RUN: annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy="config-path=%S/Inputs/gemm-tuning-test.json" -aarch64-gemm-thread-planning -aarch64-finalize-gemm-plan | FileCheck %s --check-prefix=LARGE_PLAN

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
// CANDIDATE-SAME: annc.aarch64.gemm_cost_models = {c0 = {
// CANDIDATE-SAME: annc.aarch64.gemm_strategies = {s0 = {
// CANDIDATE-SAME: cache_strategy = "fixed"
// CANDIDATE-SAME: data_type = "f32"
// CANDIDATE-SAME: isa = "sve"
// CANDIDATE-SAME: kernel_family = "annc-sve-f32-v1"
// CANDIDATE-SAME: kernel_strategy = "fixed"
// CANDIDATE-SAME: rhs_packing = "direct"
// CANDIDATE-SAME: target_arch = "hip12"
// CANDIDATE: linalg.matmul {
// CANDIDATE-SAME: annc.aarch64.gemm_candidate = {cost_model = "c0"
// CANDIDATE-SAME: planning_state = "planned"
// CANDIDATE-SAME: strategy = "s0"
// CANDIDATE-SAME: thread_count = 1 : i64
// CANDIDATE-SAME: annc.aarch64.gemm_problem = {

// HIP09_CANDIDATE-LABEL: func.func @external_destination(
// HIP09_CANDIDATE-SAME: annc.aarch64.gemm_strategies = {s0 = {
// HIP09_CANDIDATE-SAME: cache_strategy = "fixed"
// HIP09_CANDIDATE-SAME: data_type = "f32"
// HIP09_CANDIDATE-SAME: isa = "neon"
// HIP09_CANDIDATE-SAME: kernel_family = "annc-neon-f32-v1"
// HIP09_CANDIDATE-SAME: kernel_strategy = "fixed"
// HIP09_CANDIDATE-SAME: rhs_packing = "direct"
// HIP09_CANDIDATE-SAME: target_arch = "hip09"
// HIP09_CANDIDATE: linalg.matmul {
// HIP09_CANDIDATE-SAME: annc.aarch64.gemm_candidate = {cost_model = "c0"
// HIP09_CANDIDATE-SAME: strategy = "s0"

// HIP09_PLAN-LABEL: func.func @external_destination(
// HIP09_PLAN-SAME: annc.aarch64.gemm_plans = {p0 = {
// HIP09_PLAN-SAME: first_kc_mode = "overwrite"
// HIP09_PLAN-SAME: rhs_pack_source = "none"
// HIP09_PLAN-SAME: vector_length_bytes = 16 : i64
// HIP09_PLAN-SAME: annc.aarch64.gemm_strategies = {s0 = {
// HIP09_PLAN-SAME: cache_strategy = "fixed"
// HIP09_PLAN-SAME: data_type = "f32"
// HIP09_PLAN-SAME: execution_kind = "gemm"
// HIP09_PLAN-SAME: isa = "neon"
// HIP09_PLAN-SAME: kernel_family = "annc-neon-f32-v1"
// HIP09_PLAN-SAME: kernel_strategy = "fixed"
// HIP09_PLAN-SAME: rhs_packing = "direct"
// HIP09_PLAN-SAME: target_arch = "hip09"
// HIP09_PLAN: linalg.matmul {
// HIP09_PLAN-SAME: annc.aarch64.gemm_plan = "p0"

// PLAN-LABEL: func.func @external_destination(
// PLAN-SAME: annc.aarch64.gemm_plans = {p0 = {
// PLAN-NOT: annc.aarch64.gemm_candidate
// PLAN-NOT: annc.aarch64.gemm_problem
// PLAN-SAME: first_kc_mode = "overwrite"
// PLAN-SAME: k = 8 : i64
// PLAN-SAME: m = 12 : i64
// PLAN-SAME: macro_order = "mkn"
// PLAN-SAME: micro_order = "mn"
// PLAN-SAME: n = 20 : i64
// PLAN-SAME: next_kc_mode = "accumulate"
// PLAN-SAME: rhs_pack_source = "none"
// PLAN-SAME: shard_direction = "rows"
// PLAN-SAME: strategy = "s0"
// PLAN-SAME: thread_count = 1 : i64
// PLAN-SAME: thread_partition = "static-2d"
// PLAN-SAME: vector_length_bytes = 32 : i64
// PLAN-SAME: version = 4 : i64
// PLAN-SAME: annc.aarch64.gemm_strategies = {s0 = {
// PLAN-SAME: cache_strategy = "fixed"
// PLAN-SAME: data_type = "f32"
// PLAN-SAME: isa = "sve"
// PLAN-SAME: kc = 8 : i64
// PLAN-SAME: kernel_family = "annc-sve-f32-v1"
// PLAN-SAME: kernel_strategy = "fixed"
// PLAN-SAME: mc = 12 : i64
// PLAN-SAME: mr = 3 : i64
// PLAN-SAME: nc = 16 : i64
// PLAN-SAME: panel_lanes = 2 : i64
// PLAN-SAME: rhs_packing = "direct"
// PLAN-SAME: target_arch = "hip12"
// PLAN: linalg.matmul {
// PLAN-SAME: annc.aarch64.gemm_plan = "p0"

// LARGE_CANDIDATE-LABEL: func.func @packed_destination(
// LARGE_CANDIDATE-SAME: annc.aarch64.gemm_strategies = {s0 = {
// LARGE_CANDIDATE-SAME: execution_kind = "gemm"
// LARGE_CANDIDATE-SAME: rhs_packing = "packed"
// LARGE_CANDIDATE: linalg.matmul {

// LARGE_PLAN-LABEL: func.func @packed_destination(
// LARGE_PLAN-SAME: annc.aarch64.gemm_plans = {p0 = {
// LARGE_PLAN-SAME: pack_b_execution = "full-then-compute"
// LARGE_PLAN-SAME: pack_b_schema = "annc-sve-packed-b-v2"
// LARGE_PLAN-SAME: annc.aarch64.gemm_strategies = {s0 = {
// LARGE_PLAN-SAME: rhs_packing = "packed"
// LARGE_PLAN: linalg.matmul {

func.func @external_destination(
    %c: memref<12x20xf32>,
    %a: memref<12x8xf32>,
    %b: memref<8x20xf32>) {
  linalg.matmul
      ins(%a, %b : memref<12x8xf32>, memref<8x20xf32>)
      outs(%c : memref<12x20xf32>)
  return
}

func.func @packed_destination(
    %c: memref<12x20xf32>,
    %a: memref<12x20xf32>,
    %b: memref<20x20xf32>) {
  linalg.matmul
      ins(%a, %b : memref<12x20xf32>, memref<20x20xf32>)
      outs(%c : memref<12x20xf32>)
  return
}
