// RUN: annc-asm %s -aarch64-gemm-leaf-materialization -aarch64-gemm-microkernel-lowering -aarch64-gemm-abi-lowering -aarch64-verify-gemm-schedule | FileCheck %s

// CHECK-DAG: llvm.func @annc_aarch64_sve_kernel_mr3_n2vl_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32)
// CHECK-DAG: llvm.func @annc_aarch64_sve_kernel_mr2_n2vl_acc_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32)
// CHECK: llvm.func @annc_aarch64_sve_packb_f32(!llvm.ptr, !llvm.ptr, i32, i32, i32)
// CHECK-LABEL: func.func @sve_full_tile(
// CHECK: memref.alloca(%{{.*}}) {alignment = 64 : i64} : memref<?xf32>
// CHECK: llvm.call @annc_aarch64_sve_packb_f32
// CHECK: llvm.call @annc_aarch64_sve_kernel_mr3_n2vl_f32
// CHECK-NOT: @__annc_aarch64_gemm_microkernel_leaf
// CHECK-LABEL: func.func @sve_tail_accumulate(
// CHECK: llvm.call @annc_aarch64_sve_packb_f32
// CHECK: llvm.call @annc_aarch64_sve_kernel_mr2_n2vl_acc_f32
// CHECK-NOT: @__annc_aarch64_gemm_microkernel_leaf
// CHECK: return
func.func @sve_full_tile(%c: memref<3x16xf32>, %a: memref<3x8xf32>,
                         %b: memref<8x16xf32>) attributes {annc.aarch64.gemm_strategies = {s0 = {version = 4 : i64, target_arch = "hip12", isa = "sve", data_type = "f32", kernel_family = "annc-sve-f32-v1", cache_strategy = "fixed", kernel_strategy = "fixed", cache_parameters = {mc = 3 : i64, nc = 16 : i64, kc = 8 : i64}, kernel_parameters = {mr = 3 : i64, panel_lanes = 2 : i64}, execution_kind = "gemm", rhs_packing = "packed"}}} {
  linalg.matmul {
    annc.aarch64.gemm_stage = "kernel_tiled",
    annc.aarch64.kc_mode = "overwrite",
    annc.aarch64.gemm_problem = {version = 1 : i64, m = 3 : i64, n = 16 : i64, k = 8 : i64, lda = 8 : i64, ldb = 16 : i64, ldc = 16 : i64}, annc.aarch64.gemm_strategy = "s0", annc.aarch64.gemm_candidate = {version = 4 : i64, planning_state = "planned", strategy = "s0", max_thread_count = 1 : i64, thread_count = 1 : i64, tasks_m = 1 : i64, tasks_n = 1 : i64, shard_direction = "rows", thread_partition = "static-2d"}
  } ins(%a, %b : memref<3x8xf32>, memref<8x16xf32>)
    outs(%c : memref<3x16xf32>)
  return
}

func.func @sve_tail_accumulate(%c: memref<2x11xf32>, %a: memref<2x5xf32>,
                               %b: memref<5x11xf32>) attributes {annc.aarch64.gemm_strategies = {s0 = {version = 4 : i64, target_arch = "hip12", isa = "sve", data_type = "f32", kernel_family = "annc-sve-f32-v1", cache_strategy = "fixed", kernel_strategy = "fixed", cache_parameters = {mc = 3 : i64, nc = 16 : i64, kc = 8 : i64}, kernel_parameters = {mr = 3 : i64, panel_lanes = 2 : i64}, execution_kind = "gemm", rhs_packing = "packed"}}} {
  linalg.matmul {
    annc.aarch64.gemm_stage = "kernel_tiled",
    annc.aarch64.kc_mode = "accumulate",
    annc.aarch64.gemm_problem = {version = 1 : i64, m = 2 : i64, n = 11 : i64, k = 5 : i64, lda = 5 : i64, ldb = 11 : i64, ldc = 11 : i64}, annc.aarch64.gemm_strategy = "s0", annc.aarch64.gemm_candidate = {version = 4 : i64, planning_state = "planned", strategy = "s0", max_thread_count = 1 : i64, thread_count = 1 : i64, tasks_m = 1 : i64, tasks_n = 1 : i64, shard_direction = "rows", thread_partition = "static-2d"}
  } ins(%a, %b : memref<2x5xf32>, memref<5x11xf32>)
    outs(%c : memref<2x11xf32>)
  return
}
