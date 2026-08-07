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
                         %b: memref<8x16xf32>) {
  linalg.matmul {
    annc.aarch64.gemm_stage = "kernel_tiled",
    annc.aarch64.kc_mode = "overwrite",
    annc.aarch64.gemm_plan = {
      target_arch = "kp950", isa = "sve", data_type = "f32",
      kernel_family = "annc-sve-f32-v1", first_kc_mode = "overwrite",
      k = 8 : i64, kc = 8 : i64, lda = 8 : i64, ldb = 16 : i64,
      ldc = 16 : i64, m = 3 : i64, macro_order = "mkn", mc = 3 : i64,
      mr = 3 : i64, n = 16 : i64, nc = 16 : i64,
      next_kc_mode = "accumulate", panel_lanes = 2 : i64,
      rhs_packing = "packed", thread_count = 1 : i64,
      thread_partition = "serial", vector_length_bytes = 32 : i64,
      version = 1 : i64
    }
  } ins(%a, %b : memref<3x8xf32>, memref<8x16xf32>)
    outs(%c : memref<3x16xf32>)
  return
}

func.func @sve_tail_accumulate(%c: memref<2x11xf32>, %a: memref<2x5xf32>,
                               %b: memref<5x11xf32>) {
  linalg.matmul {
    annc.aarch64.gemm_stage = "kernel_tiled",
    annc.aarch64.kc_mode = "accumulate",
    annc.aarch64.gemm_plan = {
      target_arch = "kp950", isa = "sve", data_type = "f32",
      kernel_family = "annc-sve-f32-v1", first_kc_mode = "overwrite",
      k = 5 : i64, kc = 8 : i64, lda = 5 : i64, ldb = 11 : i64,
      ldc = 11 : i64, m = 2 : i64, macro_order = "mkn", mc = 3 : i64,
      mr = 3 : i64, n = 11 : i64, nc = 16 : i64,
      next_kc_mode = "accumulate", panel_lanes = 2 : i64,
      rhs_packing = "packed", thread_count = 1 : i64,
      thread_partition = "serial", vector_length_bytes = 32 : i64,
      version = 1 : i64
    }
  } ins(%a, %b : memref<2x5xf32>, memref<5x11xf32>)
    outs(%c : memref<2x11xf32>)
  return
}
