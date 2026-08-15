// RUN: annc-asm %s -aarch64-gemm-leaf-materialization -aarch64-gemm-microkernel-lowering -aarch64-gemm-abi-lowering -aarch64-verify-gemm-schedule | FileCheck %s

// CHECK-DAG: llvm.func @annc_aarch64_neon_kernel_mr3_n8_kg_r0_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32)
// CHECK-DAG: llvm.func @annc_aarch64_neon_kernel_mr2_n3_kg_r1_acc_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32)
// CHECK: llvm.func @annc_aarch64_neon_packb_f32(!llvm.ptr, !llvm.ptr, i32, i32, i32)
// CHECK-NOT: @annc_aarch64_neon_microkernel_f32
// CHECK-LABEL: func.func @pack_and_call(
// CHECK: %[[PACKED:.*]] = memref.alloca() {alignment = 64 : i64} : memref<128xf32>
// CHECK-NOT: memref.dim
// CHECK-NOT: arith.minui
// CHECK: memref.extract_aligned_pointer_as_index %{{.*}}
// CHECK: llvm.call @annc_aarch64_neon_packb_f32
// CHECK: llvm.call @annc_aarch64_neon_kernel_mr3_n8_kg_r0_f32
// CHECK-NOT: linalg.matmul
// CHECK-NOT: __annc_aarch64_gemm
// CHECK-LABEL: func.func @tail_and_accumulate(
// CHECK: llvm.call @annc_aarch64_neon_packb_f32
// CHECK: llvm.call @annc_aarch64_neon_kernel_mr2_n3_kg_r1_acc_f32
// CHECK-NOT: @annc_aarch64_neon_microkernel_f32
// CHECK-NOT: memref.dim
// CHECK-NOT: arith.minui
// CHECK: return
func.func @pack_and_call(%c: memref<3x8xf32>, %a: memref<3x8xf32>,
                         %b: memref<8x8xf32>) {
  linalg.matmul {
    annc.aarch64.gemm_stage = "kernel_tiled",
    annc.aarch64.kc_mode = "overwrite",
    annc.aarch64.gemm_plan = {
      target_arch = "kp950", isa = "neon", data_type = "f32",
      kernel_family = "annc-neon-f32-v1", first_kc_mode = "overwrite",
      k = 8 : i64, kc = 8 : i64, lda = 8 : i64, ldb = 8 : i64,
      ldc = 8 : i64, m = 3 : i64, macro_order = "mkn", mc = 3 : i64,
      mr = 3 : i64, n = 8 : i64, nc = 16 : i64,
      next_kc_mode = "accumulate", panel_lanes = 2 : i64,
      rhs_packing = "packed", thread_count = 1 : i64,
      thread_partition = "serial", vector_length_bytes = 16 : i64,
      version = 1 : i64
    }
  } ins(%a, %b : memref<3x8xf32>, memref<8x8xf32>)
    outs(%c : memref<3x8xf32>)
  return
}

func.func @tail_and_accumulate(%c: memref<2x3xf32>, %a: memref<2x5xf32>,
                               %b: memref<5x3xf32>) {
  linalg.matmul {
    annc.aarch64.gemm_stage = "kernel_tiled",
    annc.aarch64.kc_mode = "accumulate",
    annc.aarch64.gemm_plan = {
      target_arch = "kp950", isa = "neon", data_type = "f32",
      kernel_family = "annc-neon-f32-v1", first_kc_mode = "overwrite",
      k = 5 : i64, kc = 8 : i64, lda = 5 : i64, ldb = 3 : i64,
      ldc = 3 : i64, m = 2 : i64, macro_order = "mkn", mc = 3 : i64,
      mr = 3 : i64, n = 3 : i64, nc = 16 : i64,
      next_kc_mode = "accumulate", panel_lanes = 2 : i64,
      rhs_packing = "packed", thread_count = 1 : i64,
      thread_partition = "serial", vector_length_bytes = 16 : i64,
      version = 1 : i64
    }
  } ins(%a, %b : memref<2x5xf32>, memref<5x3xf32>)
    outs(%c : memref<2x3xf32>)
  return
}
