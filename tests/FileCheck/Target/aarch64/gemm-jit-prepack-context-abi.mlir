// RUN: echo '[1]' > %t.json
// RUN: env ANNC_ENABLE_LOWERING_CONTEXT_ABI=1 annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy='config-path=%S/Inputs/gemm-tuning-hip09-neon-test.json enable-prepack=true intra-thread-count=8' -aarch64-gemm-thread-planning -aarch64-gemm-prepack-rhs='jit-inputs=%t.json packed-c=%t.c' -aarch64-gemm-thread-tiling -aarch64-gemm-cache-blocking -aarch64-gemm-kernel-tiling -aarch64-gemm-leaf-materialization | grep -vE "^\[[Aa][Nn][Nn][Cc][ :]" | FileCheck %s
// RUN: FileCheck %s --check-prefix=PACK < %t.c

// Context ABI variant of gemm-jit-prepack.mlir: every input is fetched through
// annc_context_get_input_memref, so the RHS root is the descriptor load of the
// slot call instead of an entry BlockArgument. The immutable slot number equals
// the erased entry argument index, and prepack eligibility comes from the
// fusion.metadata boundary record instead of annc.aarch64.rhs_name (the V2
// lowering never attaches that attribute).
// CHECK-NOT: memref.global
// CHECK-LABEL: func.func @jit_prepack_context(
// CHECK: call @annc_context_parallel_for_gemm
// CHECK-LABEL: func.func private @__annc_gemm_thread_task_
// CHECK-NOT: memref.get_global
// CHECK-NOT: call @__annc_aarch64_gemm_pack_b_leaf
// CHECK: call @__annc_aarch64_gemm_microkernel_leaf
// PACK: static void pack_0
// PACK: annc_aarch64_neon_packb_f32
// PACK: {1,1,73,67,8,16,0,4,4964,64}
// PACK: annc_jit_packed_rhs_v1
// PACK-SAME: *count=1
func.func private @annc_context_get_input_memref(!llvm.ptr, i32) -> !llvm.ptr

func.func @jit_prepack_context(%context: !llvm.ptr)
    attributes {fusion.metadata = {abi = "annc_context_v3", args = [
      {dtype = "f32", rank = 2, role = "dynamic", shape = array<i64: 32, 73>, tf_name = "input"},
      {dtype = "f32", rank = 2, role = "fixed", shape = array<i64: 73, 67>, tf_name = "weight"}
    ]}} {
  %slot0 = llvm.mlir.constant(0 : i32) : i32
  %lhs_ptr = call @annc_context_get_input_memref(%context, %slot0)
      : (!llvm.ptr, i32) -> !llvm.ptr
  %lhs_desc = llvm.load %lhs_ptr
      : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %a = builtin.unrealized_conversion_cast %lhs_desc
      : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<32x73xf32>
  %slot1 = llvm.mlir.constant(1 : i32) : i32
  %rhs_ptr = call @annc_context_get_input_memref(%context, %slot1)
      : (!llvm.ptr, i32) -> !llvm.ptr
  %rhs_desc = llvm.load %rhs_ptr
      : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %b = builtin.unrealized_conversion_cast %rhs_desc
      : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<73x67xf32>
  %slot2 = llvm.mlir.constant(2 : i32) : i32
  %out_ptr = call @annc_context_get_input_memref(%context, %slot2)
      : (!llvm.ptr, i32) -> !llvm.ptr
  %out_desc = llvm.load %out_ptr
      : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %c = builtin.unrealized_conversion_cast %out_desc
      : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<32x67xf32>
  linalg.matmul
      ins(%a, %b : memref<32x73xf32>, memref<73x67xf32>)
      outs(%c : memref<32x67xf32>)
  return
}
