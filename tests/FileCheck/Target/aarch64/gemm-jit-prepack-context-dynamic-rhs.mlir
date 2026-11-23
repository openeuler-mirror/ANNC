// RUN: echo '[1]' > %t.json
// RUN: env ANNC_ENABLE_LOWERING_CONTEXT_ABI=1 annc-asm %s -aarch64-extract-gemm-problem -aarch64-select-gemm-strategy='config-path=%S/Inputs/gemm-tuning-hip09-neon-test.json enable-prepack=true' -aarch64-gemm-thread-planning -aarch64-gemm-prepack-rhs='jit-inputs=%t.json packed-c=%t.c' | FileCheck %s
// RUN: FileCheck %s --check-prefix=PACK < %t.c

// The GEMM RHS traces to a context input slot whose fusion.metadata boundary
// record marks it dynamic (an activation, not a checkpoint weight): strategy
// selection must not choose kPrepacked, and prepareJitRhs must not emit any
// packing code even when the manifest lists the slot.
// CHECK-LABEL: func.func @dynamic_rhs_context(
// CHECK: rhs_packing = "packed"
// CHECK-NOT: rhs_data_input
// PACK-NOT: static void pack_0
// PACK: annc_jit_packed_rhs_v1
// PACK-SAME: *count=0
func.func private @annc_context_get_input_memref(!llvm.ptr, i32) -> !llvm.ptr

func.func @dynamic_rhs_context(%context: !llvm.ptr)
    attributes {fusion.metadata = {abi = "annc_context_v3", args = [
      {dtype = "f32", rank = 2, role = "dynamic", shape = array<i64: 32, 73>, tf_name = "input"},
      {dtype = "f32", rank = 2, role = "dynamic", shape = array<i64: 73, 67>, tf_name = "activation"}
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
