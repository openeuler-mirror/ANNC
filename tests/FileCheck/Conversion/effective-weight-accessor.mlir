// RUN: annc-asm --split-input-file %s -convert-atir-to-linalg | FileCheck %s
// The accessor must expose the very same global that the kernel consumes.
// CHECK: memref.global "private" @[[WEIGHT:__annc_site_const_[0-9]+]]
// CHECK-LABEL: func.func @folded
// CHECK: memref.get_global @[[WEIGHT]]
// CHECK-LABEL: func.func @__annc_effective_weight_folded() -> i64
// CHECK-SAME: llvm.emit_c_interface
// CHECK: %[[BUFFER:.*]] = memref.get_global @[[WEIGHT]]
// CHECK: %[[PTR:.*]] = memref.extract_aligned_pointer_as_index %[[BUFFER]]
// CHECK: %[[ADDR:.*]] = arith.index_cast %[[PTR]] : index to i64
// CHECK: return %[[ADDR]] : i64
// CHECK-LABEL: func.func @__annc_effective_weight_size_folded() -> i64
// CHECK-SAME: llvm.emit_c_interface
// CHECK: %[[SIZE:.*]] = arith.constant 4 : i64
// CHECK: return %[[SIZE]] : i64
module {
  func.func @folded() -> !atir.tensor<1x2x2xf32, data = dense<2.0> : tensor<1x2x2xf32>>
      attributes {annc.kernel, fusion.pattern = "batched_matmul_add"} {
    %w = atir.constant "private" @pre_affine_weight -> <1x2x2xf32, data = dense<2.0> : tensor<1x2x2xf32>> {annc.effective_weight}
    return %w : !atir.tensor<1x2x2xf32, data = dense<2.0> : tensor<1x2x2xf32>>
  }
}

// -----

// CHECK-LABEL: func.func @__annc_effective_weight_unfolded() -> i64
// CHECK-SAME: llvm.emit_c_interface
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK: return %[[ZERO]] : i64
// CHECK-LABEL: func.func @__annc_effective_weight_size_unfolded() -> i64
// CHECK-SAME: llvm.emit_c_interface
// CHECK: %[[ZERO_SIZE:.*]] = arith.constant 0 : i64
// CHECK: return %[[ZERO_SIZE]] : i64
module {
  func.func @unfolded() attributes {annc.kernel, fusion.pattern = "batched_matmul_add"} {
    return
  }
}
