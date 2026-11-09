// RUN: annc-asm %s -atir-fast-codegen -lower-context-kernel-entries -convert-atir-to-linalg | FileCheck %s
// RUN: annc-asm %s -atir-fast-codegen -lower-context-kernel-entries -convert-atir-to-linalg | mlir-opt -convert-func-to-llvm -reconcile-unrealized-casts | FileCheck %s --check-prefix=LLVM
// CHECK: func.func private @ANNCKernel_ContextV3_KPFusedGather_float_int64_t_int32_t_int32_t_auto(!llvm.ptr) -> i32 attributes {llvm.emit_c_interface}
// CHECK-LABEL: func.func private @fused_kp_fused_gather(
// CHECK-SAME: %[[CTX:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK-NOT: memref
// CHECK: %[[STATUS:[a-zA-Z0-9_]+]] = call @ANNCKernel_ContextV3_KPFusedGather_float_int64_t_int32_t_int32_t_auto(%[[CTX]]) : (!llvm.ptr) -> i32
// CHECK: return %[[STATUS]] : i32
// LLVM: llvm.func @_mlir_ciface_ANNCKernel_ContextV3_KPFusedGather_float_int64_t_int32_t_int32_t_auto(!llvm.ptr) -> i32
// LLVM-LABEL: llvm.func @_mlir_ciface_fused_kp_fused_gather(
// LLVM-SAME: %{{[a-zA-Z0-9_]+}}: !llvm.ptr) -> i32

module attributes {module.state = "atir"} {
  func.func private @fused_kp_fused_gather(
      %data: !atir.tensor<2x4xf32>, %keys: !atir.tensor<4x3xi64>,
      %begin: !atir.tensor<2xi32>)
      -> (!atir.tensor<?xi64>, !atir.tensor<?xi32>, !atir.tensor<?x4xf32>)
      attributes {
        annc.kernel, llvm.emit_c_interface,
        fusion.metadata = {abi = "annc_context_v3"}
      } {
    %axis = atir.constant "public" @"axis" ->
      <i32, name = "axis", data = dense<0> : tensor<i32>>
    %end = atir.constant "public" @"end" ->
      <2xi32, name = "end", data = dense<[0, 0]> : tensor<2xi32>>
    %stride = atir.constant "public" @"stride" ->
      <2xi32, name = "stride", data = dense<[1, 1]> : tensor<2xi32>>
    %slice_buffer = "atir.buffer"() : () -> !atir.tensor<?xi64>
    %slice = atir.StridedSlice %slice_buffer, %keys, %begin, %end, %stride {
      beginMask = 1 : i32, endMask = 1 : i32, ellipsisMask = 0 : i32,
      newAxisMask = 0 : i32, shrinkAxisMask = 2 : i32
    } : !atir.tensor<?xi64>, !atir.tensor<4x3xi64>, !atir.tensor<2xi32>,
        !atir.tensor<2xi32, name = "end", data = dense<0> : tensor<2xi32>>,
        !atir.tensor<2xi32, name = "stride", data = dense<1> : tensor<2xi32>>
        -> !atir.tensor<?xi64>
    %unique_values, %unique_indices = atir.Unique %slice :
      (!atir.tensor<?xi64>) -> (!atir.tensor<?xi64>, !atir.tensor<?xi32>)
    %unique2_values, %unique2_indices = atir.Unique %unique_values :
      (!atir.tensor<?xi64>) ->
      (!atir.tensor<?xi64>, !atir.tensor<?xi32>)
    %inner_buffer = "atir.buffer"() : () -> !atir.tensor<?x4xf32>
    %inner = atir.Gather %inner_buffer, %data, %unique2_values, %axis :
      !atir.tensor<?x4xf32>, !atir.tensor<2x4xf32>, !atir.tensor<?xi64>,
      !atir.tensor<i32, name = "axis", data = dense<0> : tensor<i32>>
      -> !atir.tensor<?x4xf32>
    %outer_buffer = "atir.buffer"() : () -> !atir.tensor<?x4xf32>
    %outer = atir.Gather %outer_buffer, %inner, %unique2_indices, %axis :
      !atir.tensor<?x4xf32>, !atir.tensor<?x4xf32>, !atir.tensor<?xi32>,
      !atir.tensor<i32, name = "axis", data = dense<0> : tensor<i32>>
      -> !atir.tensor<?x4xf32>
    return %unique_values, %unique_indices, %outer : !atir.tensor<?xi64>,
      !atir.tensor<?xi32>, !atir.tensor<?x4xf32>
  }
}
