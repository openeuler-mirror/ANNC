// RUN: ANNC_GEMM=1 annc-opt %s -atir-op-fusion-v2 | FileCheck %s
// RUN: ANNC_GEMM=99 annc-opt %s -atir-op-fusion-v2 | FileCheck %s
// RUN: ANNC_GEMM=0 annc-opt %s -atir-op-fusion-v2 | FileCheck %s --check-prefix=GEMM-OFF
// RUN: annc-opt %s -atir-op-fusion-v2 | FileCheck %s --check-prefix=GEMM-OFF
//
// op-fusion-v2 pattern: dot GEMM (the V2 split following the V1
// FuseMatMulAsFuncCall contract). The split happens before lowering: the
// stablehlo.dot anchor moves into the kernel verbatim (legalize attrs kept)
// and the result travels through the return value. The backend gemm contract
// (full fusion.metadata + top-level fusion.pattern + jit/kernel/ciface)
// mirrors the V1 subgraph (wd_dcn/presort shape: dynamic lhs [?,K])
module {
  func.func @main(%arg0: tensor<?x5xf32>, %arg1: tensor<5x1xf32>) -> tensor<?x1xf32> attributes {tf.arg_names = ["model/input_x:0", "model/dense/kernel:0"]} {
    %0 = stablehlo.dot %arg0, %arg1, precision = [DEFAULT, DEFAULT] : (tensor<?x5xf32>, tensor<5x1xf32>) -> tensor<?x1xf32> loc("model/dense/MatMul")
    return %0 : tensor<?x1xf32>
  }

  // main side: one call replaces the dot (two inputs + the return value).
  // CHECK-LABEL: func.func @main(
  // CHECK: %[[CALL:.*]] = call @fused_matmul_{{[0-9a-fA-F]+}}(%arg0, %arg1) : (tensor<?x5xf32>, tensor<5x1xf32>) -> tensor<?x1xf32>
  // CHECK: return %[[CALL]]

  // Negative: a non-f32 (i32) dot stays untouched.
  // CHECK-LABEL: func.func @main_i32
  // CHECK: stablehlo.dot
  // Negative: dot_general (BatchMatMulV2 form) is out of the first batch.
  // CHECK-LABEL: func.func @main_bmm
  // CHECK: stablehlo.dot_general
  func.func @main_i32(%arg0: tensor<?x5xi32>, %arg1: tensor<5x1xi32>) -> tensor<?x1xi32> {
    %0 = stablehlo.dot %arg0, %arg1, precision = [DEFAULT, DEFAULT] : (tensor<?x5xi32>, tensor<5x1xi32>) -> tensor<?x1xi32> loc("model/dense_i32/MatMul")
    return %0 : tensor<?x1xi32>
  }

  func.func @main_bmm(%arg0: tensor<?x1x492xf32>, %arg1: tensor<?x492x1xf32>) -> tensor<?x1x1xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<?x1x492xf32>, tensor<?x492x1xf32>) -> tensor<?x1x1xf32> loc("model/dense_bmm/BatchMatMulV2")
    return %0 : tensor<?x1x1xf32>
  }

  // Kernel: signature (lhs, rhs) -> out; the body is the moved
  // stablehlo.dot (pre-lowering carrier); top-level attrs jit/kernel/
  // fusion.pattern (prune contract) / llvm.emit_c_interface.
  // Metadata follows the context v3 contract (outlineContextBuiltin line):
  // args fixed-to-dynamic (static rhs weight first), kernel_arg_order empty
  // (outputs allocated through the context), symbolic_signature empty.
  // CHECK: func.func private @fused_matmul_{{[0-9a-fA-F]+}}(%{{.*}}: tensor<?x5xf32>, %{{.*}}: tensor<5x1xf32>) -> tensor<?x1xf32> attributes {annc.execution_mode = "jit", annc.kernel, fusion.metadata =
  // CHECK-SAME: abi = "annc_context_v3"
  // CHECK-SAME: role = "fixed"
  // CHECK-SAME: role = "dynamic"
  // CHECK-SAME: execution_mode = "jit"
  // CHECK-SAME: kernel_arg_order = []
  // CHECK-SAME: kernel_name = "fused_matmul_{{[0-9a-fA-F]+}}"
  // CHECK-SAME: role = "output"
  // CHECK-SAME: tf_name = "model/dense/MatMul:0"
  // CHECK-SAME: symbolic_signature = ""
  // CHECK-SAME: template_fingerprint = "
  // CHECK-SAME: tf.name = "annc_fused_model_dense_MatMul"
  // CHECK-SAME: fusion.pattern = "matmul", llvm.emit_c_interface}
  // CHECK: stablehlo.dot %{{.*}}, %{{.*}}, precision = [DEFAULT, DEFAULT] : (tensor<?x5xf32>, tensor<5x1xf32>) -> tensor<?x1xf32>
  // CHECK: return %{{.*}} : tensor<?x1xf32>

  // With GEMM off (default/0) the dot stays untouched (no site).
  // GEMM-OFF-LABEL: func.func @main(
  // GEMM-OFF: stablehlo.dot
  // GEMM-OFF-NOT: fused_matmul
}
