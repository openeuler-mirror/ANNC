// RUN: rm -rf %t.dir && mkdir -p %t.dir && split-file %s %t.dir
// RUN: annc-opt --split-input-file %t.dir/discovery.mlir -atir-gemm-epilogue-fusion -o %t.discovery.mlir
// RUN: FileCheck %t.dir/discovery.mlir < %t.discovery.mlir
// RUN: annc-asm --split-input-file %t.discovery.mlir -aarch64-atir-gemm-epilogue-commit | FileCheck %t.dir/discovery.mlir --check-prefix=COMMIT
// RUN: annc-asm --split-input-file %t.dir/conversion.mlir -convert-atir-to-linalg | FileCheck %t.dir/conversion.mlir
// RUN: annc-opt %t.dir/production.mlir -atir-gemm-epilogue-fusion -o %t.production.mlir
// RUN: not annc-opt %t.dir/invalid-candidate.mlir -atir-op-fusion 2>&1 | FileCheck %t.dir/invalid-candidate.mlir --check-prefix=INVALID
// RUN: annc-opt %t.dir/jit-template.mlir -atir-gemm-epilogue-fusion -atir-op-fusion -o %t.jit-template.mlir
// RUN: annc-asm %t.jit-template.mlir -aarch64-atir-gemm-epilogue-commit | FileCheck %t.dir/jit-template.mlir --check-prefix=JIT-COMMIT
// RUN: annc-opt %t.dir/neon-vecmat-epilogue.mlir -atir-gemm-epilogue-fusion -o %t.neon-vecmat.mlir
// RUN: annc-asm %t.neon-vecmat.mlir -annc-aarch64-gemm-pipeline='config-path=%S/Inputs/gemm-tuning-hip09-neon-test.json' | FileCheck %t.dir/neon-vecmat-epilogue.mlir --check-prefix=NEON-VECMAT
// RUN: annc-opt %t.dir/threaded-epilogue.mlir -atir-gemm-epilogue-fusion -o %t.threaded-epilogue.mlir
// RUN: annc-asm %t.threaded-epilogue.mlir -annc-aarch64-gemm-pipeline='config-path=%S/Inputs/gemm-tuning-test.json intra-thread-count=4' | FileCheck %t.dir/threaded-epilogue.mlir --check-prefix=THREAD-EPILOGUE
// Euler dev selects direct/packed/prepacked from the target plan; the legacy
// ANNC_GEMM_RHS_PACKING override is intentionally not part of this contract.
// RUN: annc-asm %t.production.mlir -annc-aarch64-gemm-pipeline='config-path=%S/Inputs/gemm-tuning-test.json' | FileCheck %t.dir/production.mlir --check-prefixes=SVE,SVE-DIRECT,SVE-PACKED
// RUN: annc-asm %t.production.mlir -annc-aarch64-gemm-pipeline='config-path=%S/Inputs/gemm-tuning-hip09-neon-test.json' | FileCheck %t.dir/production.mlir --check-prefixes=NEON,NEON-DIRECT,NEON-PACKED
// RUN: env ANNC_ENABLE_EPILOGUE_MATVEC=0 annc-asm %t.production.mlir -annc-aarch64-gemm-pipeline='config-path=%S/Inputs/gemm-tuning-hip09-neon-test.json' | FileCheck %t.dir/production.mlir --check-prefix=NEON-MATVEC-OFF
// RUN: annc-asm %t.production.mlir -aarch64-atir-gemm-epilogue-commit | FileCheck %t.dir/production.mlir --check-prefix=COMMIT
// RUN: rm -rf %t.aot && annc-asm %t.production.mlir -annc-aarch64-gemm-pipeline='config-path=%S/Inputs/gemm-tuning-hip09-neon-test.json' -o %t.mlir
// RUN: annc-model-kernel-export %t.mlir --output-dir %t.aot
// RUN: find %t.aot/model_generated_kernels -name '*.S' -print | count 7
// RUN: FileCheck %t.dir/production.mlir --check-prefix=AOT-MANIFEST < %t.aot/model_generated_kernels/manifest.json
// RUN: env ANNC_GEMM_ASM_SOURCE_DIR=/nonexistent LIBRARY_PATH="${LIBRARY_PATH:?}" annc -v %t.mlir --shared -o %t.invalid.so > %t.error 2>&1; test $? -ne 0
// RUN: FileCheck %t.dir/production.mlir --check-prefix=AOT-ERROR < %t.error
// RUN: env ANNC_GEMM_ASM_SOURCE_DIR="${ANNC_GEMM_ASM_SOURCE_DIR:?}" LIBRARY_PATH="${LIBRARY_PATH:?}" annc %t.mlir --shared -o %t.so
// RUN: python3 -c "import ctypes; ctypes.CDLL(r'%t.so')"
// RUN: llvm-nm -D --defined-only %t.so | grep generated | count 26
// RUN: llvm-nm -D --defined-only %t.so | FileCheck %t.dir/production.mlir --check-prefix=AOT-LINK

//--- production.mlir

// AOT-MANIFEST-COUNT-7: "file"
// AOT-ERROR: ANNC_GEMM_ASM_SOURCE_DIR must point to a GEMM-ASM checkout
// AOT-LINK: T {{.*}}generated_add_n_i1_relu_{{.*}}f32

// SVE-DAG: llvm.func @annc_aarch64_sve_kernel_rm_{{.*generated.*}}(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr)
// SVE-DAG: llvm.func @annc_aarch64_sve_kernel_mr{{.*generated.*}}(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr)
// NEON-DAG: llvm.func @annc_aarch64_neon_kernel_rm_{{.*generated.*}}(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr)
// NEON-DAG: llvm.func @annc_aarch64_neon_kernel_mr{{.*generated.*}}(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr)

// SVE-DIRECT-LABEL: func.func @smallshape_gemm(
// SVE-DIRECT: llvm.call @annc_aarch64_sve_kernel_rm_
// NEON-DIRECT-LABEL: func.func @smallshape_gemm(
// NEON-DIRECT: llvm.call @annc_aarch64_neon_kernel_rm_mr1_n5_kg_r1_f32
func.func @smallshape_gemm(
    %a: !atir.tensor<1x5xf32>, %b: !atir.tensor<5x5xf32>,
    %out: !atir.tensor<1x5xf32>) {
  %matmul = "atir.MatMul"(%out, %a, %b) <{
    left_transpose = false, output_transpose = false,
    right_transpose = false, withBias = false
  }> : (!atir.tensor<1x5xf32>, !atir.tensor<1x5xf32>,
        !atir.tensor<5x5xf32>) -> !atir.tensor<1x5xf32>
  return
}

// The NEON matrix-vector family retains its specialized leaf when an epilogue
// is present; K=641 spans multiple KC blocks and only Final-K is generated.
// NEON-DIRECT-LABEL: func.func @gemv_suffix_add_relu(
// NEON-DIRECT: llvm.call @annc_aarch64_neon_matvec_mr4_k8_f32
// NEON-DIRECT: llvm.call @annc_aarch64_neon_matvec_mr4_k1_generated_add_scalar_i1_relu_lBF800000_acc_f32
// NEON-DIRECT-NOT: arith.maxnumf
// NEON-MATVEC-OFF-LABEL: func.func @gemv_suffix_add_relu(
// NEON-MATVEC-OFF: llvm.call @annc_aarch64_neon_kernel_rm_
// NEON-MATVEC-OFF-NOT: llvm.call @annc_aarch64_neon_matvec
func.func @gemv_suffix_add_relu(
    %a: !atir.tensor<8x641xf32>, %b: !atir.tensor<641x1xf32>,
    %out: !atir.tensor<8x1xf32>, %bias: !atir.tensor<1xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<8x1xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false, output_transpose = false,
    right_transpose = false, withBias = false
  }> : (!atir.tensor<8x1xf32>, !atir.tensor<8x641xf32>,
        !atir.tensor<641x1xf32>) -> !atir.tensor<8x1xf32>
  %add_tmp = "atir.buffer"() : () -> !atir.tensor<8x1xf32>
  %add = "atir.Add"(%add_tmp, %matmul, %bias) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<8x1xf32>, !atir.tensor<8x1xf32>,
        !atir.tensor<1xf32>) -> !atir.tensor<8x1xf32>
  %relu = "atir.Relu"(%out, %add) <{relu_limit = -1.0 : f32}> :
      (!atir.tensor<8x1xf32>, !atir.tensor<8x1xf32>) ->
      !atir.tensor<8x1xf32>
  return
}

// A destination-style GEMM just above the smallshape threshold exercises
// planner-selected packed RHS, Single-K fusion, and both K/N tails.
// COMMIT-LABEL: func.func @single_tail_add_relu(
// COMMIT: "atir.MatMul"
// COMMIT-SAME: withBias = false
// COMMIT-SAME: annc.gemm.epilogue =
// COMMIT-NOT: "atir.Add"
// COMMIT-NOT: "atir.Relu"
// SVE-PACKED-LABEL: func.func @single_tail_add_relu(
// SVE-PACKED: llvm.call @annc_aarch64_sve_kernel_mr{{.*}}_generated_add_n_i1_relu_{{.*}}_f32
// SVE-PACKED-NOT: arith.maxnumf
// NEON-PACKED-LABEL: func.func @single_tail_add_relu(
// NEON-PACKED: llvm.call @annc_aarch64_neon_kernel_mr{{.*}}_kg_r1_generated_add_n_i1_relu_{{.*}}_f32
// NEON-PACKED-NOT: arith.maxnumf
func.func @single_tail_add_relu(
    %a: !atir.tensor<2x5xf32>, %b: !atir.tensor<5x501xf32>,
    %out: !atir.tensor<2x501xf32>, %bias: !atir.tensor<501xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<2x501xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false, output_transpose = false,
    right_transpose = false, withBias = false
  }> : (!atir.tensor<2x501xf32>, !atir.tensor<2x5xf32>,
        !atir.tensor<5x501xf32>) -> !atir.tensor<2x501xf32>
  %add_tmp = "atir.buffer"() : () -> !atir.tensor<2x501xf32>
  %add = "atir.Add"(%add_tmp, %matmul, %bias) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<2x501xf32>, !atir.tensor<2x501xf32>,
        !atir.tensor<501xf32>) -> !atir.tensor<2x501xf32>
  %relu = "atir.Relu"(%out, %add) <{relu_limit = 6.0 : f32}> :
      (!atir.tensor<2x501xf32>, !atir.tensor<2x501xf32>) ->
      !atir.tensor<2x501xf32>
  return
}

// This Single-K call and multi_k_add's Final-K call select different callee
// symbols but share one generated macro family and therefore one AOT file.
func.func @single_add(
    %a: !atir.tensor<1x5xf32>, %b: !atir.tensor<5x5xf32>,
    %out: !atir.tensor<1x5xf32>, %bias: !atir.tensor<5xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<1x5xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false, output_transpose = false,
    right_transpose = false, withBias = false
  }> : (!atir.tensor<1x5xf32>, !atir.tensor<1x5xf32>,
        !atir.tensor<5x5xf32>) -> !atir.tensor<1x5xf32>
  %add = "atir.Add"(%out, %matmul, %bias) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<1x5xf32>, !atir.tensor<1x5xf32>,
        !atir.tensor<5xf32>) -> !atir.tensor<1x5xf32>
  return
}

// M broadcast is not supported by the fused leaf. The fixed candidate keeps
// the preceding N-broadcast Add and materializes Mul as a suffix.
// SVE-DIRECT-LABEL: func.func @add_then_m_mul(
// SVE-DIRECT: llvm.call @annc_aarch64_sve_kernel_rm_mr3_n1vl_generated_add_n_i1_f32
// SVE-DIRECT: arith.mulf
// NEON-DIRECT-LABEL: func.func @add_then_m_mul(
// NEON-DIRECT: llvm.call @annc_aarch64_neon_kernel_rm_{{.*}}_generated_add_n_i1_f32
// NEON-DIRECT: arith.mulf
func.func @add_then_m_mul(
    %a: !atir.tensor<3x5xf32>, %b: !atir.tensor<5x5xf32>,
    %out: !atir.tensor<3x5xf32>, %bias: !atir.tensor<5xf32>,
    %scale: !atir.tensor<3x1xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<3x5xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false, output_transpose = false,
    right_transpose = false, withBias = false
  }> : (!atir.tensor<3x5xf32>, !atir.tensor<3x5xf32>,
        !atir.tensor<5x5xf32>) -> !atir.tensor<3x5xf32>
  %add_tmp = "atir.buffer"() : () -> !atir.tensor<3x5xf32>
  %add = "atir.Add"(%add_tmp, %matmul, %bias) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<3x5xf32>, !atir.tensor<3x5xf32>,
        !atir.tensor<5xf32>) -> !atir.tensor<3x5xf32>
  %mul = "atir.Mul"(%out, %add, %scale) :
      (!atir.tensor<3x5xf32>, !atir.tensor<3x5xf32>,
       !atir.tensor<3x1xf32>) -> !atir.tensor<3x5xf32>
  return
}

// Rank-1 length-one inputs use a constant-zero map and remain scalar through
// cache and kernel tiling.
// SVE-DIRECT-LABEL: func.func @single_scalar_add(
// SVE-DIRECT: llvm.call @annc_aarch64_sve_kernel_rm_{{.*}}_generated_add_scalar_i1_f32
// NEON-DIRECT-LABEL: func.func @single_scalar_add(
// NEON-DIRECT: llvm.call @annc_aarch64_neon_kernel_rm_{{.*}}_generated_add_scalar_i1_f32
func.func @single_scalar_add(
    %a: !atir.tensor<1x5xf32>, %b: !atir.tensor<5x5xf32>,
    %out: !atir.tensor<1x5xf32>, %scalar: !atir.tensor<1xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<1x5xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false, output_transpose = false,
    right_transpose = false, withBias = false
  }> : (!atir.tensor<1x5xf32>, !atir.tensor<1x5xf32>,
        !atir.tensor<5x5xf32>) -> !atir.tensor<1x5xf32>
  %add = "atir.Add"(%out, %matmul, %scalar) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<1x5xf32>, !atir.tensor<1x5xf32>,
        !atir.tensor<1xf32>) -> !atir.tensor<1x5xf32>
  return
}

// A unary epilogue still receives a valid EpilogueArgs directory even though
// it has no external pointer slots to populate.
// SVE-DIRECT-LABEL: func.func @single_sigmoid(
// SVE-DIRECT: llvm.call @annc_aarch64_sve_kernel_rm_{{.*}}_generated_sigmoid_f32
// NEON-DIRECT-LABEL: func.func @single_sigmoid(
// NEON-DIRECT: llvm.call @annc_aarch64_neon_kernel_rm_{{.*}}_generated_sigmoid_f32
func.func @single_sigmoid(
    %a: !atir.tensor<1x5xf32>, %b: !atir.tensor<5x5xf32>,
    %out: !atir.tensor<1x5xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<1x5xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false, output_transpose = false,
    right_transpose = false, withBias = false
  }> : (!atir.tensor<1x5xf32>, !atir.tensor<1x5xf32>,
        !atir.tensor<5x5xf32>) -> !atir.tensor<1x5xf32>
  %sigmoid = "atir.Logistic"(%out, %matmul) :
      (!atir.tensor<1x5xf32>, !atir.tensor<1x5xf32>) ->
      !atir.tensor<1x5xf32>
  return
}

// K=17 with KC=8 creates First, Middle, and Final blocks. Only Final may use
// the generated fused symbol; the earlier blocks remain ordinary kernels.
// SVE-DIRECT-LABEL: func.func @multi_k_add(
// SVE-DIRECT: llvm.call @annc_aarch64_sve_kernel_rm_{{.*}}_f32
// SVE-DIRECT: llvm.call @annc_aarch64_sve_kernel_rm_{{.*}}_acc_f32
// SVE-DIRECT: llvm.call @annc_aarch64_sve_kernel_rm_{{.*}}_generated_add_n_i1_acc_f32
// NEON-DIRECT-LABEL: func.func @multi_k_add(
// NEON-DIRECT: llvm.call @annc_aarch64_neon_kernel_rm_{{.*}}_kg_r0_f32
// NEON-DIRECT: llvm.call @annc_aarch64_neon_kernel_rm_{{.*}}_kg_r0_acc_f32
// NEON-DIRECT: llvm.call @annc_aarch64_neon_kernel_rm_{{.*}}_k1_generated_add_n_i1_acc_f32
func.func @multi_k_add(
    %a: !atir.tensor<1x17xf32>, %b: !atir.tensor<17x5xf32>,
    %out: !atir.tensor<1x5xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<1x5xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false, output_transpose = false,
    right_transpose = false, withBias = false
  }> : (!atir.tensor<1x5xf32>, !atir.tensor<1x17xf32>,
        !atir.tensor<17x5xf32>) -> !atir.tensor<1x5xf32>
  %bias = "atir.buffer"() : () -> !atir.tensor<5xf32>
  %add = "atir.Add"(%out, %matmul, %bias) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<1x5xf32>, !atir.tensor<1x5xf32>,
        !atir.tensor<5xf32>) -> !atir.tensor<1x5xf32>
  return
}

//--- discovery.mlir

// CHECK-LABEL: func.func @ordered_add_relu(
// CHECK: "atir.MatMul"
// CHECK-SAME: annc.gemm.epilogue.candidate =
// CHECK-SAME: output_role = "main_d"
// CHECK-SAME: broadcast = "n"
// CHECK-SAME: input = 0 : i64
// CHECK-SAME: opcode = "add"
// CHECK-SAME: limit = 6.000000e+00 : f32
// CHECK-SAME: opcode = "relu"
// CHECK: "atir.Add"
// CHECK: "atir.Relu"
// COMMIT-LABEL: func.func @ordered_add_relu(
// COMMIT: "atir.MatMul"
// COMMIT-SAME: annc.gemm.epilogue =
// COMMIT-NOT: "atir.Add"
// COMMIT-NOT: "atir.Relu"
func.func @ordered_add_relu(
    %a: !atir.tensor<4x8xf32>, %b: !atir.tensor<8x4xf32>,
    %out: !atir.tensor<4x4xf32>, %bias: !atir.tensor<4xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false, output_transpose = false,
    right_transpose = false, withBias = false
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  %add_tmp = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %add = "atir.Add"(%add_tmp, %bias, %matmul) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4xf32>,
        !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32>
  %relu = "atir.Relu"(%out, %add) <{relu_limit = 6.0 : f32}> :
      (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>) ->
      !atir.tensor<4x4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @terminal_reshape(
// CHECK: "atir.MatMul"
// CHECK-SAME: terminal_view = "insert_unit_dimension"
// CHECK: "atir.Add"
// CHECK: atir.Reshape
func.func @terminal_reshape(
    %a: !atir.tensor<3x8xf32>, %b: !atir.tensor<8x4xf32>,
    %out: !atir.tensor<3x1x4xf32>, %bias: !atir.tensor<4xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<3x4xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false, output_transpose = false,
    right_transpose = false, withBias = false
  }> : (!atir.tensor<3x4xf32>, !atir.tensor<3x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<3x4xf32>
  %add_tmp = "atir.buffer"() : () -> !atir.tensor<3x4xf32>
  %add = "atir.Add"(%add_tmp, %matmul, %bias) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<3x4xf32>, !atir.tensor<3x4xf32>,
        !atir.tensor<4xf32>) -> !atir.tensor<3x4xf32>
  %shape = atir.constant "private" @"shape" ->
      <3xi64, data = dense<[3, 1, 4]> : tensor<3xi64>>
  %reshape = atir.Reshape %out, %add, %shape :
      <3x1x4xf32>, <3x4xf32>,
      <3xi64, data = dense<[3, 1, 4]> : tensor<3xi64>> -> <3x1x4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @reshape_without_epilogue(
// CHECK: "atir.MatMul"
// CHECK-NOT: annc.gemm.epilogue.candidate
// CHECK: atir.Reshape
func.func @reshape_without_epilogue(
    %a: !atir.tensor<3x8xf32>, %b: !atir.tensor<8x4xf32>,
    %out: !atir.tensor<3x1x4xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<3x4xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false, output_transpose = false,
    right_transpose = false, withBias = false
  }> : (!atir.tensor<3x4xf32>, !atir.tensor<3x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<3x4xf32>
  %shape = atir.constant "private" @"plain_shape" ->
      <3xi64, data = dense<[3, 1, 4]> : tensor<3xi64>>
  %reshape = atir.Reshape %out, %matmul, %shape :
      <3x1x4xf32>, <3x4xf32>,
      <3xi64, data = dense<[3, 1, 4]> : tensor<3xi64>> -> <3x1x4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @reject_branched_chain(
// CHECK: "atir.MatMul"
// CHECK-NOT: annc.gemm.epilogue.candidate
// CHECK: "atir.Add"
// CHECK: "atir.Relu"
// CHECK: "atir.Relu"
// COMMIT-LABEL: func.func @reject_branched_chain(
func.func @reject_branched_chain(
    %a: !atir.tensor<4x8xf32>, %b: !atir.tensor<8x4xf32>,
    %bias: !atir.tensor<4xf32>, %out0: !atir.tensor<4x4xf32>,
    %out1: !atir.tensor<4x4xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false, output_transpose = false,
    right_transpose = false, withBias = false
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  %add_tmp = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %add = "atir.Add"(%add_tmp, %matmul, %bias) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>,
        !atir.tensor<4xf32>) -> !atir.tensor<4x4xf32>
  %relu0 = "atir.Relu"(%out0, %add) <{relu_limit = -1.0 : f32}> :
      (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>) ->
      !atir.tensor<4x4xf32>
  %relu1 = "atir.Relu"(%out1, %add) <{relu_limit = -1.0 : f32}> :
      (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>) ->
      !atir.tensor<4x4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @reject_dynamic_broadcast(
// CHECK: "atir.MatMul"
// CHECK-NOT: annc.gemm.epilogue.candidate
// CHECK: "atir.Add"
func.func @reject_dynamic_broadcast(
    %a: !atir.tensor<?x?xf32>, %b: !atir.tensor<?x?xf32>,
    %out: !atir.tensor<?x?xf32>, %bias: !atir.tensor<?xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<?x?xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false, output_transpose = false,
    right_transpose = false, withBias = false
  }> : (!atir.tensor<?x?xf32>, !atir.tensor<?x?xf32>,
        !atir.tensor<?x?xf32>) -> !atir.tensor<?x?xf32>
  %add = "atir.Add"(%out, %matmul, %bias) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<?x?xf32>, !atir.tensor<?x?xf32>,
        !atir.tensor<?xf32>) -> !atir.tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: func.func @matrix_candidate(
// CHECK: "atir.MatMul"
// CHECK-SAME: broadcast = "matrix"
// CHECK: "atir.Add"
// COMMIT-LABEL: func.func @matrix_candidate(
// COMMIT: "atir.MatMul"
// COMMIT-SAME: annc.gemm.epilogue =
// COMMIT-NOT: annc.gemm.epilogue.candidate
// COMMIT-NOT: "atir.Add"
func.func @matrix_candidate(
    %a: !atir.tensor<4x8xf32>, %b: !atir.tensor<8x4xf32>,
    %out: !atir.tensor<4x4xf32>, %addend: !atir.tensor<4x4xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false, output_transpose = false,
    right_transpose = false, withBias = false
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  %add = "atir.Add"(%out, %matmul, %addend) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>,
        !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @reject_transpose(
// CHECK: "atir.MatMul"
// CHECK-NOT: annc.gemm.epilogue.candidate
// CHECK: "atir.Add"
// COMMIT-LABEL: func.func @reject_transpose(
// COMMIT: "atir.MatMul"
// COMMIT: "atir.Add"
func.func @reject_transpose(
    %a: !atir.tensor<2x3xf32>, %b: !atir.tensor<3x4xf32>,
    %out: !atir.tensor<4x2xf32>, %bias: !atir.tensor<2xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<4x2xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false, output_transpose = true,
    right_transpose = false, withBias = false
  }> : (!atir.tensor<4x2xf32>, !atir.tensor<2x3xf32>,
        !atir.tensor<3x4xf32>) -> !atir.tensor<4x2xf32>
  %add = "atir.Add"(%out, %matmul, %bias) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x2xf32>, !atir.tensor<4x2xf32>,
        !atir.tensor<2xf32>) -> !atir.tensor<4x2xf32>
  return
}

// -----

// CHECK-LABEL: func.func @reject_legacy_postop(
// CHECK: "atir.MatMul"
// CHECK: do_relu = true
// CHECK-NOT: annc.gemm.epilogue.candidate
// CHECK: "atir.Add"
func.func @reject_legacy_postop(
    %a: !atir.tensor<4x8xf32>, %b: !atir.tensor<8x4xf32>,
    %out: !atir.tensor<4x4xf32>, %bias: !atir.tensor<4xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    do_relu = true, left_transpose = false, output_transpose = false,
    relu_limit = -1.0 : f32, right_transpose = false, withBias = false
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  %add = "atir.Add"(%out, %matmul, %bias) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>,
        !atir.tensor<4xf32>) -> !atir.tensor<4x4xf32>
  return
}

//--- conversion.mlir

// CHECK-LABEL: func @matrix_add
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK-NOT: "atir.Add"
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @matrix_add(
    %out: !atir.tensor<4x4xf32>, %a: !atir.tensor<4x4xf32>,
    %b: !atir.tensor<4x4xf32>) {
  %add = "atir.Add"(%out, %a, %b) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>,
        !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32>
  return
}

// -----

// CHECK-LABEL: func @dynamic_matrix_add
// CHECK: linalg.generic
// CHECK-NOT: "atir.Add"
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @dynamic_matrix_add(
    %out: !atir.tensor<?x?xf32>, %a: !atir.tensor<?x?xf32>,
    %b: !atir.tensor<?x?xf32>) {
  %add = "atir.Add"(%out, %a, %b) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<?x?xf32>, !atir.tensor<?x?xf32>,
        !atir.tensor<?x?xf32>) -> !atir.tensor<?x?xf32>
  return
}

//--- invalid-candidate.mlir

// INVALID: epilogue candidate does not match the current source chain
func.func @main(
    %a: !atir.tensor<4x8xf32>, %b: !atir.tensor<8x4xf32>,
    %out: !atir.tensor<4x4xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false, output_transpose = false,
    right_transpose = false, withBias = false
  }> {
    annc.gemm.epilogue.candidate = {
      output_role = "main_d",
      steps = [{broadcast = "n", input = 1 : i64, opcode = "add"}],
      terminal_view = "identity", version = 1 : i64
    }
  } : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  %relu = "atir.Relu"(%out, %matmul) <{relu_limit = -1.0 : f32}> :
      (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>) ->
      !atir.tensor<4x4xf32>
  return
}

//--- neon-vecmat-epilogue.mlir

// NEON-VECMAT: llvm.call @annc_aarch64_neon_kernel_rm_{{.*}}generated_add_n_i1{{.*}}_f32
// NEON-VECMAT-NOT: llvm.call @annc_aarch64_neon_vecmat
func.func @vecmat_epilogue(
    %a: !atir.tensor<1x400xf32>, %b: !atir.tensor<400x400xf32>,
    %out: !atir.tensor<1x400xf32>, %bias: !atir.tensor<400xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<1x400xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false, output_transpose = false,
    right_transpose = false, withBias = false
  }> : (!atir.tensor<1x400xf32>, !atir.tensor<1x400xf32>,
        !atir.tensor<400x400xf32>) -> !atir.tensor<1x400xf32>
  %add = "atir.Add"(%out, %matmul, %bias) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<1x400xf32>, !atir.tensor<1x400xf32>,
        !atir.tensor<400xf32>) -> !atir.tensor<1x400xf32>
  return
}

//--- jit-template.mlir

// JIT-COMMIT-LABEL: func.func private @fused_matmul_add_relu_
// JIT-COMMIT: "atir.MatMul"
// JIT-COMMIT-SAME: annc.gemm.epilogue =
// JIT-COMMIT-NOT: annc.gemm.epilogue.candidate
// JIT-COMMIT-NOT: "atir.Add"
// JIT-COMMIT-NOT: "atir.Relu"
func.func @main(
    %a: !atir.tensor<4x8xf32>, %b: !atir.tensor<8x4xf32>,
    %out: !atir.tensor<4x4xf32>, %bias: !atir.tensor<4xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false, output_transpose = false,
    right_transpose = false, withBias = false
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x8xf32>,
        !atir.tensor<8x4xf32>) -> !atir.tensor<4x4xf32>
  %add_tmp = "atir.buffer"() : () -> !atir.tensor<4x4xf32>
  %add = "atir.Add"(%add_tmp, %matmul, %bias) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>,
        !atir.tensor<4xf32>) -> !atir.tensor<4x4xf32>
  %relu = "atir.Relu"(%out, %add) <{relu_limit = -1.0 : f32}> :
      (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>) ->
      !atir.tensor<4x4xf32>
  return
}

//--- threaded-epilogue.mlir

// The thread planner must keep the ordinary dispatcher ABI unchanged while
// transporting binary epilogue inputs through the epilogue-aware task ABI.
// THREAD-EPILOGUE-LABEL: func.func @threaded_add_relu(
// THREAD-EPILOGUE: call @annc_threadpool_parallel_for_gemm_epilogue
// THREAD-EPILOGUE: func.func private @__annc_gemm_thread_task_
// THREAD-EPILOGUE-SAME: memref<*xf32>
// THREAD-EPILOGUE: llvm.call @annc_aarch64_sve_kernel_{{.*}}generated_add_n_i1_relu
// THREAD-EPILOGUE-NOT: arith.addf
func.func @threaded_add_relu(
    %a: !atir.tensor<128x128xf32>, %b: !atir.tensor<128x128xf32>,
    %out: !atir.tensor<128x128xf32>, %bias: !atir.tensor<128xf32>) {
  %tmp = "atir.buffer"() : () -> !atir.tensor<128x128xf32>
  %matmul = "atir.MatMul"(%tmp, %a, %b) <{
    left_transpose = false, output_transpose = false,
    right_transpose = false, withBias = false
  }> : (!atir.tensor<128x128xf32>, !atir.tensor<128x128xf32>,
        !atir.tensor<128x128xf32>) -> !atir.tensor<128x128xf32>
  %add_tmp = "atir.buffer"() : () -> !atir.tensor<128x128xf32>
  %add = "atir.Add"(%add_tmp, %matmul, %bias) <{
    do_relu = false, relu_limit = -1.0 : f32
  }> : (!atir.tensor<128x128xf32>, !atir.tensor<128x128xf32>,
        !atir.tensor<128xf32>) -> !atir.tensor<128x128xf32>
  %relu = "atir.Relu"(%out, %add) <{relu_limit = -1.0 : f32}> :
      (!atir.tensor<128x128xf32>, !atir.tensor<128x128xf32>) ->
      !atir.tensor<128x128xf32>
  return
}
