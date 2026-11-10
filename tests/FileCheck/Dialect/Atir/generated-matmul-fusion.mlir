// RUN: env ANNC_GEMM=1 ANNC_FAST_CODEGEN_DISABLE_CUSTOM_OPS=MatMul annc-opt --split-input-file %s --atir-gemm-epilogue-fusion --atir-op-fusion | FileCheck %s --check-prefix=FUSION --implicit-check-not=annc_context_v3 --implicit-check-not=atir.Customize
// RUN: env ANNC_GEMM=2 annc-opt --split-input-file %s --atir-gemm-epilogue-fusion --atir-op-fusion | FileCheck %s --check-prefix=FUSION --implicit-check-not=annc_context_v3 --implicit-check-not=atir.Customize
// RUN: env ANNC_GEMM=3 annc-opt --split-input-file %s --atir-gemm-epilogue-fusion --atir-op-fusion | FileCheck %s --check-prefix=FUSION --implicit-check-not=annc_context_v3 --implicit-check-not=atir.Customize
// RUN: env ANNC_GEMM=99 annc-opt --split-input-file %s --atir-gemm-epilogue-fusion --atir-op-fusion | FileCheck %s --check-prefix=FUSION --implicit-check-not=annc_context_v3 --implicit-check-not=atir.Customize
// RUN: annc-asm --split-input-file %s --atir-fast-codegen='enable-kdnn=true enable-custom-ops=MatMul' | env ANNC_GEMM=2 annc-opt --split-input-file --atir-op-fusion | FileCheck %s --check-prefix=FUSION --implicit-check-not=annc_context_v3 --implicit-check-not=atir.Customize
// RUN: env ANNC_GEMM=2 annc-opt --split-input-file %s --atir-op-fusion | annc-asm --split-input-file --atir-prune-func=execution-mode=jit --atir-fast-codegen --lower-context-kernel-entries --convert-atir-to-linalg | FileCheck %s --check-prefix=LOWER --implicit-check-not=ANNCKernel_ContextV3_MatMul --implicit-check-not=atir.Customize
// RUN: env ANNC_GEMM=2 annc-opt --split-input-file %s --atir-op-fusion | annc-asm --split-input-file --atir-prune-func=execution-mode=jit --atir-fast-codegen='enable-kdnn=true enable-custom-ops=MatMul' --lower-context-kernel-entries --convert-atir-to-linalg | FileCheck %s --check-prefix=LOWER --implicit-check-not=ANNCKernel_ContextV3_MatMul --implicit-check-not=atir.Customize

// Context builtin availability must not change a plain GEMM's implementation.
// Mode 1 disables explicit KDNN MatMul outlining to exercise the generated
// fallback even in a KDNN-enabled build. The remaining modes do not use that
// Basic-only rule. There is no epilogue to discover in either test case.
// FastCodegen must preserve ordinary GEMM both before and after OpFusion,
// including when KDNN is enabled and MatMul is explicitly allowlisted.
// FUSION-LABEL: func.func @main(
// FUSION: call @fused_matmul_
// FUSION: return
// FUSION-LABEL: func.func private @fused_matmul_
// FUSION-SAME: abi = "mlir_ciface"
// FUSION-SAME: execution_mode = "jit"
// FUSION: "atir.MatMul"
// FUSION: return
// LOWER-LABEL: func.func private @fused_matmul_
// LOWER: linalg.matmul
// LOWER: return
func.func @main(
    %lhs: !atir.tensor<64x128xf32>,
    %rhs: !atir.tensor<128x64xf32>,
    %out: !atir.tensor<64x64xf32>) -> !atir.tensor<64x64xf32> {
  %result = "atir.MatMul"(%out, %lhs, %rhs) <{
    left_transpose = false, right_transpose = false,
    output_transpose = false, withBias = false
  }> : (!atir.tensor<64x64xf32>, !atir.tensor<64x128xf32>,
        !atir.tensor<128x64xf32>) -> !atir.tensor<64x64xf32>
  return %result : !atir.tensor<64x64xf32>
}

// -----

// Dynamic batch must retain the generated memref entry as well.
// FUSION-LABEL: func.func @main(
// FUSION: call @fused_matmul_
// FUSION: return
// FUSION-LABEL: func.func private @fused_matmul_
// FUSION-SAME: abi = "mlir_ciface"
// FUSION-SAME: execution_mode = "jit"
// FUSION: "atir.MatMul"
// FUSION: return
// LOWER-LABEL: func.func private @fused_matmul_
// LOWER: linalg.matmul
// LOWER: return
func.func @main(
    %lhs: !atir.tensor<?x128xf32>,
    %rhs: !atir.tensor<128x64xf32>,
    %out: !atir.tensor<?x64xf32>) -> !atir.tensor<?x64xf32> {
  %result = "atir.MatMul"(%out, %lhs, %rhs) <{
    left_transpose = false, right_transpose = false,
    output_transpose = false, withBias = false
  }> : (!atir.tensor<?x64xf32>, !atir.tensor<?x128xf32>,
        !atir.tensor<128x64xf32>) -> !atir.tensor<?x64xf32>
  return %result : !atir.tensor<?x64xf32>
}
