// RUN: env ANNC_GEMM=0 annc-opt %s -atir-gemm-epilogue-fusion -atir-op-fusion | FileCheck %s --check-prefix=OFF
// RUN: env ANNC_GEMM=1 annc-opt %s -atir-gemm-epilogue-fusion -atir-op-fusion | FileCheck %s --check-prefix=PLAIN
// RUN: env ANNC_GEMM=2 annc-opt %s -atir-gemm-epilogue-fusion -atir-op-fusion | FileCheck %s --check-prefix=EPILOGUE
// RUN: env ANNC_GEMM=3 annc-opt %s -atir-gemm-epilogue-fusion -atir-op-fusion | FileCheck %s --check-prefix=PLAIN
// RUN: env ANNC_GEMM=99 annc-opt %s -atir-gemm-epilogue-fusion -atir-op-fusion | FileCheck %s --check-prefix=EPILOGUE

// Explicit discovery supplies candidates in every case; OpFusion must consume
// them only in Epilogue/All. Prepack must still create a plain MatMul kernel.
// OFF-LABEL: func.func @main(
// OFF-NOT: call @fused_
// OFF: "atir.MatMul"
// OFF: "atir.Add"
// OFF: "atir.Relu"
// OFF-NOT: func.func private @fused_

// PLAIN-LABEL: func.func @main(
// PLAIN: call @fused_matmul_
// PLAIN: "atir.Add"
// PLAIN: "atir.Relu"
// PLAIN-LABEL: func.func private @fused_matmul_
// PLAIN-SAME: fusion.pattern = "matmul"
// PLAIN: "atir.MatMul"
// PLAIN-NOT: "atir.Add"
// PLAIN-NOT: "atir.Relu"

// EPILOGUE-LABEL: func.func @main(
// EPILOGUE: call @fused_matmul_add_relu_
// EPILOGUE-NOT: "atir.Add"
// EPILOGUE-NOT: "atir.Relu"
// EPILOGUE-LABEL: func.func private @fused_matmul_add_relu_
// EPILOGUE-SAME: fusion.pattern = "matmul_add_relu"
// EPILOGUE: "atir.MatMul"
// EPILOGUE: "atir.Add"
// EPILOGUE: "atir.Relu"

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
