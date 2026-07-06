// RUN: annc-opt --split-input-file %s -atir-block-fusion | FileCheck %s
// BlockFusion: FuseReluRewrite + MatmulWithBiasRewrite

// Relu 和 Add 进行融合
// CHECK-LABEL: func @fuse_relu_into_add
func.func @fuse_relu_into_add(
    %out: !atir.tensor<4x4xf32>,
    %a: !atir.tensor<4x4xf32>,
    %b: !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32> {
  %add = "atir.Add"(%out, %a, %b) <{do_relu = false, relu_limit = -1.0 : f32}> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32>
  // CHECK: "atir.Add"
  // CHECK-SAME: do_relu = true
  // CHECK-NOT: "atir.Relu"
  %relu = "atir.Relu"(%out, %add) <{relu_limit = 0.0 : f32}> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32>
  return %relu : !atir.tensor<4x4xf32>
}

// -----

// 无 Relu 的 Add 保持不变
// CHECK-LABEL: func @add_no_relu
func.func @add_no_relu(
    %out: !atir.tensor<4x4xf32>,
    %a: !atir.tensor<4x4xf32>,
    %b: !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32> {
  %add = "atir.Add"(%out, %a, %b) <{do_relu = false, relu_limit = -1.0 : f32}> : (!atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>, !atir.tensor<4x4xf32>) -> !atir.tensor<4x4xf32>
  // CHECK: "atir.Add"
  // CHECK-NOT: do_relu = true
  return %add : !atir.tensor<4x4xf32>
}

// -----

// 空函数
// CHECK-LABEL: func @empty
func.func @empty() {
  return
}