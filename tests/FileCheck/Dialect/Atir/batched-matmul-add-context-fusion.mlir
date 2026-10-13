// RUN: annc-opt --split-input-file %s --atir-op-fusion | FileCheck %s
// FusedBatchedMatMulAdd 站点（execution v2 形态）：rank-3 MatMul + 广播 bias
// Add 融成一个站点，kernel 函数首参为执行上下文、fusion.metadata 的 abi 为
// annc_execution_v2、kernel_arg_order 为空（输出由 kernel 经上下文按槽位 0
// 分配）；主图区域只标记 annc.fusion_materialized。
//
// CHECK-LABEL: func.func @main
// CHECK: atir.MatMul
// CHECK-SAME: annc.fusion_materialized
// CHECK-LABEL: func.func private @fused_batched_matmul_add_
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: annc.execution_mode = "aot"
// CHECK-SAME: annc.kernel
// CHECK-SAME: abi = "annc_execution_v2"
// CHECK-SAME: args = [{dtype = "f32", rank = 3 : i64, role = "dynamic", shape = [1, 2, 3], tf_name = "x"}, {dtype = "f32", rank = 3 : i64, role = "fixed", shape = [1, 3, 4], tf_name = "w"}, {dtype = "f32", rank = 3 : i64, role = "fixed", shape = [1, 1, 4], tf_name = "b"}]
// CHECK-SAME: kernel_arg_order = []
// CHECK-SAME: member_node_names = ["n_add", "n_mm"]
// CHECK-SAME: outputs = [{dtype = "f32", rank = 3 : i64, role = "output", shape = [1, 2, 4], tf_name = "y:0"}]
// CHECK: atir.Customize
// CHECK-SAME: custom.arg_names = ["data", "weight", "bias"]
// CHECK-SAME: custom.arg_ranks = [3, 3, 3]
// CHECK-SAME: custom.op_name = "BatchedMatMulAdd"
// CHECK-SAME: custom.result_names = ["output"]
// CHECK-SAME: custom.result_ranks = [3]
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<1x2x3xf32, name = "x">) -> !atir.tensor<1x2x4xf32, name = "y"> {
    %w = atir.constant "public" @"w" -> <1x3x4xf32, name = "w", data = dense<1.000000> : tensor<1x3x4xf32>>
    %b = atir.constant "public" @"b" -> <1x1x4xf32, name = "b", data = dense<0.000000> : tensor<1x1x4xf32>>
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<1x2x4xf32, name = "mm">
    %mm = "atir.MatMul"(%mm_buf, %x, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> {metadata = {tf.name = "n_mm", tf.op = "MatMul"}} : (!atir.tensor<1x2x4xf32, name = "mm">, !atir.tensor<1x2x3xf32, name = "x">, !atir.tensor<1x3x4xf32, name = "w", data = dense<1.000000> : tensor<1x3x4xf32>>) -> !atir.tensor<1x2x4xf32, name = "mm">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<1x2x4xf32, name = "y">
    %y = "atir.Add"(%y_buf, %mm, %b) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add", tf.op = "Add", tf.output_tensors = ["y"]}} : (!atir.tensor<1x2x4xf32, name = "y">, !atir.tensor<1x2x4xf32, name = "mm">, !atir.tensor<1x1x4xf32, name = "b", data = dense<0.000000> : tensor<1x1x4xf32>>) -> !atir.tensor<1x2x4xf32, name = "y">
    return %y : !atir.tensor<1x2x4xf32, name = "y">
  }
}

// -----

// 负例：权重的前导维不是广播 1（[2, 3, 4]）——kernel 只实现 batch-1 广播的
// 形态，这类站点必须不融（ORT patch 的 rank-4/批量权重形态属后续覆盖）。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_batched_matmul_add
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<2x2x3xf32, name = "x">) -> !atir.tensor<2x2x4xf32, name = "y"> {
    %w = atir.constant "public" @"w" -> <2x3x4xf32, name = "w", data = dense<1.000000> : tensor<2x3x4xf32>>
    %b = atir.constant "public" @"b" -> <1x1x4xf32, name = "b", data = dense<0.000000> : tensor<1x1x4xf32>>
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<2x2x4xf32, name = "mm">
    %mm = "atir.MatMul"(%mm_buf, %x, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> {metadata = {tf.name = "n_mm", tf.op = "MatMul"}} : (!atir.tensor<2x2x4xf32, name = "mm">, !atir.tensor<2x2x3xf32, name = "x">, !atir.tensor<2x3x4xf32, name = "w", data = dense<1.000000> : tensor<2x3x4xf32>>) -> !atir.tensor<2x2x4xf32, name = "mm">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<2x2x4xf32, name = "y">
    %y = "atir.Add"(%y_buf, %mm, %b) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add", tf.op = "Add", tf.output_tensors = ["y"]}} : (!atir.tensor<2x2x4xf32, name = "y">, !atir.tensor<2x2x4xf32, name = "mm">, !atir.tensor<1x1x4xf32, name = "b", data = dense<0.000000> : tensor<1x1x4xf32>>) -> !atir.tensor<2x2x4xf32, name = "y">
    return %y : !atir.tensor<2x2x4xf32, name = "y">
  }
}

// -----

// 负例 2：同一块但模块没有 ONNX 前端标记（annc.contract_version）。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_batched_matmul_add
module attributes {module.state = "atir"} {
  func.func @main(%x: !atir.tensor<1x2x3xf32, name = "x">) -> !atir.tensor<1x2x4xf32, name = "y"> {
    %w = atir.constant "public" @"w" -> <1x3x4xf32, name = "w", data = dense<1.000000> : tensor<1x3x4xf32>>
    %b = atir.constant "public" @"b" -> <1x1x4xf32, name = "b", data = dense<0.000000> : tensor<1x1x4xf32>>
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<1x2x4xf32, name = "mm">
    %mm = "atir.MatMul"(%mm_buf, %x, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> {metadata = {tf.name = "n_mm", tf.op = "MatMul"}} : (!atir.tensor<1x2x4xf32, name = "mm">, !atir.tensor<1x2x3xf32, name = "x">, !atir.tensor<1x3x4xf32, name = "w", data = dense<1.000000> : tensor<1x3x4xf32>>) -> !atir.tensor<1x2x4xf32, name = "mm">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<1x2x4xf32, name = "y">
    %y = "atir.Add"(%y_buf, %mm, %b) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add", tf.op = "Add", tf.output_tensors = ["y"]}} : (!atir.tensor<1x2x4xf32, name = "y">, !atir.tensor<1x2x4xf32, name = "mm">, !atir.tensor<1x1x4xf32, name = "b", data = dense<0.000000> : tensor<1x1x4xf32>>) -> !atir.tensor<1x2x4xf32, name = "y">
    return %y : !atir.tensor<1x2x4xf32, name = "y">
  }
}
