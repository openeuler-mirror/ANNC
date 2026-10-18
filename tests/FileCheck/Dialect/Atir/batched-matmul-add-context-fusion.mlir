// RUN: annc-opt --split-input-file %s --atir-op-fusion | FileCheck %s
// RUN: env ANNC_EP_FUSION_BATCHED_MATMUL_ADD=0 annc-opt --split-input-file %s --atir-op-fusion | FileCheck %s --check-prefix=DISABLED
// DISABLED: atir.MatMul
// DISABLED-NOT: custom.op_name = "BatchedMatMulAdd
// DISABLED-NOT: func.func private @fused_batched_matmul_add_
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

// 正例：批量权重 [2, K, N] 与逐 batch bias [2, 1, N]，batch 维严格等于
// 输出 batch。ORT patch 支持该形态，本线也必须融合并在 kernel 中按
// batch 选取 weight/bias。
// CHECK-LABEL: func.func @main
// CHECK: func.func private @fused_batched_matmul_add_
// CHECK: custom.arg_names = ["data", "weight", "bias"]
// CHECK-SAME: custom.op_name = "BatchedMatMulAdd"
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<2x2x3xf32, name = "x">) -> !atir.tensor<2x2x4xf32, name = "y"> {
    %w = atir.constant "public" @"w" -> <2x3x4xf32, name = "w", data = dense<1.000000> : tensor<2x3x4xf32>>
    %b = atir.constant "public" @"b" -> <2x1x4xf32, name = "b", data = dense<0.000000> : tensor<2x1x4xf32>>
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<2x2x4xf32, name = "mm">
    %mm = "atir.MatMul"(%mm_buf, %x, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> {metadata = {tf.name = "n_mm", tf.op = "MatMul"}} : (!atir.tensor<2x2x4xf32, name = "mm">, !atir.tensor<2x2x3xf32, name = "x">, !atir.tensor<2x3x4xf32, name = "w", data = dense<1.000000> : tensor<2x3x4xf32>>) -> !atir.tensor<2x2x4xf32, name = "mm">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<2x2x4xf32, name = "y">
    %y = "atir.Add"(%y_buf, %mm, %b) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add", tf.op = "Add", tf.output_tensors = ["y"]}} : (!atir.tensor<2x2x4xf32, name = "y">, !atir.tensor<2x2x4xf32, name = "mm">, !atir.tensor<2x1x4xf32, name = "b", data = dense<0.000000> : tensor<2x1x4xf32>>) -> !atir.tensor<2x2x4xf32, name = "y">
    return %y : !atir.tensor<2x2x4xf32, name = "y">
  }
}

// -----

// 负例：bias 的 batch 维与输出/权重不一致，必须不融（禁止静默取第 0 个
// batch 或越界读取）。
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

// -----

// 正例：MatMul 输入带常量 pre-affine（X*scale+offset）。patch 会把 scale
// 折进 W、把 offset·W 折进 bias；本线应生成同形 fused site，而不是把
// Mul/Add 留在动态输入前。
// CHECK-LABEL: func.func @main
// CHECK-NOT: tf_name = "affined"
// CHECK: func.func private @fused_batched_matmul_add_
// CHECK: custom.arg_names = ["data", "weight", "bias"]
// CHECK-SAME: custom.op_name = "BatchedMatMulAdd"
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<1x2x3xf32, name = "x">) -> !atir.tensor<1x2x4xf32, name = "y"> {
    %scale = atir.constant "public" @scale -> <3xf32, name = "scale", data = dense<[1.250000e+00, -5.000000e-01, 2.000000e+00]> : tensor<3xf32>>
    %offset = atir.constant "public" @offset -> <3xf32, name = "offset", data = dense<[1.000000e-01, -2.000000e-01, 3.000000e-01]> : tensor<3xf32>>
    %w = atir.constant "public" @w -> <1x3x4xf32, name = "w", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]]]> : tensor<1x3x4xf32>>
    %b = atir.constant "public" @b -> <1x1x4xf32, name = "b", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]]> : tensor<1x1x4xf32>>
    %scaled_buf = "atir.buffer"() : () -> !atir.tensor<1x2x3xf32, name = "scaled">
    %scaled = atir.Mul %scaled_buf, %x, %scale {metadata = {tf.input.0 = "x", tf.input.1 = "scale", tf.name = "n_mul", tf.op = "Mul"}} : (!atir.tensor<1x2x3xf32, name = "scaled">, !atir.tensor<1x2x3xf32, name = "x">, !atir.tensor<3xf32, name = "scale", data = dense<[1.250000e+00, -5.000000e-01, 2.000000e+00]> : tensor<3xf32>>) -> !atir.tensor<1x2x3xf32, name = "scaled">
    %affined_buf = "atir.buffer"() : () -> !atir.tensor<1x2x3xf32, name = "affined">
    %affined = "atir.Add"(%affined_buf, %scaled, %offset) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.input.0 = "scaled", tf.input.1 = "offset", tf.name = "n_affine", tf.op = "Add"}} : (!atir.tensor<1x2x3xf32, name = "affined">, !atir.tensor<1x2x3xf32, name = "scaled">, !atir.tensor<3xf32, name = "offset", data = dense<[1.000000e-01, -2.000000e-01, 3.000000e-01]> : tensor<3xf32>>) -> !atir.tensor<1x2x3xf32, name = "affined">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<1x2x4xf32, name = "mm">
    %mm = "atir.MatMul"(%mm_buf, %affined, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> {metadata = {tf.name = "n_mm", tf.op = "MatMul"}} : (!atir.tensor<1x2x4xf32, name = "mm">, !atir.tensor<1x2x3xf32, name = "affined">, !atir.tensor<1x3x4xf32, name = "w", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]]]> : tensor<1x3x4xf32>>) -> !atir.tensor<1x2x4xf32, name = "mm">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<1x2x4xf32, name = "y">
    %y = "atir.Add"(%y_buf, %mm, %b) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add", tf.op = "Add", tf.output_tensors = ["y"]}} : (!atir.tensor<1x2x4xf32, name = "y">, !atir.tensor<1x2x4xf32, name = "mm">, !atir.tensor<1x1x4xf32, name = "b", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]]> : tensor<1x1x4xf32>>) -> !atir.tensor<1x2x4xf32, name = "y">
    return %y : !atir.tensor<1x2x4xf32, name = "y">
  }
}

// -----

// 正例：pre-affine 常量取「标量 / 单元素」形状——scale 是 rank-0 标量、offset 是
// [1] 单元素，覆盖 fold 的 affineIndex 里 shape.empty()→0 与 width==1→0 两个分支
// （上一个正例只覆盖 rank-1 [K]）。折叠值逐元素断言：foldedWeight = scale·W、
// foldedBias = b + Σ_k offset·W（此例即 b + 0.5·列和）。
// CHECK-LABEL: func.func @main
// CHECK-NOT: tf_name = "s_affined"
// CHECK: func.func private @fused_batched_matmul_add_
// （dense 的 `[[` 与 FileCheck 变量语法冲突，折叠值取无括号片段断言。）
// CHECK: atir.constant "private" @pre_affine_weight
// CHECK-SAME: 2.000000e+00, 4.000000e+00, 6.000000e+00
// CHECK-SAME: 1.600000e+01, 1.800000e+01, 2.000000e+01, 2.200000e+01
// CHECK: atir.constant "private" @pre_affine_bias
// CHECK-SAME: 6.000000e+00, 8.500000e+00, 1.100000e+01, 1.350000e+01
// CHECK: custom.op_name = "BatchedMatMulAdd"
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<1x2x3xf32, name = "x">) -> !atir.tensor<1x2x4xf32, name = "y"> {
    %scale = atir.constant "public" @s_scale -> <f32, name = "s_scale", data = dense<2.0> : tensor<f32>>
    %offset = atir.constant "public" @s_offset -> <1xf32, name = "s_offset", data = dense<5.000000e-01> : tensor<1xf32>>
    %w = atir.constant "public" @w -> <1x3x4xf32, name = "w", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]]]> : tensor<1x3x4xf32>>
    %b = atir.constant "public" @b -> <1x1x4xf32, name = "b", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]]> : tensor<1x1x4xf32>>
    %scaled_buf = "atir.buffer"() : () -> !atir.tensor<1x2x3xf32, name = "s_scaled">
    %scaled = atir.Mul %scaled_buf, %x, %scale {metadata = {tf.input.0 = "x", tf.input.1 = "s_scale", tf.name = "s_mul", tf.op = "Mul"}} : (!atir.tensor<1x2x3xf32, name = "s_scaled">, !atir.tensor<1x2x3xf32, name = "x">, !atir.tensor<f32, name = "s_scale", data = dense<2.0> : tensor<f32>>) -> !atir.tensor<1x2x3xf32, name = "s_scaled">
    %affined_buf = "atir.buffer"() : () -> !atir.tensor<1x2x3xf32, name = "s_affined">
    %affined = "atir.Add"(%affined_buf, %scaled, %offset) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.input.0 = "s_scaled", tf.input.1 = "s_offset", tf.name = "s_affine", tf.op = "Add"}} : (!atir.tensor<1x2x3xf32, name = "s_affined">, !atir.tensor<1x2x3xf32, name = "s_scaled">, !atir.tensor<1xf32, name = "s_offset", data = dense<5.000000e-01> : tensor<1xf32>>) -> !atir.tensor<1x2x3xf32, name = "s_affined">
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<1x2x4xf32, name = "s_mm">
    %mm = "atir.MatMul"(%mm_buf, %affined, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> {metadata = {tf.name = "s_mm", tf.op = "MatMul"}} : (!atir.tensor<1x2x4xf32, name = "s_mm">, !atir.tensor<1x2x3xf32, name = "s_affined">, !atir.tensor<1x3x4xf32, name = "w", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]]]> : tensor<1x3x4xf32>>) -> !atir.tensor<1x2x4xf32, name = "s_mm">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<1x2x4xf32, name = "y">
    %y = "atir.Add"(%y_buf, %mm, %b) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "s_add", tf.op = "Add", tf.output_tensors = ["y"]}} : (!atir.tensor<1x2x4xf32, name = "y">, !atir.tensor<1x2x4xf32, name = "s_mm">, !atir.tensor<1x1x4xf32, name = "b", data = dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]]]> : tensor<1x1x4xf32>>) -> !atir.tensor<1x2x4xf32, name = "y">
    return %y : !atir.tensor<1x2x4xf32, name = "y">
  }
}

// -----

// 正例：rank-4 形态（model3 的 domain_2 站点）——data [2, 1, ?, 32] ×
// W [2, 1, 32, 2] + bias [2, 1, 1, 2] -> [2, 1, ?, 2]。M 是动态维，由 kernel
// 从 memref 描述符读；两条批量维按 numpy 广播（data/W 逐维为 1 或输出维），
// bias 的批量前缀必须与输出逐维一致（与 patch HasSupportedShapes 同口径）。
// CHECK-LABEL: func.func @main
// CHECK: atir.MatMul
// CHECK-SAME: annc.fusion_materialized
// CHECK-LABEL: func.func private @fused_batched_matmul_add_
// CHECK-SAME: abi = "annc_execution_v2"
// CHECK-SAME: kernel_arg_order = []
// CHECK-SAME: member_node_names = ["n_add", "n_mm"]
// CHECK: atir.Customize
// CHECK-SAME: custom.arg_names = ["data", "weight", "bias"]
// CHECK-SAME: custom.arg_ranks = [4, 4, 4]
// CHECK-SAME: custom.op_name = "BatchedMatMulAdd4"
// CHECK-SAME: custom.result_names = ["output"]
// CHECK-SAME: custom.result_ranks = [4]
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<2x1x?x32xf32, name = "x">) -> !atir.tensor<2x1x?x2xf32, name = "y"> {
    %w = atir.constant "public" @"w" -> <2x1x32x2xf32, name = "w", data = dense<1.000000> : tensor<2x1x32x2xf32>>
    %b = atir.constant "public" @"b" -> <2x1x1x2xf32, name = "b", data = dense<0.000000> : tensor<2x1x1x2xf32>>
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<2x1x?x2xf32, name = "mm">
    %mm = "atir.MatMul"(%mm_buf, %x, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> {metadata = {tf.name = "n_mm", tf.op = "MatMul"}} : (!atir.tensor<2x1x?x2xf32, name = "mm">, !atir.tensor<2x1x?x32xf32, name = "x">, !atir.tensor<2x1x32x2xf32, name = "w", data = dense<1.000000> : tensor<2x1x32x2xf32>>) -> !atir.tensor<2x1x?x2xf32, name = "mm">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<2x1x?x2xf32, name = "y">
    %y = "atir.Add"(%y_buf, %mm, %b) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add", tf.op = "Add", tf.output_tensors = ["y"]}} : (!atir.tensor<2x1x?x2xf32, name = "y">, !atir.tensor<2x1x?x2xf32, name = "mm">, !atir.tensor<2x1x1x2xf32, name = "b", data = dense<0.000000> : tensor<2x1x1x2xf32>>) -> !atir.tensor<2x1x?x2xf32, name = "y">
    return %y : !atir.tensor<2x1x?x2xf32, name = "y">
  }
}

// -----

// 负例 3：rank-4 权重与 bias 的批量维不匹配（W 的第 0 维 3 既不等于 1 也不
// 等于输出的 2；bias 的第 1 维 2 也不等于输出的 1）。必须不融，禁止按错误
// 的批量索引读取 W/bias。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_batched_matmul_add
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<2x1x?x32xf32, name = "x">) -> !atir.tensor<2x1x?x2xf32, name = "y"> {
    %w = atir.constant "public" @"w" -> <3x1x32x2xf32, name = "w", data = dense<1.000000> : tensor<3x1x32x2xf32>>
    %b = atir.constant "public" @"b" -> <2x2x1x2xf32, name = "b", data = dense<0.000000> : tensor<2x2x1x2xf32>>
    %mm_buf = "atir.buffer"() : () -> !atir.tensor<2x1x?x2xf32, name = "mm">
    %mm = "atir.MatMul"(%mm_buf, %x, %w) <{do_relu = false, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f32, right_transpose = false, withBias = false}> {metadata = {tf.name = "n_mm", tf.op = "MatMul"}} : (!atir.tensor<2x1x?x2xf32, name = "mm">, !atir.tensor<2x1x?x32xf32, name = "x">, !atir.tensor<3x1x32x2xf32, name = "w", data = dense<1.000000> : tensor<3x1x32x2xf32>>) -> !atir.tensor<2x1x?x2xf32, name = "mm">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<2x1x?x2xf32, name = "y">
    %y = "atir.Add"(%y_buf, %mm, %b) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add", tf.op = "Add", tf.output_tensors = ["y"]}} : (!atir.tensor<2x1x?x2xf32, name = "y">, !atir.tensor<2x1x?x2xf32, name = "mm">, !atir.tensor<2x2x1x2xf32, name = "b", data = dense<0.000000> : tensor<2x2x1x2xf32>>) -> !atir.tensor<2x1x?x2xf32, name = "y">
    return %y : !atir.tensor<2x1x?x2xf32, name = "y">
  }
}
