// RUN: annc-opt --split-input-file %s --atir-op-fusion | FileCheck %s
// FusedMaskedMul 站点（execution v2 形态）：链 + 链的每个 Mul 融成一个站点，
// kernel 函数首参为执行上下文、fusion.metadata 的 abi 为 annc_execution_v2、
// kernel_arg_order 为空（输出由 kernel 经上下文按槽位分配），成员名单记录整块；
// 主图区域不再重写，只标记 annc.fusion_materialized（AOT 裁剪丢弃主图函数）。
// 元数（链的 Mul 数）决定 kernel 算子名与符号：MaskedMul1..MaskedMul4。
//
// CHECK-LABEL: func.func @main
// CHECK: atir.Mul
// CHECK-SAME: annc.fusion_materialized
// CHECK-LABEL: func.func private @fused_masked_mul_
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: annc.execution_mode = "aot"
// CHECK-SAME: annc.kernel
// CHECK-SAME: abi = "annc_execution_v2"
// CHECK-SAME: kernel_arg_order = []
// CHECK-SAME: member_node_names = ["mul_out", "Unsqueeze:0", "Cast:0", "Not:0", "Equal:0"]
// CHECK-SAME: outputs = [{dtype = "f32", rank = 3 : i64, role = "output", shape = [1, 4, 8], tf_name = "mul_out:0"}]
// CHECK: atir.Customize
// CHECK-SAME: custom.arg_names = ["data0", "ids", "padding"]
// CHECK-SAME: custom.arg_ranks = [3, 2, -1]
// CHECK-SAME: custom.attr.padding = 0
// CHECK-SAME: custom.op_name = "MaskedMul1"
// CHECK-SAME: custom.result_names = ["output0"]
// CHECK-SAME: custom.result_ranks = [3]
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%ids: !atir.tensor<1x4xi64, name = "ids">, %data: !atir.tensor<1x4x8xf32, name = "data">) -> !atir.tensor<1x4x8xf32, name = "mul_out"> {
    %pad = atir.constant "public" @"pad" -> <i64, name = "pad", data = dense<0> : tensor<i64>>
    %true = atir.constant "private" @"Not/true" -> <i32, encoding = <"bool">, name = "Not/true", data = dense<1> : tensor<i32>>
    %eq = atir.Compare %ids, %pad {comparisonDirection = "EQ"} : (!atir.tensor<1x4xi64, name = "ids">, !atir.tensor<i64, name = "pad", data = dense<0> : tensor<i64>>) -> !atir.tensor<1x4xi32, encoding = <"bool">, name = "Equal:0">
    %not = atir.Compare %eq, %true {comparisonDirection = "NE"} : (!atir.tensor<1x4xi32, encoding = <"bool">, name = "Equal:0">, !atir.tensor<i32, encoding = <"bool">, name = "Not/true", data = dense<1> : tensor<i32>>) -> !atir.tensor<1x4xi32, encoding = <"bool">, name = "Not:0">
    %cast_buf = "atir.buffer"() : () -> !atir.tensor<1x4xf32, name = "Cast:0">
    %cast = atir.Cast %cast_buf, %not : <1x4xf32, name = "Cast:0">, <1x4xi32, encoding = <"bool">, name = "Not:0"> -> <1x4xf32, name = "Cast:0">
    %axes = atir.constant "private" @"Unsqueeze/axes" -> <1xi64, name = "Unsqueeze/axes", data = dense<2> : tensor<1xi64>>
    %unsq_buf = "atir.buffer"() : () -> !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">
    %unsq = atir.ExpandDims %unsq_buf, %cast, %axes : <1x4x1xf32, name = "Unsqueeze:0">, <1x4xf32, name = "Cast:0">, <1xi64, name = "Unsqueeze/axes", data = dense<2> : tensor<1xi64>> -> <1x4x1xf32, name = "Unsqueeze:0">
    %mul_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "mul_out">
    %mul = "atir.Mul"(%mul_buf, %data, %unsq) : (!atir.tensor<1x4x8xf32, name = "mul_out">, !atir.tensor<1x4x8xf32, name = "data">, !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">) -> !atir.tensor<1x4x8xf32, name = "mul_out">
    return %mul : !atir.tensor<1x4x8xf32, name = "mul_out">
  }
}

// -----

// 一条链喂两个 Mul（带 Tile 广播）：融合为一个 MaskedMul2 站点，两个输出都
// 在站点内（链死了，两个 Mul 必须一起走）。Tile 的 multiples 只沿末维重复，
// 且必须等于 data 的末维。
// CHECK-LABEL: func.func @main
// CHECK: atir.Mul
// CHECK-SAME: annc.fusion_materialized
// CHECK-LABEL: func.func private @fused_masked_mul_
// CHECK-SAME: abi = "annc_execution_v2"
// CHECK-SAME: kernel_arg_order = []
// CHECK-SAME: member_node_names = ["mul_b", "mul_a", "Tile:0", "Unsqueeze:0", "Cast:0", "Not:0", "Equal:0"]
// CHECK: atir.Customize
// CHECK-SAME: custom.arg_names = ["data0", "data1", "ids", "padding"]
// CHECK-SAME: custom.arg_ranks = [3, 3, 2, -1]
// CHECK-SAME: custom.attr.padding = -1
// CHECK-SAME: custom.op_name = "MaskedMul2"
// CHECK-SAME: custom.result_names = ["output0", "output1"]
// CHECK-SAME: custom.result_ranks = [3, 3]
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%ids: !atir.tensor<1x4xi64, name = "ids">, %a: !atir.tensor<1x4x8xf32, name = "a">, %b: !atir.tensor<1x4x8xf32, name = "b">) -> (!atir.tensor<1x4x8xf32, name = "mul_a">, !atir.tensor<1x4x8xf32, name = "mul_b">) {
    %pad = atir.constant "public" @"pad" -> <i64, name = "pad", data = dense<-1> : tensor<i64>>
    %true = atir.constant "private" @"Not/true" -> <i32, encoding = <"bool">, name = "Not/true", data = dense<1> : tensor<i32>>
    %eq = atir.Compare %ids, %pad {comparisonDirection = "EQ"} : (!atir.tensor<1x4xi64, name = "ids">, !atir.tensor<i64, name = "pad", data = dense<-1> : tensor<i64>>) -> !atir.tensor<1x4xi32, encoding = <"bool">, name = "Equal:0">
    %not = atir.Compare %eq, %true {comparisonDirection = "NE"} : (!atir.tensor<1x4xi32, encoding = <"bool">, name = "Equal:0">, !atir.tensor<i32, encoding = <"bool">, name = "Not/true", data = dense<1> : tensor<i32>>) -> !atir.tensor<1x4xi32, encoding = <"bool">, name = "Not:0">
    %cast_buf = "atir.buffer"() : () -> !atir.tensor<1x4xf32, name = "Cast:0">
    %cast = atir.Cast %cast_buf, %not : <1x4xf32, name = "Cast:0">, <1x4xi32, encoding = <"bool">, name = "Not:0"> -> <1x4xf32, name = "Cast:0">
    %axes = atir.constant "private" @"Unsqueeze/axes" -> <1xi64, name = "Unsqueeze/axes", data = dense<2> : tensor<1xi64>>
    %unsq_buf = "atir.buffer"() : () -> !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">
    %unsq = atir.ExpandDims %unsq_buf, %cast, %axes : <1x4x1xf32, name = "Unsqueeze:0">, <1x4xf32, name = "Cast:0">, <1xi64, name = "Unsqueeze/axes", data = dense<2> : tensor<1xi64>> -> <1x4x1xf32, name = "Unsqueeze:0">
    %mult = atir.constant "private" @"Tile/multiples" -> <3xi32, name = "Tile/multiples", data = dense<[1, 1, 8]> : tensor<3xi32>>
    %tile_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "Tile:0">
    %tile = "atir.Tile"(%tile_buf, %unsq, %mult) : (!atir.tensor<1x4x8xf32, name = "Tile:0">, !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">, !atir.tensor<3xi32, name = "Tile/multiples", data = dense<[1, 1, 8]> : tensor<3xi32>>) -> !atir.tensor<1x4x8xf32, name = "Tile:0">
    %mul_a_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "mul_a">
    %mul_a = "atir.Mul"(%mul_a_buf, %a, %tile) : (!atir.tensor<1x4x8xf32, name = "mul_a">, !atir.tensor<1x4x8xf32, name = "a">, !atir.tensor<1x4x8xf32, name = "Tile:0">) -> !atir.tensor<1x4x8xf32, name = "mul_a">
    %mul_b_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "mul_b">
    %mul_b = "atir.Mul"(%mul_b_buf, %b, %tile) : (!atir.tensor<1x4x8xf32, name = "mul_b">, !atir.tensor<1x4x8xf32, name = "b">, !atir.tensor<1x4x8xf32, name = "Tile:0">) -> !atir.tensor<1x4x8xf32, name = "mul_b">
    return %mul_a, %mul_b : !atir.tensor<1x4x8xf32, name = "mul_a">, !atir.tensor<1x4x8xf32, name = "mul_b">
  }
}

// -----

// 负例：同一块但模块没有 ONNX 前端标记（annc.contract_version）——站点只服务
// ONNX 宿主（kernel 期望 ONNX 执行上下文），TF 侧不产出。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_masked_mul
module attributes {module.state = "atir"} {
  func.func @main(%ids: !atir.tensor<1x4xi64, name = "ids">, %data: !atir.tensor<1x4x8xf32, name = "data">) -> !atir.tensor<1x4x8xf32, name = "mul_out"> {
    %pad = atir.constant "public" @"pad" -> <i64, name = "pad", data = dense<0> : tensor<i64>>
    %true = atir.constant "private" @"Not/true" -> <i32, encoding = <"bool">, name = "Not/true", data = dense<1> : tensor<i32>>
    %eq = atir.Compare %ids, %pad {comparisonDirection = "EQ"} : (!atir.tensor<1x4xi64, name = "ids">, !atir.tensor<i64, name = "pad", data = dense<0> : tensor<i64>>) -> !atir.tensor<1x4xi32, encoding = <"bool">, name = "Equal:0">
    %not = atir.Compare %eq, %true {comparisonDirection = "NE"} : (!atir.tensor<1x4xi32, encoding = <"bool">, name = "Equal:0">, !atir.tensor<i32, encoding = <"bool">, name = "Not/true", data = dense<1> : tensor<i32>>) -> !atir.tensor<1x4xi32, encoding = <"bool">, name = "Not:0">
    %cast_buf = "atir.buffer"() : () -> !atir.tensor<1x4xf32, name = "Cast:0">
    %cast = atir.Cast %cast_buf, %not : <1x4xf32, name = "Cast:0">, <1x4xi32, encoding = <"bool">, name = "Not:0"> -> <1x4xf32, name = "Cast:0">
    %axes = atir.constant "private" @"Unsqueeze/axes" -> <1xi64, name = "Unsqueeze/axes", data = dense<2> : tensor<1xi64>>
    %unsq_buf = "atir.buffer"() : () -> !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">
    %unsq = atir.ExpandDims %unsq_buf, %cast, %axes : <1x4x1xf32, name = "Unsqueeze:0">, <1x4xf32, name = "Cast:0">, <1xi64, name = "Unsqueeze/axes", data = dense<2> : tensor<1xi64>> -> <1x4x1xf32, name = "Unsqueeze:0">
    %mul_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "mul_out">
    %mul = "atir.Mul"(%mul_buf, %data, %unsq) : (!atir.tensor<1x4x8xf32, name = "mul_out">, !atir.tensor<1x4x8xf32, name = "data">, !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">) -> !atir.tensor<1x4x8xf32, name = "mul_out">
    return %mul : !atir.tensor<1x4x8xf32, name = "mul_out">
  }
}
