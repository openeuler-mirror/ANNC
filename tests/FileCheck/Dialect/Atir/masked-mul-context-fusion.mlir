// RUN: annc-opt --split-input-file %s --atir-op-fusion | FileCheck %s
// RUN: env ANNC_EP_FUSION_MASKED_MUL=0 annc-opt --split-input-file %s --atir-op-fusion | FileCheck %s --check-prefix=DISABLED
// DISABLED: atir.Mul
// DISABLED-NOT: custom.op_name = "MaskedMul
// DISABLED-NOT: func.func private @fused_masked_mul_
// FusedMaskedMul 站点（execution v2 形态）：一个站点就是「mask 链 + 链喂的全部
// Mul」这一整块子图，融合节点的输出 = 子图对外提供的全部值（每个 Mul 的结果，
// 加上链上仍被图外读的值）。Mul 之间夹着的逐元素路径（Sum → Div 常量）也一起
// 进站点、由 kernel 算掉——成员之间不允许夹组外节点。kernel 函数首参为执行
// 上下文、输出由 kernel 经上下文按槽位分配；主图区域只标记
// annc.fusion_materialized（AOT 裁剪丢弃主图）。
//
// kernel 算子名：实参个数（data + Sum 的其它输入）+ 是否多写一个 flat 输出。
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
// CHECK-SAME: custom.arg_names = ["arg0", "ids", "dataArgs", "rawMuls", "fedMuls", "argMap", "sumMask", "interiorKind", "divisorBits", "padding"]
// CHECK-SAME: custom.arg_ranks = [3, 2, -1, -1, -1, -1, -1, -1, -1, -1]
// CHECK-SAME: custom.attr.argMap = 0
// CHECK-SAME: custom.attr.dataArgs = 1
// CHECK-SAME: custom.attr.divisorBits = 1065353216
// CHECK-SAME: custom.attr.fedMuls = 0
// CHECK-SAME: custom.attr.interiorKind = 0
// CHECK-SAME: custom.attr.padding = 0
// CHECK-SAME: custom.attr.rawMuls = 1
// CHECK-SAME: custom.attr.sumMask = 0
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

// 负例：一条链喂出两个 fed Mul，但各自夹在**不同的** Sum/Div 路径后面（除数
// 2 与 3 也不同）。kernel 的 ABI 只有一份 sumMask/interiorKind/divisorBits，
// 且把同一个 d 写进全部 fed 输出——这种站点融合了会静默算错，匹配期必须拒绝。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_masked_mul
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%ids: !atir.tensor<1x4xi64, name = "ids">, %x: !atir.tensor<1x4x8xf32, name = "x">, %e1: !atir.tensor<1x4x8xf32, name = "e1">, %e2: !atir.tensor<1x4x8xf32, name = "e2">) -> (!atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x8xf32, name = "m_b">, !atir.tensor<1x4x8xf32, name = "m_c">) {
    %pad = atir.constant "public" @"pad" -> <i64, name = "pad", data = dense<0> : tensor<i64>>
    %true = atir.constant "private" @"Not/true" -> <i32, encoding = <"bool">, name = "Not/true", data = dense<1> : tensor<i32>>
    %eq = atir.Compare %ids, %pad {comparisonDirection = "EQ"} : (!atir.tensor<1x4xi64, name = "ids">, !atir.tensor<i64, name = "pad", data = dense<0> : tensor<i64>>) -> !atir.tensor<1x4xi32, encoding = <"bool">, name = "Equal:0">
    %not = atir.Compare %eq, %true {comparisonDirection = "NE"} : (!atir.tensor<1x4xi32, encoding = <"bool">, name = "Equal:0">, !atir.tensor<i32, encoding = <"bool">, name = "Not/true", data = dense<1> : tensor<i32>>) -> !atir.tensor<1x4xi32, encoding = <"bool">, name = "Not:0">
    %cast_buf = "atir.buffer"() : () -> !atir.tensor<1x4xf32, name = "Cast:0">
    %cast = atir.Cast %cast_buf, %not : <1x4xf32, name = "Cast:0">, <1x4xi32, encoding = <"bool">, name = "Not:0"> -> <1x4xf32, name = "Cast:0">
    %axes = atir.constant "private" @"Unsqueeze/axes" -> <1xi64, name = "Unsqueeze/axes", data = dense<2> : tensor<1xi64>>
    %unsq_buf = "atir.buffer"() : () -> !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">
    %unsq = atir.ExpandDims %unsq_buf, %cast, %axes : <1x4x1xf32, name = "Unsqueeze:0">, <1x4xf32, name = "Cast:0">, <1xi64, name = "Unsqueeze/axes", data = dense<2> : tensor<1xi64>> -> <1x4x1xf32, name = "Unsqueeze:0">
    %a_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "m_a">
    %m_a = "atir.Mul"(%a_buf, %x, %unsq) {metadata = {tf.name = "m_a", tf.op = "Mul"}} : (!atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x8xf32, name = "x">, !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">) -> !atir.tensor<1x4x8xf32, name = "m_a">
    %addn1_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "addn1">
    %addn1 = "atir.Add"(%addn1_buf, %m_a, %e1) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "addn1", tf.op = "Sum"}} : (!atir.tensor<1x4x8xf32, name = "addn1">, !atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x8xf32, name = "e1">) -> !atir.tensor<1x4x8xf32, name = "addn1">
    %two = atir.constant "public" @two -> <f32, name = "two", data = dense<2.000000e+00> : tensor<f32>>
    %div1_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "truediv1">
    %div1 = atir.RealDiv %div1_buf, %addn1, %two {metadata = {tf.name = "truediv1", tf.op = "Div"}} : (!atir.tensor<1x4x8xf32, name = "truediv1">, !atir.tensor<1x4x8xf32, name = "addn1">, !atir.tensor<f32, name = "two", data = dense<2.000000e+00> : tensor<f32>>) -> !atir.tensor<1x4x8xf32, name = "truediv1">
    %b_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "m_b">
    %m_b = "atir.Mul"(%b_buf, %div1, %unsq) {metadata = {tf.name = "m_b", tf.op = "Mul"}} : (!atir.tensor<1x4x8xf32, name = "m_b">, !atir.tensor<1x4x8xf32, name = "truediv1">, !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">) -> !atir.tensor<1x4x8xf32, name = "m_b">
    %addn2_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "addn2">
    %addn2 = "atir.Add"(%addn2_buf, %m_a, %e2) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "addn2", tf.op = "Sum"}} : (!atir.tensor<1x4x8xf32, name = "addn2">, !atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x8xf32, name = "e2">) -> !atir.tensor<1x4x8xf32, name = "addn2">
    %three = atir.constant "public" @three -> <f32, name = "three", data = dense<3.000000e+00> : tensor<f32>>
    %div2_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "truediv2">
    %div2 = atir.RealDiv %div2_buf, %addn2, %three {metadata = {tf.name = "truediv2", tf.op = "Div"}} : (!atir.tensor<1x4x8xf32, name = "truediv2">, !atir.tensor<1x4x8xf32, name = "addn2">, !atir.tensor<f32, name = "three", data = dense<3.000000e+00> : tensor<f32>>) -> !atir.tensor<1x4x8xf32, name = "truediv2">
    %c_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "m_c">
    %m_c = "atir.Mul"(%c_buf, %div2, %unsq) {metadata = {tf.name = "m_c", tf.op = "Mul"}} : (!atir.tensor<1x4x8xf32, name = "m_c">, !atir.tensor<1x4x8xf32, name = "truediv2">, !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">) -> !atir.tensor<1x4x8xf32, name = "m_c">
    return %m_a, %m_b, %m_c : !atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x8xf32, name = "m_b">, !atir.tensor<1x4x8xf32, name = "m_c">
  }
}

// -----

// 负例：无 Tile 的链喂两个末维不同的 Mul（8 与 4）——mask [d0,d1,1] 能广播到
// 任意末维，但 kernel 按同一个 n 读写全部实参与输出，这种站点融合了会在
// 运行时被实参同形检查拒掉（session 失败而不是回退），匹配期必须拒绝。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_masked_mul
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%ids: !atir.tensor<1x4xi64, name = "ids">, %x: !atir.tensor<1x4x8xf32, name = "x">, %y: !atir.tensor<1x4x4xf32, name = "y">) -> (!atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x4xf32, name = "m_b">) {
    %pad = atir.constant "public" @"pad" -> <i64, name = "pad", data = dense<0> : tensor<i64>>
    %true = atir.constant "private" @"Not/true" -> <i32, encoding = <"bool">, name = "Not/true", data = dense<1> : tensor<i32>>
    %eq = atir.Compare %ids, %pad {comparisonDirection = "EQ"} : (!atir.tensor<1x4xi64, name = "ids">, !atir.tensor<i64, name = "pad", data = dense<0> : tensor<i64>>) -> !atir.tensor<1x4xi32, encoding = <"bool">, name = "Equal:0">
    %not = atir.Compare %eq, %true {comparisonDirection = "NE"} : (!atir.tensor<1x4xi32, encoding = <"bool">, name = "Equal:0">, !atir.tensor<i32, encoding = <"bool">, name = "Not/true", data = dense<1> : tensor<i32>>) -> !atir.tensor<1x4xi32, encoding = <"bool">, name = "Not:0">
    %cast_buf = "atir.buffer"() : () -> !atir.tensor<1x4xf32, name = "Cast:0">
    %cast = atir.Cast %cast_buf, %not : <1x4xf32, name = "Cast:0">, <1x4xi32, encoding = <"bool">, name = "Not:0"> -> <1x4xf32, name = "Cast:0">
    %axes = atir.constant "private" @"Unsqueeze/axes" -> <1xi64, name = "Unsqueeze/axes", data = dense<2> : tensor<1xi64>>
    %unsq_buf = "atir.buffer"() : () -> !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">
    %unsq = atir.ExpandDims %unsq_buf, %cast, %axes : <1x4x1xf32, name = "Unsqueeze:0">, <1x4xf32, name = "Cast:0">, <1xi64, name = "Unsqueeze/axes", data = dense<2> : tensor<1xi64>> -> <1x4x1xf32, name = "Unsqueeze:0">
    %a_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "m_a">
    %m_a = "atir.Mul"(%a_buf, %x, %unsq) {metadata = {tf.name = "m_a", tf.op = "Mul"}} : (!atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x8xf32, name = "x">, !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">) -> !atir.tensor<1x4x8xf32, name = "m_a">
    %b_buf = "atir.buffer"() : () -> !atir.tensor<1x4x4xf32, name = "m_b">
    %m_b = "atir.Mul"(%b_buf, %y, %unsq) {metadata = {tf.name = "m_b", tf.op = "Mul"}} : (!atir.tensor<1x4x4xf32, name = "m_b">, !atir.tensor<1x4x4xf32, name = "y">, !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">) -> !atir.tensor<1x4x4xf32, name = "m_b">
    return %m_a, %m_b : !atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x4xf32, name = "m_b">
  }
}

// -----

// 一条链喂两个 Mul（带 Tile 广播）：一个站点吃掉链和两个 Mul，输出就是两个
// Mul 的结果（mask 不再物化）。Tile 的 multiples 只沿末维重复，且必须等于 data
// 的末维。
// CHECK-LABEL: func.func @main
// CHECK: atir.Mul
// CHECK-SAME: annc.fusion_materialized
// CHECK-LABEL: func.func private @fused_masked_mul_
// CHECK-SAME: abi = "annc_execution_v2"
// CHECK-SAME: kernel_arg_order = []
// CHECK-SAME: member_node_names = ["mul_b", "mul_a", "Tile:0", "Unsqueeze:0", "Cast:0", "Not:0", "Equal:0"]
// CHECK-SAME: outputs = [{dtype = "f32", rank = 3 : i64, role = "output", shape = [1, 4, 8], tf_name = "mul_b:0"}, {dtype = "f32", rank = 3 : i64, role = "output", shape = [1, 4, 8], tf_name = "mul_a:0"}]
// CHECK: atir.Customize
// CHECK-SAME: custom.arg_names = ["arg0", "arg1", "ids", "dataArgs", "rawMuls", "fedMuls", "argMap", "sumMask", "interiorKind", "divisorBits", "padding"]
// CHECK-SAME: custom.arg_ranks = [3, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1]
// CHECK-SAME: custom.attr.argMap = 16
// CHECK-SAME: custom.attr.dataArgs = 2
// CHECK-SAME: custom.attr.divisorBits = 1065353216
// CHECK-SAME: custom.attr.fedMuls = 0
// CHECK-SAME: custom.attr.interiorKind = 0
// CHECK-SAME: custom.attr.padding = -1
// CHECK-SAME: custom.attr.rawMuls = 2
// CHECK-SAME: custom.attr.sumMask = 0
// CHECK-SAME: custom.op_name = "MaskedMul2"
// CHECK-SAME: custom.result_names = ["output0", "output1"]
// CHECK-SAME: custom.result_ranks = [3, 3]
// 一条链只落一个站点：此后到下一用例之间不应再出现第二个 Customize。
// CHECK-NOT: atir.Customize
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

// 正例：Mul 之间夹着逐元素路径（model1 的真实形状）——两个 raw Mul 的结果先进
// 一个 Sum（多输入 atir.Add，还带一个外部输入），再 RealDiv 除常量，最后又喂给
// 第三个 Mul。这些节点一起进站点（否则两个成员之间夹着组外节点，ORT 会作废整次
// 提交），由 kernel 一并算掉；输出就是三个 Mul 的结果。
// CHECK-LABEL: func.func @main
// CHECK: func.func private @fused_masked_mul_
// CHECK-SAME: member_node_names = ["m_b", "m_a", "m_c", "addn", "truediv", "Unsqueeze:0", "Cast:0", "Not:0", "Equal:0"]
// CHECK-SAME: outputs = [{dtype = "f32", rank = 3 : i64, role = "output", shape = [1, 4, 8], tf_name = "m_b:0"}, {dtype = "f32", rank = 3 : i64, role = "output", shape = [1, 4, 8], tf_name = "m_a:0"}, {dtype = "f32", rank = 3 : i64, role = "output", shape = [1, 4, 8], tf_name = "m_c:0"}]
// CHECK: atir.Customize
// CHECK-SAME: custom.arg_names = ["arg0", "arg1", "arg2", "ids", "dataArgs", "rawMuls", "fedMuls", "argMap", "sumMask", "interiorKind", "divisorBits", "padding"]
// CHECK-SAME: custom.arg_ranks = [3, 3, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1]
// CHECK-SAME: custom.attr.argMap = 16
// CHECK-SAME: custom.attr.dataArgs = 2
// CHECK-SAME: custom.attr.divisorBits = 1073741824
// CHECK-SAME: custom.attr.fedMuls = 1
// CHECK-SAME: custom.attr.interiorKind = 0
// CHECK-SAME: custom.attr.rawMuls = 2
// CHECK-SAME: custom.attr.sumMask = 3
// CHECK-SAME: custom.op_name = "MaskedMul3"
// CHECK-SAME: custom.result_ranks = [3, 3, 3]
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%ids: !atir.tensor<1x4xi64, name = "ids">, %x: !atir.tensor<1x4x8xf32, name = "x">, %z: !atir.tensor<1x4x8xf32, name = "z">, %e: !atir.tensor<1x4x8xf32, name = "e">) -> (!atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x8xf32, name = "m_b">, !atir.tensor<1x4x8xf32, name = "m_c">) {
    %pad = atir.constant "public" @"pad" -> <i64, name = "pad", data = dense<0> : tensor<i64>>
    %true = atir.constant "private" @"Not/true" -> <i32, encoding = <"bool">, name = "Not/true", data = dense<1> : tensor<i32>>
    %eq = atir.Compare %ids, %pad {comparisonDirection = "EQ"} : (!atir.tensor<1x4xi64, name = "ids">, !atir.tensor<i64, name = "pad", data = dense<0> : tensor<i64>>) -> !atir.tensor<1x4xi32, encoding = <"bool">, name = "Equal:0">
    %not = atir.Compare %eq, %true {comparisonDirection = "NE"} : (!atir.tensor<1x4xi32, encoding = <"bool">, name = "Equal:0">, !atir.tensor<i32, encoding = <"bool">, name = "Not/true", data = dense<1> : tensor<i32>>) -> !atir.tensor<1x4xi32, encoding = <"bool">, name = "Not:0">
    %cast_buf = "atir.buffer"() : () -> !atir.tensor<1x4xf32, name = "Cast:0">
    %cast = atir.Cast %cast_buf, %not : <1x4xf32, name = "Cast:0">, <1x4xi32, encoding = <"bool">, name = "Not:0"> -> <1x4xf32, name = "Cast:0">
    %axes = atir.constant "private" @"Unsqueeze/axes" -> <1xi64, name = "Unsqueeze/axes", data = dense<2> : tensor<1xi64>>
    %unsq_buf = "atir.buffer"() : () -> !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">
    %unsq = atir.ExpandDims %unsq_buf, %cast, %axes : <1x4x1xf32, name = "Unsqueeze:0">, <1x4xf32, name = "Cast:0">, <1xi64, name = "Unsqueeze/axes", data = dense<2> : tensor<1xi64>> -> <1x4x1xf32, name = "Unsqueeze:0">
    %a_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "m_a">
    %m_a = "atir.Mul"(%a_buf, %x, %unsq) {metadata = {tf.name = "m_a", tf.op = "Mul"}} : (!atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x8xf32, name = "x">, !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">) -> !atir.tensor<1x4x8xf32, name = "m_a">
    %b_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "m_b">
    %m_b = "atir.Mul"(%b_buf, %z, %unsq) {metadata = {tf.name = "m_b", tf.op = "Mul"}} : (!atir.tensor<1x4x8xf32, name = "m_b">, !atir.tensor<1x4x8xf32, name = "z">, !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">) -> !atir.tensor<1x4x8xf32, name = "m_b">
    %addn_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "addn">
    %addn = "atir.Add"(%addn_buf, %m_a, %m_b, %e) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "addn", tf.op = "Sum"}} : (!atir.tensor<1x4x8xf32, name = "addn">, !atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x8xf32, name = "m_b">, !atir.tensor<1x4x8xf32, name = "e">) -> !atir.tensor<1x4x8xf32, name = "addn">
    %two = atir.constant "public" @two -> <f32, name = "two", data = dense<2.000000e+00> : tensor<f32>>
    %div_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "truediv">
    %div = atir.RealDiv %div_buf, %addn, %two {metadata = {tf.name = "truediv", tf.op = "Div"}} : (!atir.tensor<1x4x8xf32, name = "truediv">, !atir.tensor<1x4x8xf32, name = "addn">, !atir.tensor<f32, name = "two", data = dense<2.000000e+00> : tensor<f32>>) -> !atir.tensor<1x4x8xf32, name = "truediv">
    %c_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "m_c">
    %m_c = "atir.Mul"(%c_buf, %div, %unsq) {metadata = {tf.name = "m_c", tf.op = "Mul"}} : (!atir.tensor<1x4x8xf32, name = "m_c">, !atir.tensor<1x4x8xf32, name = "truediv">, !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">) -> !atir.tensor<1x4x8xf32, name = "m_c">
    return %m_a, %m_b, %m_c : !atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x8xf32, name = "m_b">, !atir.tensor<1x4x8xf32, name = "m_c">
  }
}

// -----

// 负例：两个 Mul 之间夹着一个**表达不了**的节点（这里是 Relu，既不是 Sum/Div
// 那条路径，也不是可以做逐元素乘积的 Mul）——中间节点不能一起吃进 kernel，被
// 它喂的那个 Mul 也就不进站点：整块只产出 m_a 那一个站点。
// CHECK-LABEL: func.func @main
// CHECK: func.func private @fused_masked_mul_
// CHECK-SAME: member_node_names = ["m_a", "Unsqueeze:0", "Cast:0", "Not:0", "Equal:0"]
// CHECK-NOT: fused_masked_mul
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%ids: !atir.tensor<1x4xi64, name = "ids">, %x: !atir.tensor<1x4x8xf32, name = "x">, %y: !atir.tensor<1x4x8xf32, name = "y">) -> (!atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x8xf32, name = "m_b">) {
    %pad = atir.constant "public" @"pad" -> <i64, name = "pad", data = dense<0> : tensor<i64>>
    %true = atir.constant "private" @"Not/true" -> <i32, encoding = <"bool">, name = "Not/true", data = dense<1> : tensor<i32>>
    %eq = atir.Compare %ids, %pad {comparisonDirection = "EQ"} : (!atir.tensor<1x4xi64, name = "ids">, !atir.tensor<i64, name = "pad", data = dense<0> : tensor<i64>>) -> !atir.tensor<1x4xi32, encoding = <"bool">, name = "Equal:0">
    %not = atir.Compare %eq, %true {comparisonDirection = "NE"} : (!atir.tensor<1x4xi32, encoding = <"bool">, name = "Equal:0">, !atir.tensor<i32, encoding = <"bool">, name = "Not/true", data = dense<1> : tensor<i32>>) -> !atir.tensor<1x4xi32, encoding = <"bool">, name = "Not:0">
    %cast_buf = "atir.buffer"() : () -> !atir.tensor<1x4xf32, name = "Cast:0">
    %cast = atir.Cast %cast_buf, %not : <1x4xf32, name = "Cast:0">, <1x4xi32, encoding = <"bool">, name = "Not:0"> -> <1x4xf32, name = "Cast:0">
    %axes = atir.constant "private" @"Unsqueeze/axes" -> <1xi64, name = "Unsqueeze/axes", data = dense<2> : tensor<1xi64>>
    %unsq_buf = "atir.buffer"() : () -> !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">
    %unsq = atir.ExpandDims %unsq_buf, %cast, %axes : <1x4x1xf32, name = "Unsqueeze:0">, <1x4xf32, name = "Cast:0">, <1xi64, name = "Unsqueeze/axes", data = dense<2> : tensor<1xi64>> -> <1x4x1xf32, name = "Unsqueeze:0">
    %a_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "m_a">
    %m_a = "atir.Mul"(%a_buf, %x, %unsq) : (!atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x8xf32, name = "x">, !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">) -> !atir.tensor<1x4x8xf32, name = "m_a">
    %mid_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "mid">
    %mid = "atir.Relu"(%mid_buf, %m_a) <{relu_limit = -1.0 : f32}> : (!atir.tensor<1x4x8xf32, name = "mid">, !atir.tensor<1x4x8xf32, name = "m_a">) -> !atir.tensor<1x4x8xf32, name = "mid">
    %b_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "m_b">
    %m_b = "atir.Mul"(%b_buf, %mid, %y) : (!atir.tensor<1x4x8xf32, name = "m_b">, !atir.tensor<1x4x8xf32, name = "mid">, !atir.tensor<1x4x8xf32, name = "y">) -> !atir.tensor<1x4x8xf32, name = "m_b">
    return %m_a, %m_b : !atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x8xf32, name = "m_b">
  }
}

// -----

// 正例：两个 Mul 之间夹着一个逐元素乘积（model1 的 NotEqual 链：两个 masked 结
// 果相乘，积再乘 mask）。中间那个 Mul 一起吃进 kernel（interiorKind = 1），站
// 点把自己的每个对外值都产出：m_a / m_b / m_c。
// CHECK-LABEL: func.func @main
// CHECK: func.func private @fused_masked_mul_
// CHECK: atir.Customize
// CHECK-SAME: custom.arg_names = ["arg0", "arg1", "ids", "dataArgs", "rawMuls", "fedMuls", "argMap", "sumMask", "interiorKind", "divisorBits", "padding"]
// CHECK-SAME: custom.attr.dataArgs = 2
// CHECK-SAME: custom.attr.divisorBits = 1065353216
// CHECK-SAME: custom.attr.fedMuls = 1
// CHECK-SAME: custom.attr.interiorKind = 1
// CHECK-SAME: custom.attr.rawMuls = 2
// CHECK-SAME: custom.attr.sumMask = 3
// CHECK-SAME: custom.op_name = "MaskedMul2"
// CHECK-SAME: custom.result_names = ["output0", "output1", "output2"]
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%ids: !atir.tensor<1x4xi64, name = "ids">, %x: !atir.tensor<1x4x8xf32, name = "x">, %z: !atir.tensor<1x4x8xf32, name = "z">) -> (!atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x8xf32, name = "m_b">, !atir.tensor<1x4x8xf32, name = "m_c">) {
    %pad = atir.constant "public" @"pad" -> <i64, name = "pad", data = dense<0> : tensor<i64>>
    %true = atir.constant "private" @"Not/true" -> <i32, encoding = <"bool">, name = "Not/true", data = dense<1> : tensor<i32>>
    %eq = atir.Compare %ids, %pad {comparisonDirection = "EQ"} : (!atir.tensor<1x4xi64, name = "ids">, !atir.tensor<i64, name = "pad", data = dense<0> : tensor<i64>>) -> !atir.tensor<1x4xi32, encoding = <"bool">, name = "Equal:0">
    %not = atir.Compare %eq, %true {comparisonDirection = "NE"} : (!atir.tensor<1x4xi32, encoding = <"bool">, name = "Equal:0">, !atir.tensor<i32, encoding = <"bool">, name = "Not/true", data = dense<1> : tensor<i32>>) -> !atir.tensor<1x4xi32, encoding = <"bool">, name = "Not:0">
    %cast_buf = "atir.buffer"() : () -> !atir.tensor<1x4xf32, name = "Cast:0">
    %cast = atir.Cast %cast_buf, %not : <1x4xf32, name = "Cast:0">, <1x4xi32, encoding = <"bool">, name = "Not:0"> -> <1x4xf32, name = "Cast:0">
    %axes = atir.constant "private" @"Unsqueeze/axes" -> <1xi64, name = "Unsqueeze/axes", data = dense<2> : tensor<1xi64>>
    %unsq_buf = "atir.buffer"() : () -> !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">
    %unsq = atir.ExpandDims %unsq_buf, %cast, %axes : <1x4x1xf32, name = "Unsqueeze:0">, <1x4xf32, name = "Cast:0">, <1xi64, name = "Unsqueeze/axes", data = dense<2> : tensor<1xi64>> -> <1x4x1xf32, name = "Unsqueeze:0">
    %a_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "m_a">
    %m_a = "atir.Mul"(%a_buf, %x, %unsq) {metadata = {tf.name = "m_a", tf.op = "Mul"}} : (!atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x8xf32, name = "x">, !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">) -> !atir.tensor<1x4x8xf32, name = "m_a">
    %b_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "m_b">
    %m_b = "atir.Mul"(%b_buf, %z, %unsq) {metadata = {tf.name = "m_b", tf.op = "Mul"}} : (!atir.tensor<1x4x8xf32, name = "m_b">, !atir.tensor<1x4x8xf32, name = "z">, !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">) -> !atir.tensor<1x4x8xf32, name = "m_b">
    %mid_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "mid">
    %mid = "atir.Mul"(%mid_buf, %m_a, %m_b) {metadata = {tf.name = "mid", tf.op = "Mul"}} : (!atir.tensor<1x4x8xf32, name = "mid">, !atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x8xf32, name = "m_b">) -> !atir.tensor<1x4x8xf32, name = "mid">
    %c_buf = "atir.buffer"() : () -> !atir.tensor<1x4x8xf32, name = "m_c">
    %m_c = "atir.Mul"(%c_buf, %mid, %unsq) {metadata = {tf.name = "m_c", tf.op = "Mul"}} : (!atir.tensor<1x4x8xf32, name = "m_c">, !atir.tensor<1x4x8xf32, name = "mid">, !atir.tensor<1x4x1xf32, name = "Unsqueeze:0">) -> !atir.tensor<1x4x8xf32, name = "m_c">
    return %m_a, %m_b, %m_c : !atir.tensor<1x4x8xf32, name = "m_a">, !atir.tensor<1x4x8xf32, name = "m_b">, !atir.tensor<1x4x8xf32, name = "m_c">
  }
}

// -----

// 正例：链的 Cast 输出还被链外的形状类消费者读（model3 的 Shape/Expand 复用
// 同一个 mask）。站点把它作为额外输出产出（MaskedMul1Flat）：子图有多少对外
// 的值就产出多少。
// CHECK-LABEL: func.func @main
// CHECK: func.func private @fused_masked_mul_
// CHECK-SAME: member_node_names = ["mul_out", "Unsqueeze:0", "Cast:0", "Not:0", "Equal:0"]
// CHECK-SAME: outputs = [{dtype = "f32", rank = 3 : i64, role = "output", shape = [1, 4, 8], tf_name = "mul_out:0"}, {dtype = "f32", rank = 2 : i64, role = "output", shape = [1, 4], tf_name = "Cast:0"}]
// CHECK: atir.Customize
// CHECK-SAME: custom.arg_names = ["arg0", "ids", "dataArgs", "rawMuls", "fedMuls", "argMap", "sumMask", "interiorKind", "divisorBits", "padding"]
// CHECK-SAME: custom.arg_ranks = [3, 2, -1, -1, -1, -1, -1, -1, -1, -1]
// CHECK-SAME: custom.op_name = "MaskedMul1Flat"
// CHECK-SAME: custom.result_names = ["output0", "flat"]
// CHECK-SAME: custom.result_ranks = [3, 2]
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%ids: !atir.tensor<1x4xi64, name = "ids">, %data: !atir.tensor<1x4x8xf32, name = "data">) -> (!atir.tensor<1x4x8xf32, name = "mul_out">, !atir.tensor<2xi64, name = "mask_shape">) {
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
    %shape = atir.Shape %cast : (!atir.tensor<1x4xf32, name = "Cast:0">) -> !atir.tensor<2xi64, name = "mask_shape">
    return %mul, %shape : !atir.tensor<1x4x8xf32, name = "mul_out">, !atir.tensor<2xi64, name = "mask_shape">
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
