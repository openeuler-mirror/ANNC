// RUN: annc-opt --split-input-file %s --atir-op-fusion | FileCheck %s
// RUN: env ANNC_EP_FUSION_LAYER_NORM=0 annc-opt --split-input-file %s --atir-op-fusion | FileCheck %s --check-prefix=DISABLED
// DISABLED: atir.ReduceMean
// DISABLED-NOT: custom.op_name = "LayerNorm
// DISABLED-NOT: func.func private @fused_layer_norm_
// FusedLayerNorm 站点（execution v2 形态）：RM-LN 链的 11 个中间算子加末端
// Add 全部融成一个站点，kernel 函数首参为执行上下文、fusion.metadata 的 abi 为
// annc_execution_v2、kernel_arg_order 为空（输出由 kernel 经上下文按槽位 0
// 分配）；主图区域只标记 annc.fusion_materialized。
//
// CHECK-LABEL: func.func @main
// CHECK: atir.ReduceMean
// CHECK-SAME: annc.fusion_materialized
// CHECK-LABEL: func.func private @fused_layer_norm_
// CHECK-SAME: !llvm.ptr
// CHECK-SAME: annc.execution_mode = "aot"
// CHECK-SAME: annc.kernel
// CHECK-SAME: abi = "annc_execution_v2"
// CHECK-SAME: args = [{dtype = "f32", rank = 2 : i64, role = "dynamic", shape = [2, 4], tf_name = "x"}, {dtype = "f32", rank = 1 : i64, role = "fixed", shape = [4], tf_name = "gamma"}, {dtype = "f32", rank = 1 : i64, role = "fixed", shape = [4], tf_name = "beta"}, {dtype = "f32", rank = 0 : i64, role = "fixed", shape = [], tf_name = "eps"}]
// CHECK-SAME: kernel_arg_order = []
// CHECK-SAME: member_node_names = ["n_out", "n_xg", "n_bias", "n_mg", "n_g", "n_recip", "n_sqrt", "n_add1", "n_var", "n_sq", "n_sub1", "n_mean"]
// CHECK-SAME: outputs = [{dtype = "f32", rank = 2 : i64, role = "output", shape = [2, 4], tf_name = "y:0"}]
// CHECK: atir.Customize
// CHECK-SAME: custom.arg_names = ["input", "gamma", "beta", "eps"]
// CHECK-SAME: custom.arg_ranks = [2, 1, 1, 0]
// CHECK-SAME: custom.op_name = "LayerNorm"
// CHECK-SAME: custom.result_names = ["output"]
// CHECK-SAME: custom.result_ranks = [2]
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<2x4xf32, name = "x">) -> !atir.tensor<2x4xf32, name = "y"> {
    %gamma = atir.constant "public" @"gamma" -> <4xf32, name = "gamma", data = dense<1.000000> : tensor<4xf32>>
    %beta = atir.constant "public" @"beta" -> <4xf32, name = "beta", data = dense<0.000000> : tensor<4xf32>>
    %eps = atir.constant "public" @"eps" -> <f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<1> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = true, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <2x1xf32, name = "mean">, <2x4xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<1> : tensor<1xi64>> -> <2x1xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<2x4xf32, name = "d">, !atir.tensor<2x4xf32, name = "x">, !atir.tensor<2x1xf32, name = "mean">) -> !atir.tensor<2x4xf32, name = "d">
    %sq_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "sq">
    %sq = atir.Mul %sq_buf, %d, %d {metadata = {tf.name = "n_sq", tf.op = "Mul"}} : (!atir.tensor<2x4xf32, name = "sq">, !atir.tensor<2x4xf32, name = "d">, !atir.tensor<2x4xf32, name = "d">) -> !atir.tensor<2x4xf32, name = "sq">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %sq, %axes {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <2x1xf32, name = "var">, <2x4xf32, name = "sq">, <1xi64, name = "n_mean/axes", data = dense<1> : tensor<1xi64>> -> <2x1xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %var, %eps) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<2x1xf32, name = "var_eps">, !atir.tensor<2x1xf32, name = "var">, !atir.tensor<f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>) -> !atir.tensor<2x1xf32, name = "var_eps">
    %std_buf = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "std">
    %std = atir.Sqrt %std_buf, %ve {metadata = {tf.name = "n_sqrt", tf.op = "Sqrt"}} : <2x1xf32, name = "std">, <2x1xf32, name = "var_eps"> -> <2x1xf32, name = "std">
    %rs_buf = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "rsqrt">
    %rs = atir.Reciprocal %rs_buf, %std {metadata = {tf.name = "n_recip", tf.op = "Reciprocal"}} : <2x1xf32, name = "rsqrt">, <2x1xf32, name = "std"> -> <2x1xf32, name = "rsqrt">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "g">
    %g = atir.Mul %g_buf, %rs, %gamma {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<2x4xf32, name = "g">, !atir.tensor<2x1xf32, name = "rsqrt">, !atir.tensor<4xf32, name = "gamma", data = dense<1.000000> : tensor<4xf32>>) -> !atir.tensor<2x4xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %x, %g {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<2x4xf32, name = "xg">, !atir.tensor<2x4xf32, name = "x">, !atir.tensor<2x4xf32, name = "g">) -> !atir.tensor<2x4xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<2x4xf32, name = "mg">, !atir.tensor<2x1xf32, name = "mean">, !atir.tensor<2x4xf32, name = "g">) -> !atir.tensor<2x4xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<2x4xf32, name = "bias">, !atir.tensor<4xf32, name = "beta", data = dense<0.000000> : tensor<4xf32>>, !atir.tensor<2x4xf32, name = "mg">) -> !atir.tensor<2x4xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<2x4xf32, name = "y">, !atir.tensor<2x4xf32, name = "xg">, !atir.tensor<2x4xf32, name = "bias">) -> !atir.tensor<2x4xf32, name = "y">
    return %y : !atir.tensor<2x4xf32, name = "y">
  }
}

// -----

// 负例：mean 多了一个站点外的消费者（共享矩）——融合后这个消费者就没有值可
// 读，站点必须不成立（kernel 单输出，提供不了共享的 mean）。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_layer_norm
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<2x4xf32, name = "x">) -> (!atir.tensor<2x4xf32, name = "y">, !atir.tensor<2x1xf32, name = "mean_extra">) {
    %gamma = atir.constant "public" @"gamma" -> <4xf32, name = "gamma", data = dense<1.000000> : tensor<4xf32>>
    %beta = atir.constant "public" @"beta" -> <4xf32, name = "beta", data = dense<0.000000> : tensor<4xf32>>
    %eps = atir.constant "public" @"eps" -> <f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<1> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = true, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <2x1xf32, name = "mean">, <2x4xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<1> : tensor<1xi64>> -> <2x1xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<2x4xf32, name = "d">, !atir.tensor<2x4xf32, name = "x">, !atir.tensor<2x1xf32, name = "mean">) -> !atir.tensor<2x4xf32, name = "d">
    %sq_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "sq">
    %sq = atir.Mul %sq_buf, %d, %d {metadata = {tf.name = "n_sq", tf.op = "Mul"}} : (!atir.tensor<2x4xf32, name = "sq">, !atir.tensor<2x4xf32, name = "d">, !atir.tensor<2x4xf32, name = "d">) -> !atir.tensor<2x4xf32, name = "sq">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %sq, %axes {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <2x1xf32, name = "var">, <2x4xf32, name = "sq">, <1xi64, name = "n_mean/axes", data = dense<1> : tensor<1xi64>> -> <2x1xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %var, %eps) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<2x1xf32, name = "var_eps">, !atir.tensor<2x1xf32, name = "var">, !atir.tensor<f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>) -> !atir.tensor<2x1xf32, name = "var_eps">
    %std_buf = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "std">
    %std = atir.Sqrt %std_buf, %ve {metadata = {tf.name = "n_sqrt", tf.op = "Sqrt"}} : <2x1xf32, name = "std">, <2x1xf32, name = "var_eps"> -> <2x1xf32, name = "std">
    %rs_buf = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "rsqrt">
    %rs = atir.Reciprocal %rs_buf, %std {metadata = {tf.name = "n_recip", tf.op = "Reciprocal"}} : <2x1xf32, name = "rsqrt">, <2x1xf32, name = "std"> -> <2x1xf32, name = "rsqrt">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "g">
    %g = atir.Mul %g_buf, %rs, %gamma {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<2x4xf32, name = "g">, !atir.tensor<2x1xf32, name = "rsqrt">, !atir.tensor<4xf32, name = "gamma", data = dense<1.000000> : tensor<4xf32>>) -> !atir.tensor<2x4xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %x, %g {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<2x4xf32, name = "xg">, !atir.tensor<2x4xf32, name = "x">, !atir.tensor<2x4xf32, name = "g">) -> !atir.tensor<2x4xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<2x4xf32, name = "mg">, !atir.tensor<2x1xf32, name = "mean">, !atir.tensor<2x4xf32, name = "g">) -> !atir.tensor<2x4xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<2x4xf32, name = "bias">, !atir.tensor<4xf32, name = "beta", data = dense<0.000000> : tensor<4xf32>>, !atir.tensor<2x4xf32, name = "mg">) -> !atir.tensor<2x4xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<2x4xf32, name = "y">, !atir.tensor<2x4xf32, name = "xg">, !atir.tensor<2x4xf32, name = "bias">) -> !atir.tensor<2x4xf32, name = "y">
    %extra_buf = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "mean_extra">
    %extra = atir.Sqrt %extra_buf, %mean {metadata = {tf.name = "n_extra", tf.op = "Sqrt"}} : <2x1xf32, name = "mean_extra">, <2x1xf32, name = "mean"> -> <2x1xf32, name = "mean_extra">
    return %y, %extra : !atir.tensor<2x4xf32, name = "y">, !atir.tensor<2x1xf32, name = "mean_extra">
  }
}

// -----

// 负例 2：同一块但模块没有 ONNX 前端标记（annc.contract_version）——站点只
// 服务 ONNX 宿主，TF 侧不产出。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_layer_norm
module attributes {module.state = "atir"} {
  func.func @main(%x: !atir.tensor<2x4xf32, name = "x">) -> !atir.tensor<2x4xf32, name = "y"> {
    %gamma = atir.constant "public" @"gamma" -> <4xf32, name = "gamma", data = dense<1.000000> : tensor<4xf32>>
    %beta = atir.constant "public" @"beta" -> <4xf32, name = "beta", data = dense<0.000000> : tensor<4xf32>>
    %eps = atir.constant "public" @"eps" -> <f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<1> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = true, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <2x1xf32, name = "mean">, <2x4xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<1> : tensor<1xi64>> -> <2x1xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<2x4xf32, name = "d">, !atir.tensor<2x4xf32, name = "x">, !atir.tensor<2x1xf32, name = "mean">) -> !atir.tensor<2x4xf32, name = "d">
    %sq_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "sq">
    %sq = atir.Mul %sq_buf, %d, %d {metadata = {tf.name = "n_sq", tf.op = "Mul"}} : (!atir.tensor<2x4xf32, name = "sq">, !atir.tensor<2x4xf32, name = "d">, !atir.tensor<2x4xf32, name = "d">) -> !atir.tensor<2x4xf32, name = "sq">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %sq, %axes {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <2x1xf32, name = "var">, <2x4xf32, name = "sq">, <1xi64, name = "n_mean/axes", data = dense<1> : tensor<1xi64>> -> <2x1xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %var, %eps) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<2x1xf32, name = "var_eps">, !atir.tensor<2x1xf32, name = "var">, !atir.tensor<f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>) -> !atir.tensor<2x1xf32, name = "var_eps">
    %std_buf = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "std">
    %std = atir.Sqrt %std_buf, %ve {metadata = {tf.name = "n_sqrt", tf.op = "Sqrt"}} : <2x1xf32, name = "std">, <2x1xf32, name = "var_eps"> -> <2x1xf32, name = "std">
    %rs_buf = "atir.buffer"() : () -> !atir.tensor<2x1xf32, name = "rsqrt">
    %rs = atir.Reciprocal %rs_buf, %std {metadata = {tf.name = "n_recip", tf.op = "Reciprocal"}} : <2x1xf32, name = "rsqrt">, <2x1xf32, name = "std"> -> <2x1xf32, name = "rsqrt">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "g">
    %g = atir.Mul %g_buf, %rs, %gamma {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<2x4xf32, name = "g">, !atir.tensor<2x1xf32, name = "rsqrt">, !atir.tensor<4xf32, name = "gamma", data = dense<1.000000> : tensor<4xf32>>) -> !atir.tensor<2x4xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %x, %g {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<2x4xf32, name = "xg">, !atir.tensor<2x4xf32, name = "x">, !atir.tensor<2x4xf32, name = "g">) -> !atir.tensor<2x4xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<2x4xf32, name = "mg">, !atir.tensor<2x1xf32, name = "mean">, !atir.tensor<2x4xf32, name = "g">) -> !atir.tensor<2x4xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<2x4xf32, name = "bias">, !atir.tensor<4xf32, name = "beta", data = dense<0.000000> : tensor<4xf32>>, !atir.tensor<2x4xf32, name = "mg">) -> !atir.tensor<2x4xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<2x4xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<2x4xf32, name = "y">, !atir.tensor<2x4xf32, name = "xg">, !atir.tensor<2x4xf32, name = "bias">) -> !atir.tensor<2x4xf32, name = "y">
    return %y : !atir.tensor<2x4xf32, name = "y">
  }
}

// -----

// 正例 2：x 为 3x8，归约轴常量写成 -1——isLastAxisMean 的负轴归一化分支，
// 与正例 1（轴 1、2x4）互补；站点形态不变。
// CHECK-LABEL: func.func @main
// CHECK: atir.ReduceMean
// CHECK-SAME: annc.fusion_materialized
// CHECK-LABEL: func.func private @fused_layer_norm_
// CHECK-SAME: abi = "annc_execution_v2"
// CHECK-SAME: args = [{dtype = "f32", rank = 2 : i64, role = "dynamic", shape = [3, 8], tf_name = "x"}
// CHECK: atir.Customize
// CHECK-SAME: custom.arg_names = ["input", "gamma", "beta", "eps"]
// CHECK-SAME: custom.arg_ranks = [2, 1, 1, 0]
// CHECK-SAME: custom.op_name = "LayerNorm"
// CHECK-SAME: custom.result_ranks = [2]
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<3x8xf32, name = "x">) -> !atir.tensor<3x8xf32, name = "y"> {
    %gamma = atir.constant "public" @"gamma" -> <8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>
    %beta = atir.constant "public" @"beta" -> <8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>
    %eps = atir.constant "public" @"eps" -> <f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = true, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <3x1xf32, name = "mean">, <3x8xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x1xf32, name = "mean">) -> !atir.tensor<3x8xf32, name = "d">
    %sq_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "sq">
    %sq = atir.Mul %sq_buf, %d, %d {metadata = {tf.name = "n_sq", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "sq">, !atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "d">) -> !atir.tensor<3x8xf32, name = "sq">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %sq, %axes {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <3x1xf32, name = "var">, <3x8xf32, name = "sq">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %var, %eps) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<3x1xf32, name = "var_eps">, !atir.tensor<3x1xf32, name = "var">, !atir.tensor<f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>) -> !atir.tensor<3x1xf32, name = "var_eps">
    %std_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "std">
    %std = atir.Sqrt %std_buf, %ve {metadata = {tf.name = "n_sqrt", tf.op = "Sqrt"}} : <3x1xf32, name = "std">, <3x1xf32, name = "var_eps"> -> <3x1xf32, name = "std">
    %rs_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "rsqrt">
    %rs = atir.Reciprocal %rs_buf, %std {metadata = {tf.name = "n_recip", tf.op = "Reciprocal"}} : <3x1xf32, name = "rsqrt">, <3x1xf32, name = "std"> -> <3x1xf32, name = "rsqrt">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "g">
    %g = atir.Mul %g_buf, %rs, %gamma {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "g">, !atir.tensor<3x1xf32, name = "rsqrt">, !atir.tensor<8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>) -> !atir.tensor<3x8xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %x, %g {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "mg">, !atir.tensor<3x1xf32, name = "mean">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "bias">, !atir.tensor<8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>, !atir.tensor<3x8xf32, name = "mg">) -> !atir.tensor<3x8xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "bias">) -> !atir.tensor<3x8xf32, name = "y">
    return %y : !atir.tensor<3x8xf32, name = "y">
  }
}

// -----

// 负例：两次 ReduceMean 的归约轴不一致（mean 走末轴、variance 走轴 0）。
// kernel 只实现末轴归约，轴不一致的“LayerNorm”不是同一形态，站点不成立。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_layer_norm
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<3x8xf32, name = "x">) -> !atir.tensor<3x8xf32, name = "y"> {
    %gamma = atir.constant "public" @"gamma" -> <8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>
    %beta = atir.constant "public" @"beta" -> <8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>
    %eps = atir.constant "public" @"eps" -> <f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>>
    %axes0 = atir.constant "private" @"n_var/axes" -> <1xi64, name = "n_var/axes", data = dense<0> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = true, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <3x1xf32, name = "mean">, <3x8xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x1xf32, name = "mean">) -> !atir.tensor<3x8xf32, name = "d">
    %sq_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "sq">
    %sq = atir.Mul %sq_buf, %d, %d {metadata = {tf.name = "n_sq", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "sq">, !atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "d">) -> !atir.tensor<3x8xf32, name = "sq">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %sq, %axes0 {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <3x1xf32, name = "var">, <3x8xf32, name = "sq">, <1xi64, name = "n_var/axes", data = dense<0> : tensor<1xi64>> -> <3x1xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %var, %eps) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<3x1xf32, name = "var_eps">, !atir.tensor<3x1xf32, name = "var">, !atir.tensor<f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>) -> !atir.tensor<3x1xf32, name = "var_eps">
    %std_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "std">
    %std = atir.Sqrt %std_buf, %ve {metadata = {tf.name = "n_sqrt", tf.op = "Sqrt"}} : <3x1xf32, name = "std">, <3x1xf32, name = "var_eps"> -> <3x1xf32, name = "std">
    %rs_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "rsqrt">
    %rs = atir.Reciprocal %rs_buf, %std {metadata = {tf.name = "n_recip", tf.op = "Reciprocal"}} : <3x1xf32, name = "rsqrt">, <3x1xf32, name = "std"> -> <3x1xf32, name = "rsqrt">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "g">
    %g = atir.Mul %g_buf, %rs, %gamma {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "g">, !atir.tensor<3x1xf32, name = "rsqrt">, !atir.tensor<8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>) -> !atir.tensor<3x8xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %x, %g {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "mg">, !atir.tensor<3x1xf32, name = "mean">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "bias">, !atir.tensor<8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>, !atir.tensor<3x8xf32, name = "mg">) -> !atir.tensor<3x8xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "bias">) -> !atir.tensor<3x8xf32, name = "y">
    return %y : !atir.tensor<3x8xf32, name = "y">
  }
}

// -----

// 负例：mean 的 ReduceMean keep_dims=false，结果为秩 1 <3xf32>；keepdims
// 形态是站点契约的一部分（kernel 按 [rows,1] 广播读取），不保留维度就不融。
// 注意：本模块保留 [3,8] 与 [3] 的 Sub/Mul 写法只为触发该守卫——ATIR
// verifier 不校验广播兼容性，这个组合本身不是合法的 ONNX 广播图。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_layer_norm
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<3x8xf32, name = "x">) -> !atir.tensor<3x8xf32, name = "y"> {
    %gamma = atir.constant "public" @"gamma" -> <8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>
    %beta = atir.constant "public" @"beta" -> <8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>
    %eps = atir.constant "public" @"eps" -> <f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<3xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = false, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <3xf32, name = "mean">, <3x8xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3xf32, name = "mean">) -> !atir.tensor<3x8xf32, name = "d">
    %sq_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "sq">
    %sq = atir.Mul %sq_buf, %d, %d {metadata = {tf.name = "n_sq", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "sq">, !atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "d">) -> !atir.tensor<3x8xf32, name = "sq">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %sq, %axes {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <3x1xf32, name = "var">, <3x8xf32, name = "sq">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %var, %eps) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<3x1xf32, name = "var_eps">, !atir.tensor<3x1xf32, name = "var">, !atir.tensor<f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>) -> !atir.tensor<3x1xf32, name = "var_eps">
    %std_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "std">
    %std = atir.Sqrt %std_buf, %ve {metadata = {tf.name = "n_sqrt", tf.op = "Sqrt"}} : <3x1xf32, name = "std">, <3x1xf32, name = "var_eps"> -> <3x1xf32, name = "std">
    %rs_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "rsqrt">
    %rs = atir.Reciprocal %rs_buf, %std {metadata = {tf.name = "n_recip", tf.op = "Reciprocal"}} : <3x1xf32, name = "rsqrt">, <3x1xf32, name = "std"> -> <3x1xf32, name = "rsqrt">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "g">
    %g = atir.Mul %g_buf, %rs, %gamma {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "g">, !atir.tensor<3x1xf32, name = "rsqrt">, !atir.tensor<8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>) -> !atir.tensor<3x8xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %x, %g {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "mg">, !atir.tensor<3xf32, name = "mean">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "bias">, !atir.tensor<8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>, !atir.tensor<3x8xf32, name = "mg">) -> !atir.tensor<3x8xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "bias">) -> !atir.tensor<3x8xf32, name = "y">
    return %y : !atir.tensor<3x8xf32, name = "y">
  }
}

// -----

// 负例：链上缺 Sqrt——Reciprocal 直接吃 Add(var, eps)。kernel 按
// Reciprocal(Sqrt(Add(var, eps))) 的固定链计算，缺一环就不融。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_layer_norm
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<3x8xf32, name = "x">) -> !atir.tensor<3x8xf32, name = "y"> {
    %gamma = atir.constant "public" @"gamma" -> <8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>
    %beta = atir.constant "public" @"beta" -> <8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>
    %eps = atir.constant "public" @"eps" -> <f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = true, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <3x1xf32, name = "mean">, <3x8xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x1xf32, name = "mean">) -> !atir.tensor<3x8xf32, name = "d">
    %sq_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "sq">
    %sq = atir.Mul %sq_buf, %d, %d {metadata = {tf.name = "n_sq", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "sq">, !atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "d">) -> !atir.tensor<3x8xf32, name = "sq">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %sq, %axes {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <3x1xf32, name = "var">, <3x8xf32, name = "sq">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %var, %eps) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<3x1xf32, name = "var_eps">, !atir.tensor<3x1xf32, name = "var">, !atir.tensor<f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>) -> !atir.tensor<3x1xf32, name = "var_eps">
    %rs_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "rsqrt">
    %rs = atir.Reciprocal %rs_buf, %ve {metadata = {tf.name = "n_recip", tf.op = "Reciprocal"}} : <3x1xf32, name = "rsqrt">, <3x1xf32, name = "var_eps"> -> <3x1xf32, name = "rsqrt">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "g">
    %g = atir.Mul %g_buf, %rs, %gamma {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "g">, !atir.tensor<3x1xf32, name = "rsqrt">, !atir.tensor<8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>) -> !atir.tensor<3x8xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %x, %g {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "mg">, !atir.tensor<3x1xf32, name = "mean">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "bias">, !atir.tensor<8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>, !atir.tensor<3x8xf32, name = "mg">) -> !atir.tensor<3x8xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "bias">) -> !atir.tensor<3x8xf32, name = "y">
    return %y : !atir.tensor<3x8xf32, name = "y">
  }
}

// -----

// 负例：链上缺 Reciprocal——g 直接乘 Sqrt 的输出。尾段形态不匹配，不融。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_layer_norm
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<3x8xf32, name = "x">) -> !atir.tensor<3x8xf32, name = "y"> {
    %gamma = atir.constant "public" @"gamma" -> <8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>
    %beta = atir.constant "public" @"beta" -> <8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>
    %eps = atir.constant "public" @"eps" -> <f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = true, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <3x1xf32, name = "mean">, <3x8xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x1xf32, name = "mean">) -> !atir.tensor<3x8xf32, name = "d">
    %sq_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "sq">
    %sq = atir.Mul %sq_buf, %d, %d {metadata = {tf.name = "n_sq", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "sq">, !atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "d">) -> !atir.tensor<3x8xf32, name = "sq">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %sq, %axes {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <3x1xf32, name = "var">, <3x8xf32, name = "sq">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %var, %eps) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<3x1xf32, name = "var_eps">, !atir.tensor<3x1xf32, name = "var">, !atir.tensor<f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>) -> !atir.tensor<3x1xf32, name = "var_eps">
    %std_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "std">
    %std = atir.Sqrt %std_buf, %ve {metadata = {tf.name = "n_sqrt", tf.op = "Sqrt"}} : <3x1xf32, name = "std">, <3x1xf32, name = "var_eps"> -> <3x1xf32, name = "std">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "g">
    %g = atir.Mul %g_buf, %std, %gamma {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "g">, !atir.tensor<3x1xf32, name = "std">, !atir.tensor<8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>) -> !atir.tensor<3x8xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %x, %g {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "mg">, !atir.tensor<3x1xf32, name = "mean">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "bias">, !atir.tensor<8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>, !atir.tensor<3x8xf32, name = "mg">) -> !atir.tensor<3x8xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "bias">) -> !atir.tensor<3x8xf32, name = "y">
    return %y : !atir.tensor<3x8xf32, name = "y">
  }
}

// -----

// 负例：variance 直接对 d=Sub(x, mean) 归约，缺 Mul(d, d) 平方这一步；
// 方差语义不同，站点不成立。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_layer_norm
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<3x8xf32, name = "x">) -> !atir.tensor<3x8xf32, name = "y"> {
    %gamma = atir.constant "public" @"gamma" -> <8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>
    %beta = atir.constant "public" @"beta" -> <8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>
    %eps = atir.constant "public" @"eps" -> <f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = true, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <3x1xf32, name = "mean">, <3x8xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x1xf32, name = "mean">) -> !atir.tensor<3x8xf32, name = "d">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %d, %axes {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <3x1xf32, name = "var">, <3x8xf32, name = "d">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %var, %eps) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<3x1xf32, name = "var_eps">, !atir.tensor<3x1xf32, name = "var">, !atir.tensor<f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>) -> !atir.tensor<3x1xf32, name = "var_eps">
    %std_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "std">
    %std = atir.Sqrt %std_buf, %ve {metadata = {tf.name = "n_sqrt", tf.op = "Sqrt"}} : <3x1xf32, name = "std">, <3x1xf32, name = "var_eps"> -> <3x1xf32, name = "std">
    %rs_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "rsqrt">
    %rs = atir.Reciprocal %rs_buf, %std {metadata = {tf.name = "n_recip", tf.op = "Reciprocal"}} : <3x1xf32, name = "rsqrt">, <3x1xf32, name = "std"> -> <3x1xf32, name = "rsqrt">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "g">
    %g = atir.Mul %g_buf, %rs, %gamma {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "g">, !atir.tensor<3x1xf32, name = "rsqrt">, !atir.tensor<8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>) -> !atir.tensor<3x8xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %x, %g {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "mg">, !atir.tensor<3x1xf32, name = "mean">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "bias">, !atir.tensor<8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>, !atir.tensor<3x8xf32, name = "mg">) -> !atir.tensor<3x8xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "bias">) -> !atir.tensor<3x8xf32, name = "y">
    return %y : !atir.tensor<3x8xf32, name = "y">
  }
}

// -----

// 负例：gamma 是 rank-2 常量（多维参数），而 kernel 的 schema 要求
// rank-1 的 scale/bias；参数维度不匹配，不融。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_layer_norm
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<3x8xf32, name = "x">) -> !atir.tensor<3x8xf32, name = "y"> {
    %gamma = atir.constant "public" @"gamma" -> <1x8xf32, name = "gamma", data = dense<1.000000> : tensor<1x8xf32>>
    %beta = atir.constant "public" @"beta" -> <8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>
    %eps = atir.constant "public" @"eps" -> <f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = true, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <3x1xf32, name = "mean">, <3x8xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x1xf32, name = "mean">) -> !atir.tensor<3x8xf32, name = "d">
    %sq_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "sq">
    %sq = atir.Mul %sq_buf, %d, %d {metadata = {tf.name = "n_sq", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "sq">, !atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "d">) -> !atir.tensor<3x8xf32, name = "sq">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %sq, %axes {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <3x1xf32, name = "var">, <3x8xf32, name = "sq">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %var, %eps) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<3x1xf32, name = "var_eps">, !atir.tensor<3x1xf32, name = "var">, !atir.tensor<f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>) -> !atir.tensor<3x1xf32, name = "var_eps">
    %std_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "std">
    %std = atir.Sqrt %std_buf, %ve {metadata = {tf.name = "n_sqrt", tf.op = "Sqrt"}} : <3x1xf32, name = "std">, <3x1xf32, name = "var_eps"> -> <3x1xf32, name = "std">
    %rs_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "rsqrt">
    %rs = atir.Reciprocal %rs_buf, %std {metadata = {tf.name = "n_recip", tf.op = "Reciprocal"}} : <3x1xf32, name = "rsqrt">, <3x1xf32, name = "std"> -> <3x1xf32, name = "rsqrt">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "g">
    %g = atir.Mul %g_buf, %rs, %gamma {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "g">, !atir.tensor<3x1xf32, name = "rsqrt">, !atir.tensor<1x8xf32, name = "gamma", data = dense<1.000000> : tensor<1x8xf32>>) -> !atir.tensor<3x8xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %x, %g {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "mg">, !atir.tensor<3x1xf32, name = "mean">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "bias">, !atir.tensor<8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>, !atir.tensor<3x8xf32, name = "mg">) -> !atir.tensor<3x8xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "bias">) -> !atir.tensor<3x8xf32, name = "y">
    return %y : !atir.tensor<3x8xf32, name = "y">
  }
}

// -----

// 负例：eps 是函数实参而不是常量——动态 eps 无法折进 kernel 属性，不融。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_layer_norm
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<3x8xf32, name = "x">, %eps: !atir.tensor<f32, name = "eps">) -> !atir.tensor<3x8xf32, name = "y"> {
    %gamma = atir.constant "public" @"gamma" -> <8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>
    %beta = atir.constant "public" @"beta" -> <8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = true, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <3x1xf32, name = "mean">, <3x8xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x1xf32, name = "mean">) -> !atir.tensor<3x8xf32, name = "d">
    %sq_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "sq">
    %sq = atir.Mul %sq_buf, %d, %d {metadata = {tf.name = "n_sq", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "sq">, !atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "d">) -> !atir.tensor<3x8xf32, name = "sq">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %sq, %axes {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <3x1xf32, name = "var">, <3x8xf32, name = "sq">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %var, %eps) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<3x1xf32, name = "var_eps">, !atir.tensor<3x1xf32, name = "var">, !atir.tensor<f32, name = "eps">) -> !atir.tensor<3x1xf32, name = "var_eps">
    %std_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "std">
    %std = atir.Sqrt %std_buf, %ve {metadata = {tf.name = "n_sqrt", tf.op = "Sqrt"}} : <3x1xf32, name = "std">, <3x1xf32, name = "var_eps"> -> <3x1xf32, name = "std">
    %rs_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "rsqrt">
    %rs = atir.Reciprocal %rs_buf, %std {metadata = {tf.name = "n_recip", tf.op = "Reciprocal"}} : <3x1xf32, name = "rsqrt">, <3x1xf32, name = "std"> -> <3x1xf32, name = "rsqrt">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "g">
    %g = atir.Mul %g_buf, %rs, %gamma {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "g">, !atir.tensor<3x1xf32, name = "rsqrt">, !atir.tensor<8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>) -> !atir.tensor<3x8xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %x, %g {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "mg">, !atir.tensor<3x1xf32, name = "mean">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "bias">, !atir.tensor<8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>, !atir.tensor<3x8xf32, name = "mg">) -> !atir.tensor<3x8xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "bias">) -> !atir.tensor<3x8xf32, name = "y">
    return %y : !atir.tensor<3x8xf32, name = "y">
  }
}

// -----

// 负例：激活张量 rank 未知（unranked，name 之外没有形状信息）。匹配器
// 要求 x 是 rank-2、末端归约轴确定，未知 rank 不融。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_layer_norm
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<*xf32, name = "x">) -> !atir.tensor<*xf32, name = "y"> {
    %gamma = atir.constant "public" @"gamma" -> <8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>
    %beta = atir.constant "public" @"beta" -> <8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>
    %eps = atir.constant "public" @"eps" -> <f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<*xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = true, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <*xf32, name = "mean">, <*xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <*xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<*xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<*xf32, name = "d">, !atir.tensor<*xf32, name = "x">, !atir.tensor<*xf32, name = "mean">) -> !atir.tensor<*xf32, name = "d">
    %sq_buf = "atir.buffer"() : () -> !atir.tensor<*xf32, name = "sq">
    %sq = atir.Mul %sq_buf, %d, %d {metadata = {tf.name = "n_sq", tf.op = "Mul"}} : (!atir.tensor<*xf32, name = "sq">, !atir.tensor<*xf32, name = "d">, !atir.tensor<*xf32, name = "d">) -> !atir.tensor<*xf32, name = "sq">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<*xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %sq, %axes {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <*xf32, name = "var">, <*xf32, name = "sq">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <*xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<*xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %var, %eps) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<*xf32, name = "var_eps">, !atir.tensor<*xf32, name = "var">, !atir.tensor<f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>) -> !atir.tensor<*xf32, name = "var_eps">
    %std_buf = "atir.buffer"() : () -> !atir.tensor<*xf32, name = "std">
    %std = atir.Sqrt %std_buf, %ve {metadata = {tf.name = "n_sqrt", tf.op = "Sqrt"}} : <*xf32, name = "std">, <*xf32, name = "var_eps"> -> <*xf32, name = "std">
    %rs_buf = "atir.buffer"() : () -> !atir.tensor<*xf32, name = "rsqrt">
    %rs = atir.Reciprocal %rs_buf, %std {metadata = {tf.name = "n_recip", tf.op = "Reciprocal"}} : <*xf32, name = "rsqrt">, <*xf32, name = "std"> -> <*xf32, name = "rsqrt">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<*xf32, name = "g">
    %g = atir.Mul %g_buf, %rs, %gamma {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<*xf32, name = "g">, !atir.tensor<*xf32, name = "rsqrt">, !atir.tensor<8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>) -> !atir.tensor<*xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<*xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %x, %g {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<*xf32, name = "xg">, !atir.tensor<*xf32, name = "x">, !atir.tensor<*xf32, name = "g">) -> !atir.tensor<*xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<*xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<*xf32, name = "mg">, !atir.tensor<*xf32, name = "mean">, !atir.tensor<*xf32, name = "g">) -> !atir.tensor<*xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<*xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<*xf32, name = "bias">, !atir.tensor<8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>, !atir.tensor<*xf32, name = "mg">) -> !atir.tensor<*xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<*xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<*xf32, name = "y">, !atir.tensor<*xf32, name = "xg">, !atir.tensor<*xf32, name = "bias">) -> !atir.tensor<*xf32, name = "y">
    return %y : !atir.tensor<*xf32, name = "y">
  }
}

// -----

// 负例：g 有链外的额外消费者（这里被一个 Sqrt 读）。kernel 单输出提供
// 不了站点中间值，站点必须不成立。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_layer_norm
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<3x8xf32, name = "x">) -> (!atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "extra">) {
    %gamma = atir.constant "public" @"gamma" -> <8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>
    %beta = atir.constant "public" @"beta" -> <8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>
    %eps = atir.constant "public" @"eps" -> <f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = true, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <3x1xf32, name = "mean">, <3x8xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x1xf32, name = "mean">) -> !atir.tensor<3x8xf32, name = "d">
    %sq_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "sq">
    %sq = atir.Mul %sq_buf, %d, %d {metadata = {tf.name = "n_sq", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "sq">, !atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "d">) -> !atir.tensor<3x8xf32, name = "sq">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %sq, %axes {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <3x1xf32, name = "var">, <3x8xf32, name = "sq">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %var, %eps) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<3x1xf32, name = "var_eps">, !atir.tensor<3x1xf32, name = "var">, !atir.tensor<f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>) -> !atir.tensor<3x1xf32, name = "var_eps">
    %std_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "std">
    %std = atir.Sqrt %std_buf, %ve {metadata = {tf.name = "n_sqrt", tf.op = "Sqrt"}} : <3x1xf32, name = "std">, <3x1xf32, name = "var_eps"> -> <3x1xf32, name = "std">
    %rs_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "rsqrt">
    %rs = atir.Reciprocal %rs_buf, %std {metadata = {tf.name = "n_recip", tf.op = "Reciprocal"}} : <3x1xf32, name = "rsqrt">, <3x1xf32, name = "std"> -> <3x1xf32, name = "rsqrt">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "g">
    %g = atir.Mul %g_buf, %rs, %gamma {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "g">, !atir.tensor<3x1xf32, name = "rsqrt">, !atir.tensor<8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>) -> !atir.tensor<3x8xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %x, %g {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "mg">, !atir.tensor<3x1xf32, name = "mean">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "bias">, !atir.tensor<8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>, !atir.tensor<3x8xf32, name = "mg">) -> !atir.tensor<3x8xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "bias">) -> !atir.tensor<3x8xf32, name = "y">
    %extra_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "extra">
    %extra = atir.Sqrt %extra_buf, %g {metadata = {tf.name = "n_extra", tf.op = "Sqrt"}} : <3x8xf32, name = "extra">, <3x8xf32, name = "g"> -> <3x8xf32, name = "extra">
    return %y, %extra : !atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "extra">
  }
}

// -----

// 负例：gamma 长度（7）与 x 末维（8）不一致。kernel 按 cols 读取
// gamma/beta，匹配器现在同样校验首维；这种站点必须在匹配期拒绝，
// 否则会先融合、到运行时才被 kernel 以 session 失败拒掉。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_layer_norm
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<3x8xf32, name = "x">) -> !atir.tensor<3x8xf32, name = "y"> {
    %gamma = atir.constant "public" @"gamma" -> <7xf32, name = "gamma", data = dense<1.000000> : tensor<7xf32>>
    %beta = atir.constant "public" @"beta" -> <8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>
    %eps = atir.constant "public" @"eps" -> <f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = true, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <3x1xf32, name = "mean">, <3x8xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x1xf32, name = "mean">) -> !atir.tensor<3x8xf32, name = "d">
    %sq_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "sq">
    %sq = atir.Mul %sq_buf, %d, %d {metadata = {tf.name = "n_sq", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "sq">, !atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "d">) -> !atir.tensor<3x8xf32, name = "sq">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %sq, %axes {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <3x1xf32, name = "var">, <3x8xf32, name = "sq">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %var, %eps) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<3x1xf32, name = "var_eps">, !atir.tensor<3x1xf32, name = "var">, !atir.tensor<f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>) -> !atir.tensor<3x1xf32, name = "var_eps">
    %std_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "std">
    %std = atir.Sqrt %std_buf, %ve {metadata = {tf.name = "n_sqrt", tf.op = "Sqrt"}} : <3x1xf32, name = "std">, <3x1xf32, name = "var_eps"> -> <3x1xf32, name = "std">
    %rs_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "rsqrt">
    %rs = atir.Reciprocal %rs_buf, %std {metadata = {tf.name = "n_recip", tf.op = "Reciprocal"}} : <3x1xf32, name = "rsqrt">, <3x1xf32, name = "std"> -> <3x1xf32, name = "rsqrt">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "g">
    %g = atir.Mul %g_buf, %rs, %gamma {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "g">, !atir.tensor<3x1xf32, name = "rsqrt">, !atir.tensor<7xf32, name = "gamma", data = dense<1.000000> : tensor<7xf32>>) -> !atir.tensor<3x8xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %x, %g {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "mg">, !atir.tensor<3x1xf32, name = "mean">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "bias">, !atir.tensor<8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>, !atir.tensor<3x8xf32, name = "mg">) -> !atir.tensor<3x8xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "bias">) -> !atir.tensor<3x8xf32, name = "y">
    return %y : !atir.tensor<3x8xf32, name = "y">
  }
}


// -----

// 负例：beta 长度（7）与 x 末维（8）不一致，与 gamma 同一守卫。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_layer_norm
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<3x8xf32, name = "x">) -> !atir.tensor<3x8xf32, name = "y"> {
    %gamma = atir.constant "public" @"gamma" -> <8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>
    %beta = atir.constant "public" @"beta" -> <7xf32, name = "beta", data = dense<0.000000> : tensor<7xf32>>
    %eps = atir.constant "public" @"eps" -> <f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = true, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <3x1xf32, name = "mean">, <3x8xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x1xf32, name = "mean">) -> !atir.tensor<3x8xf32, name = "d">
    %sq_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "sq">
    %sq = atir.Mul %sq_buf, %d, %d {metadata = {tf.name = "n_sq", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "sq">, !atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "d">) -> !atir.tensor<3x8xf32, name = "sq">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %sq, %axes {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <3x1xf32, name = "var">, <3x8xf32, name = "sq">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %var, %eps) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<3x1xf32, name = "var_eps">, !atir.tensor<3x1xf32, name = "var">, !atir.tensor<f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>) -> !atir.tensor<3x1xf32, name = "var_eps">
    %std_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "std">
    %std = atir.Sqrt %std_buf, %ve {metadata = {tf.name = "n_sqrt", tf.op = "Sqrt"}} : <3x1xf32, name = "std">, <3x1xf32, name = "var_eps"> -> <3x1xf32, name = "std">
    %rs_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "rsqrt">
    %rs = atir.Reciprocal %rs_buf, %std {metadata = {tf.name = "n_recip", tf.op = "Reciprocal"}} : <3x1xf32, name = "rsqrt">, <3x1xf32, name = "std"> -> <3x1xf32, name = "rsqrt">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "g">
    %g = atir.Mul %g_buf, %rs, %gamma {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "g">, !atir.tensor<3x1xf32, name = "rsqrt">, !atir.tensor<8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>) -> !atir.tensor<3x8xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %x, %g {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "mg">, !atir.tensor<3x1xf32, name = "mean">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "bias">, !atir.tensor<7xf32, name = "beta", data = dense<0.000000> : tensor<7xf32>>, !atir.tensor<3x8xf32, name = "mg">) -> !atir.tensor<3x8xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "bias">) -> !atir.tensor<3x8xf32, name = "y">
    return %y : !atir.tensor<3x8xf32, name = "y">
  }
}


// -----

// 负例：末端 Add 带 do_relu——LayerNorm 站点不吸收 relu，交给通用路径。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_layer_norm
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<3x8xf32, name = "x">) -> !atir.tensor<3x8xf32, name = "y"> {
    %gamma = atir.constant "public" @"gamma" -> <8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>
    %beta = atir.constant "public" @"beta" -> <8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>
    %eps = atir.constant "public" @"eps" -> <f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = true, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <3x1xf32, name = "mean">, <3x8xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x1xf32, name = "mean">) -> !atir.tensor<3x8xf32, name = "d">
    %sq_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "sq">
    %sq = atir.Mul %sq_buf, %d, %d {metadata = {tf.name = "n_sq", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "sq">, !atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "d">) -> !atir.tensor<3x8xf32, name = "sq">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %sq, %axes {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <3x1xf32, name = "var">, <3x8xf32, name = "sq">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %var, %eps) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<3x1xf32, name = "var_eps">, !atir.tensor<3x1xf32, name = "var">, !atir.tensor<f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>) -> !atir.tensor<3x1xf32, name = "var_eps">
    %std_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "std">
    %std = atir.Sqrt %std_buf, %ve {metadata = {tf.name = "n_sqrt", tf.op = "Sqrt"}} : <3x1xf32, name = "std">, <3x1xf32, name = "var_eps"> -> <3x1xf32, name = "std">
    %rs_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "rsqrt">
    %rs = atir.Reciprocal %rs_buf, %std {metadata = {tf.name = "n_recip", tf.op = "Reciprocal"}} : <3x1xf32, name = "rsqrt">, <3x1xf32, name = "std"> -> <3x1xf32, name = "rsqrt">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "g">
    %g = atir.Mul %g_buf, %rs, %gamma {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "g">, !atir.tensor<3x1xf32, name = "rsqrt">, !atir.tensor<8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>) -> !atir.tensor<3x8xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %x, %g {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "mg">, !atir.tensor<3x1xf32, name = "mean">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "bias">, !atir.tensor<8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>, !atir.tensor<3x8xf32, name = "mg">) -> !atir.tensor<3x8xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = true, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "bias">) -> !atir.tensor<3x8xf32, name = "y">
    return %y : !atir.tensor<3x8xf32, name = "y">
  }
}


// -----

// 负例：末端 Add 带 scalar 属性（标量广播形态）——不是逐元素 LayerNorm，
// 站点不成立。
// CHECK-LABEL: func.func @main
// CHECK-NOT: fused_layer_norm
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<3x8xf32, name = "x">) -> !atir.tensor<3x8xf32, name = "y"> {
    %gamma = atir.constant "public" @"gamma" -> <8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>
    %beta = atir.constant "public" @"beta" -> <8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>
    %eps = atir.constant "public" @"eps" -> <f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = true, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <3x1xf32, name = "mean">, <3x8xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x1xf32, name = "mean">) -> !atir.tensor<3x8xf32, name = "d">
    %sq_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "sq">
    %sq = atir.Mul %sq_buf, %d, %d {metadata = {tf.name = "n_sq", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "sq">, !atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "d">) -> !atir.tensor<3x8xf32, name = "sq">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %sq, %axes {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <3x1xf32, name = "var">, <3x8xf32, name = "sq">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %var, %eps) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<3x1xf32, name = "var_eps">, !atir.tensor<3x1xf32, name = "var">, !atir.tensor<f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>) -> !atir.tensor<3x1xf32, name = "var_eps">
    %std_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "std">
    %std = atir.Sqrt %std_buf, %ve {metadata = {tf.name = "n_sqrt", tf.op = "Sqrt"}} : <3x1xf32, name = "std">, <3x1xf32, name = "var_eps"> -> <3x1xf32, name = "std">
    %rs_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "rsqrt">
    %rs = atir.Reciprocal %rs_buf, %std {metadata = {tf.name = "n_recip", tf.op = "Reciprocal"}} : <3x1xf32, name = "rsqrt">, <3x1xf32, name = "std"> -> <3x1xf32, name = "rsqrt">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "g">
    %g = atir.Mul %g_buf, %rs, %gamma {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "g">, !atir.tensor<3x1xf32, name = "rsqrt">, !atir.tensor<8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>) -> !atir.tensor<3x8xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %x, %g {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "mg">, !atir.tensor<3x1xf32, name = "mean">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "bias">, !atir.tensor<8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>, !atir.tensor<3x8xf32, name = "mg">) -> !atir.tensor<3x8xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = false, relu_limit = -1.000000e+00 : f32, scalar = 1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "bias">) -> !atir.tensor<3x8xf32, name = "y">
    return %y : !atir.tensor<3x8xf32, name = "y">
  }
}


// -----

// 正例：x 在站点外还有消费者（这里被一个 Sqrt 读）。x 是边界输入，
// 融合节点以实参取它，站点外的消费者不影响融合资格。
// CHECK-LABEL: func.func @main
// CHECK: atir.ReduceMean
// CHECK-SAME: annc.fusion_materialized
// CHECK: func.func private @fused_layer_norm_
// CHECK: atir.Customize
// CHECK-SAME: custom.op_name = "LayerNorm"
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<3x8xf32, name = "x">) -> (!atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "x_extra">) {
    %gamma = atir.constant "public" @"gamma" -> <8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>
    %beta = atir.constant "public" @"beta" -> <8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>
    %eps = atir.constant "public" @"eps" -> <f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = true, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <3x1xf32, name = "mean">, <3x8xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x1xf32, name = "mean">) -> !atir.tensor<3x8xf32, name = "d">
    %sq_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "sq">
    %sq = atir.Mul %sq_buf, %d, %d {metadata = {tf.name = "n_sq", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "sq">, !atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "d">) -> !atir.tensor<3x8xf32, name = "sq">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %sq, %axes {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <3x1xf32, name = "var">, <3x8xf32, name = "sq">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %var, %eps) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<3x1xf32, name = "var_eps">, !atir.tensor<3x1xf32, name = "var">, !atir.tensor<f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>) -> !atir.tensor<3x1xf32, name = "var_eps">
    %std_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "std">
    %std = atir.Sqrt %std_buf, %ve {metadata = {tf.name = "n_sqrt", tf.op = "Sqrt"}} : <3x1xf32, name = "std">, <3x1xf32, name = "var_eps"> -> <3x1xf32, name = "std">
    %rs_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "rsqrt">
    %rs = atir.Reciprocal %rs_buf, %std {metadata = {tf.name = "n_recip", tf.op = "Reciprocal"}} : <3x1xf32, name = "rsqrt">, <3x1xf32, name = "std"> -> <3x1xf32, name = "rsqrt">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "g">
    %g = atir.Mul %g_buf, %rs, %gamma {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "g">, !atir.tensor<3x1xf32, name = "rsqrt">, !atir.tensor<8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>) -> !atir.tensor<3x8xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %x, %g {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "mg">, !atir.tensor<3x1xf32, name = "mean">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "bias">, !atir.tensor<8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>, !atir.tensor<3x8xf32, name = "mg">) -> !atir.tensor<3x8xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "bias">) -> !atir.tensor<3x8xf32, name = "y">
    %x_extra_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "x_extra">
    %x_extra = atir.Sqrt %x_extra_buf, %x {metadata = {tf.name = "n_x_extra", tf.op = "Sqrt"}} : <3x8xf32, name = "x_extra">, <3x8xf32, name = "x"> -> <3x8xf32, name = "x_extra">
    return %y, %x_extra : !atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "x_extra">
  }
}


// -----

// 正例：可交换操作数顺序——eps 放在 Add 的左侧，g 与 gamma 的 Mul、
// x 与 g 的 Mul 都左右互换。匹配器按“常量/非 Mul 操作数”识别角色，
// 顺序变化不改变融合资格。
// CHECK-LABEL: func.func @main
// CHECK: atir.ReduceMean
// CHECK-SAME: annc.fusion_materialized
// CHECK: func.func private @fused_layer_norm_
// CHECK: atir.Customize
// CHECK-SAME: custom.op_name = "LayerNorm"
module attributes {module.state = "atir", annc.contract_version = "1.0.0"} {
  func.func @main(%x: !atir.tensor<3x8xf32, name = "x">) -> !atir.tensor<3x8xf32, name = "y"> {
    %gamma = atir.constant "public" @"gamma" -> <8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>
    %beta = atir.constant "public" @"beta" -> <8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>
    %eps = atir.constant "public" @"eps" -> <f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>
    %axes = atir.constant "private" @"n_mean/axes" -> <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>>
    %mean_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "mean">
    %mean = atir.ReduceMean %mean_buf, %x, %axes {keep_dims = true, metadata = {tf.name = "n_mean", tf.op = "ReduceMean"}} : <3x1xf32, name = "mean">, <3x8xf32, name = "x">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "mean">
    %d_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "d">
    %d = atir.Sub %d_buf, %x, %mean {metadata = {tf.name = "n_sub1", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "x">, !atir.tensor<3x1xf32, name = "mean">) -> !atir.tensor<3x8xf32, name = "d">
    %sq_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "sq">
    %sq = atir.Mul %sq_buf, %d, %d {metadata = {tf.name = "n_sq", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "sq">, !atir.tensor<3x8xf32, name = "d">, !atir.tensor<3x8xf32, name = "d">) -> !atir.tensor<3x8xf32, name = "sq">
    %var_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var">
    %var = atir.ReduceMean %var_buf, %sq, %axes {keep_dims = true, metadata = {tf.name = "n_var", tf.op = "ReduceMean"}} : <3x1xf32, name = "var">, <3x8xf32, name = "sq">, <1xi64, name = "n_mean/axes", data = dense<-1> : tensor<1xi64>> -> <3x1xf32, name = "var">
    %ve_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "var_eps">
    %ve = "atir.Add"(%ve_buf, %eps, %var) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_add1", tf.op = "Add"}} : (!atir.tensor<3x1xf32, name = "var_eps">, !atir.tensor<f32, name = "eps", data = dense<9.99999974E-6> : tensor<f32>>, !atir.tensor<3x1xf32, name = "var">) -> !atir.tensor<3x1xf32, name = "var_eps">
    %std_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "std">
    %std = atir.Sqrt %std_buf, %ve {metadata = {tf.name = "n_sqrt", tf.op = "Sqrt"}} : <3x1xf32, name = "std">, <3x1xf32, name = "var_eps"> -> <3x1xf32, name = "std">
    %rs_buf = "atir.buffer"() : () -> !atir.tensor<3x1xf32, name = "rsqrt">
    %rs = atir.Reciprocal %rs_buf, %std {metadata = {tf.name = "n_recip", tf.op = "Reciprocal"}} : <3x1xf32, name = "rsqrt">, <3x1xf32, name = "std"> -> <3x1xf32, name = "rsqrt">
    %g_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "g">
    %g = atir.Mul %g_buf, %gamma, %rs {metadata = {tf.name = "n_g", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "g">, !atir.tensor<8xf32, name = "gamma", data = dense<1.000000> : tensor<8xf32>>, !atir.tensor<3x1xf32, name = "rsqrt">) -> !atir.tensor<3x8xf32, name = "g">
    %xg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "xg">
    %xg = atir.Mul %xg_buf, %g, %x {metadata = {tf.name = "n_xg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "g">, !atir.tensor<3x8xf32, name = "x">) -> !atir.tensor<3x8xf32, name = "xg">
    %mg_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "mg">
    %mg = atir.Mul %mg_buf, %mean, %g {metadata = {tf.name = "n_mg", tf.op = "Mul"}} : (!atir.tensor<3x8xf32, name = "mg">, !atir.tensor<3x1xf32, name = "mean">, !atir.tensor<3x8xf32, name = "g">) -> !atir.tensor<3x8xf32, name = "mg">
    %bias_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "bias">
    %bias = atir.Sub %bias_buf, %beta, %mg {metadata = {tf.name = "n_bias", tf.op = "Sub"}} : (!atir.tensor<3x8xf32, name = "bias">, !atir.tensor<8xf32, name = "beta", data = dense<0.000000> : tensor<8xf32>>, !atir.tensor<3x8xf32, name = "mg">) -> !atir.tensor<3x8xf32, name = "bias">
    %y_buf = "atir.buffer"() : () -> !atir.tensor<3x8xf32, name = "y">
    %y = "atir.Add"(%y_buf, %xg, %bias) <{do_relu = false, relu_limit = -1.000000e+00 : f32}> {metadata = {tf.name = "n_out", tf.op = "Add"}} : (!atir.tensor<3x8xf32, name = "y">, !atir.tensor<3x8xf32, name = "xg">, !atir.tensor<3x8xf32, name = "bias">) -> !atir.tensor<3x8xf32, name = "y">
    return %y : !atir.tensor<3x8xf32, name = "y">
  }
}

