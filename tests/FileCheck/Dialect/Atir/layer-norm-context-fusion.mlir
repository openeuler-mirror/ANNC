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
