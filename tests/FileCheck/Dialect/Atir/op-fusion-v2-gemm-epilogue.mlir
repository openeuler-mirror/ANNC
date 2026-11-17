// RUN: ANNC_GEMM=2 annc-opt %s -atir-op-fusion-v2 | FileCheck %s
// RUN: ANNC_GEMM=99 annc-opt %s -atir-op-fusion-v2 | FileCheck %s
// RUN: ANNC_GEMM=1 annc-opt %s -atir-op-fusion-v2 | FileCheck %s --check-prefix=GEMM-BASIC
// RUN: ANNC_GEMM=0 annc-opt %s -atir-op-fusion-v2 | FileCheck %s --check-prefix=GEMM-OFF
// RUN: annc-opt %s -atir-op-fusion-v2 | FileCheck %s --check-prefix=GEMM-OFF
//
// op-fusion-v2 pattern: dot GEMM tail fusion (the V2 carrier form of the
// V1 FuseMatMulAsFuncCall deployment). ANNC_GEMM=2/99 (epilogue/all) absorbs
// the single-user tail chain (add/multiply with rank-1 broadcast weights,
// maximum with a zero constant = relu); aux scaffolding (broadcast/shape
// chains/constants) clones into the kernel and tail inputs become kernel
// params. =1 (basic) splits the bare dot only (tail stays in main).
module {
  // Main case: full four-step tail (add/mul/add/relu); main keeps call+return.
  // CHECK-LABEL: func.func @main(
  // CHECK: %[[CALL:.*]] = call @fused_matmul_add_mul_add_relu_{{[0-9a-fA-F]+}}(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<?x400xf32>, tensor<400x400xf32>, tensor<400xf32>, tensor<400xf32>, tensor<400xf32>) -> tensor<?x400xf32>
  // CHECK: return %[[CALL]] : tensor<?x400xf32>
  func.func @main(%arg0: tensor<?x400xf32>, %arg1: tensor<400x400xf32>, %arg2: tensor<400xf32>, %arg3: tensor<400xf32>, %arg4: tensor<400xf32>) -> tensor<?x400xf32> attributes {tf.arg_names = ["model/input_x:0", "model/dense/kernel:0", "model/dense/bias:0", "model/dense/scale:0", "model/dense/shift:0"]} {
    %batch = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x400xf32>) -> tensor<i32> loc("model/Shape")
    %batch_vector = stablehlo.reshape %batch : (tensor<i32>) -> tensor<1xi32> loc("model/Shape")
    %width = stablehlo.constant dense<400> : tensor<1xi32> loc("model/ShapeConst")
    %shape = stablehlo.concatenate %batch_vector, %width, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc("model/ShapeConcat")
    %dot = stablehlo.dot %arg0, %arg1, precision = [DEFAULT, DEFAULT] : (tensor<?x400xf32>, tensor<400x400xf32>) -> tensor<?x400xf32> loc("model/dense_1/dense/MatMul")
    %bias = stablehlo.dynamic_broadcast_in_dim %arg2, %shape, dims = [1] : (tensor<400xf32>, tensor<2xi32>) -> tensor<?x400xf32> loc("model/dense_1/dense/BiasAdd")
    %add = stablehlo.add %dot, %bias : tensor<?x400xf32> loc(fused["AddV2:", "model/dense_1/dense/add/add"])
    %scale = stablehlo.dynamic_broadcast_in_dim %arg3, %shape, dims = [1] : (tensor<400xf32>, tensor<2xi32>) -> tensor<?x400xf32> loc("model/dense_1/bn/mul_bcast")
    %mul = stablehlo.multiply %add, %scale : tensor<?x400xf32> loc(fused["Mul:", "model/dense_1/bn/mul"])
    %shift = stablehlo.dynamic_broadcast_in_dim %arg4, %shape, dims = [1] : (tensor<400xf32>, tensor<2xi32>) -> tensor<?x400xf32> loc("model/dense_1/bn/add_bcast")
    %shifted = stablehlo.add %mul, %shift : tensor<?x400xf32> loc(fused["AddV2:", "model/dense_1/bn/add"])
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc("model/ReluZero")
    %zeros = stablehlo.dynamic_broadcast_in_dim %zero, %shape, dims = [] : (tensor<f32>, tensor<2xi32>) -> tensor<?x400xf32> loc("model/ReluZeroBcast")
    %relu = stablehlo.maximum %shifted, %zeros : tensor<?x400xf32> loc(fused["Relu:", "model/dense_1/relu"])
    return %relu : tensor<?x400xf32>
  }

  // Truncation case: the mid-chain add has two users (mul + negate), so the
  // chain truncates there (prefix keeps dot + add; V2 truncation semantics).
  // The multi-user value rides the call return for its main-side consumer;
  // the bias broadcast and shape chain stay in main for the mul.
  // CHECK-LABEL: func.func @main_trunc(
  // CHECK: stablehlo.dynamic_broadcast_in_dim %arg2
  // CHECK: %[[CT:.*]] = call @fused_matmul_add_{{[0-9a-fA-F]+}}(%arg0, %arg1, %arg2) : (tensor<?x64xf32>, tensor<64x32xf32>, tensor<32xf32>) -> tensor<?x32xf32>
  // CHECK: stablehlo.multiply %[[CT]]
  // CHECK: stablehlo.negate %[[CT]]
  // CHECK: return
  func.func @main_trunc(%arg0: tensor<?x64xf32>, %arg1: tensor<64x32xf32>, %arg2: tensor<32xf32>) -> tensor<?x32xf32> attributes {tf.arg_names = ["model/input_t:0", "model/dense_t/kernel:0", "model/dense_t/bias:0"]} {
    %batch_t = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x64xf32>) -> tensor<i32> loc("model/dense_t/Shape")
    %bv_t = stablehlo.reshape %batch_t : (tensor<i32>) -> tensor<1xi32> loc("model/dense_t/Shape")
    %w_t = stablehlo.constant dense<32> : tensor<1xi32> loc("model/dense_t/ShapeConst")
    %shape_t = stablehlo.concatenate %bv_t, %w_t, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc("model/dense_t/ShapeConcat")
    %dot = stablehlo.dot %arg0, %arg1, precision = [DEFAULT, DEFAULT] : (tensor<?x64xf32>, tensor<64x32xf32>) -> tensor<?x32xf32> loc("model/dense_t/MatMul")
    %bias_t = stablehlo.dynamic_broadcast_in_dim %arg2, %shape_t, dims = [1] : (tensor<32xf32>, tensor<2xi32>) -> tensor<?x32xf32> loc("model/dense_t/bias_bcast")
    %add_tr = stablehlo.add %dot, %bias_t : tensor<?x32xf32> loc("model/dense_t/add")
    %mul_tr = stablehlo.multiply %add_tr, %bias_t : tensor<?x32xf32> loc("model/dense_t/mul")
    %neg = stablehlo.negate %add_tr : tensor<?x32xf32> loc("model/dense_t/negate")
    %sum = stablehlo.add %mul_tr, %neg : tensor<?x32xf32> loc("model/dense_t/sum")
    return %sum : tensor<?x32xf32>
  }

  // Scaffold co-reference on a mid-chain value: the shape_of reader of the
  // intermediate add must survive by re-pointing to the call result (the
  // epilogue steps are shape-preserving); regression guard for the
  // use-after-free that erased members used to leave behind.
  // CHECK-LABEL: func.func @main_scaffold_mid(
  // CHECK: %[[CM:.*]] = call @fused_matmul_add_{{[0-9a-fA-F]+}}(%arg0, %arg1, %arg2) : (tensor<?x64xf32>, tensor<64x32xf32>, tensor<32xf32>) -> tensor<?x32xf32>
  // CHECK: %[[SH:.*]] = shape.shape_of %[[CM]] : tensor<?x32xf32> -> tensor<2xindex>
  // CHECK: %[[MU:.*]] = stablehlo.multiply %[[CM]]
  // CHECK: return %[[MU]], %[[SH]] : tensor<?x32xf32>, tensor<2xindex>
  func.func @main_scaffold_mid(%arg0: tensor<?x64xf32>, %arg1: tensor<64x32xf32>, %arg2: tensor<32xf32>) -> (tensor<?x32xf32>, tensor<2xindex>) attributes {tf.arg_names = ["model/input_s:0", "model/dense_s/kernel:0", "model/dense_s/bias:0"]} {
    %batch_s = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x64xf32>) -> tensor<i32> loc("model/dense_s/Shape")
    %bv_s = stablehlo.reshape %batch_s : (tensor<i32>) -> tensor<1xi32> loc("model/dense_s/Shape")
    %w_s = stablehlo.constant dense<32> : tensor<1xi32> loc("model/dense_s/ShapeConst")
    %shape_s = stablehlo.concatenate %bv_s, %w_s, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc("model/dense_s/ShapeConcat")
    %dot_s = stablehlo.dot %arg0, %arg1, precision = [DEFAULT, DEFAULT] : (tensor<?x64xf32>, tensor<64x32xf32>) -> tensor<?x32xf32> loc("model/dense_s/MatMul")
    %bias_s = stablehlo.dynamic_broadcast_in_dim %arg2, %shape_s, dims = [1] : (tensor<32xf32>, tensor<2xi32>) -> tensor<?x32xf32> loc("model/dense_s/bias_bcast")
    %add_s = stablehlo.add %dot_s, %bias_s : tensor<?x32xf32> loc("model/dense_s/add")
    %shp_s = shape.shape_of %add_s : tensor<?x32xf32> -> tensor<2xindex> loc("model/dense_s/shape")
    %mul_s = stablehlo.multiply %add_s, %bias_s : tensor<?x32xf32> loc("model/dense_s/mul")
    return %mul_s, %shp_s : tensor<?x32xf32>, tensor<2xindex>
  }

  // Main kernel: signature (lhs, rhs, bias, scale, shift) -> relu output;
  // aux (shape chain / three weight broadcasts / relu zero) and members
  // (dot/add/mul/add/maximum) all live in the kernel; context v3 contract:
  // fusion.pattern=matmul_add_mul_add_relu, args 4 fixed + 1 dynamic,
  // kernel_arg_order empty (context-allocated outputs), outputs=tail:0.
  // CHECK: func.func private @fused_matmul_add_mul_add_relu_{{[0-9a-fA-F]+}}(%{{.*}}: tensor<?x400xf32>, %{{.*}}: tensor<400x400xf32>, %{{.*}}: tensor<400xf32>, %{{.*}}: tensor<400xf32>, %{{.*}}: tensor<400xf32>) -> tensor<?x400xf32> attributes {annc.execution_mode = "jit", annc.kernel, fusion.metadata =
  // CHECK-SAME: abi = "annc_context_v3"
  // CHECK-SAME: role = "fixed"
  // CHECK-SAME: role = "fixed"
  // CHECK-SAME: role = "fixed"
  // CHECK-SAME: role = "fixed"
  // CHECK-SAME: role = "dynamic"
  // CHECK-SAME: execution_mode = "jit"
  // CHECK-SAME: kernel_arg_order = []
  // CHECK-SAME: kernel_name = "fused_matmul_add_mul_add_relu_{{[0-9a-fA-F]+}}"
  // CHECK-SAME: role = "output"
  // CHECK-SAME: tf_name = "model/dense_1/relu:0"
  // CHECK-SAME: symbolic_signature = ""
  // CHECK-SAME: template_fingerprint = "
  // CHECK-SAME: tf.name = "annc_fused_model_dense_1_relu"
  // CHECK-SAME: fusion.pattern = "matmul_add_mul_add_relu", llvm.emit_c_interface}
  // Kernel body in dependency order: the dot lands first (carrying the
  // V1-shaped candidate plan for backend folding), shape chains/broadcasts
  // interleave, the relu zero pads in, then maximum precedes the return.
  // CHECK: stablehlo.dot %{{.*}}, %{{.*}}, precision = [DEFAULT, DEFAULT] {annc.gemm.epilogue.candidate = {output_role = "main_d", steps = [{broadcast = "n", input = 1 : i64, opcode = "add"}, {broadcast = "n", input = 1 : i64, opcode = "mul"}, {broadcast = "n", input = 1 : i64, opcode = "add"}, {broadcast = "scalar", input = 0 : i64, limit = -1.000000e+00 : f32, opcode = "relu"}], terminal_view = "identity", version = 1 : i64}} : (tensor<?x400xf32>, tensor<400x400xf32>) -> tensor<?x400xf32>
  // CHECK: stablehlo.get_dimension_size %{{.*}}, dim = 0
  // CHECK: stablehlo.concatenate
  // CHECK: stablehlo.add
  // CHECK: stablehlo.multiply
  // CHECK: stablehlo.add
  // CHECK: stablehlo.constant dense<0.000000e+00>
  // CHECK: stablehlo.maximum
  // CHECK: return %{{.*}} : tensor<?x400xf32>

  // Truncated-site kernel: members stop at add (pattern=matmul_add); the
  // (lhs, rhs, bias)。
  // CHECK: func.func private @fused_matmul_add_{{[0-9a-fA-F]+}}(%{{.*}}: tensor<?x64xf32>, %{{.*}}: tensor<64x32xf32>, %{{.*}}: tensor<32xf32>) -> tensor<?x32xf32> attributes {annc.execution_mode = "jit", annc.kernel,
  // CHECK-SAME: kernel_arg_order = []
  // CHECK-SAME: tf.name = "annc_fused_model_dense_t_add"
  // The truncated-site dot carries a single-step candidate (add).
  // CHECK: stablehlo.dot %{{.*}}, %{{.*}}, precision = [DEFAULT, DEFAULT] {annc.gemm.epilogue.candidate = {output_role = "main_d", steps = [{broadcast = "n", input = 1 : i64, opcode = "add"}], terminal_view = "identity", version = 1 : i64}} : (tensor<?x64xf32>, tensor<64x32xf32>) -> tensor<?x32xf32>
  // CHECK: stablehlo.add
  // CHECK: return

  // basic (ANNC_GEMM=1): bare dot only; the tail consumes the call result.
  // GEMM-BASIC-LABEL: func.func @main(
  // GEMM-BASIC: %[[C:.*]] = call @fused_matmul_{{[0-9a-fA-F]+}}(%arg0, %arg1) : (tensor<?x400xf32>, tensor<400x400xf32>) -> tensor<?x400xf32>
  // GEMM-BASIC: stablehlo.add %[[C]]
  // GEMM-BASIC: stablehlo.multiply
  // GEMM-BASIC: stablehlo.maximum
  // GEMM-BASIC: return
  // GEMM-BASIC-NOT: fused_matmul_add_mul_add_relu

  // off (default/0): untouched.
  // GEMM-OFF-LABEL: func.func @main(
  // GEMM-OFF: stablehlo.dot
  // GEMM-OFF-NOT: fused_matmul

}
