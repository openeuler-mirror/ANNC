// RUN: annc-opt --split-input-file %s --atir-identity-canonicalize --atir-op-fusion | FileCheck %s
// RUN: annc-asm --split-input-file %s -atir-fast-codegen | FileCheck %s --check-prefix=FAST

// CHECK-LABEL: func.func @main
// CHECK: atir.Gather
// CHECK: atir.Gather
// CHECK-SAME: annc.fusion_materialized
// CHECK-LABEL: func.func private @fused_
// CHECK-SAME: %arg0: !llvm.ptr,
// CHECK-SAME: %arg1: !atir.tensor<2x4xf32>, %arg2: !atir.tensor<4x3xi64>, %arg3: !atir.tensor<2xi32>
// CHECK: fusion.metadata = {
// CHECK-SAME: abi = "annc_execution_v2"
// CHECK-SAME: tf_name = "first_unique:0"
// CHECK-SAME: tf_name = "first_unique:1"
// CHECK-SAME: tf_name = "outer_gather:0"

// FAST-LABEL: func.func private @fused_kp_fused_gather
// FAST-SAME: %{{.*}}: !llvm.ptr,
// FAST-SAME: %{{.*}}: !atir.tensor<2x4xf32>, %{{.*}}: !atir.tensor<4x3xi64>, %{{.*}}: !atir.tensor<2xi32>
// FAST: atir.Customize
// FAST-SAME: custom.op_name = "KPFusedGather"
// FAST-SAME: custom.result_names = ["unique_values", "unique_indices", "gathered"]
// FAST-NOT: atir.Unique
// FAST-NOT: atir.Gather
module attributes {module.state = "atir"} {
  func.func @main(
      %data: !atir.tensor<2x4xf32>, %keys: !atir.tensor<4x3xi64>,
      %begin: !atir.tensor<2xi32>)
      -> (!atir.tensor<?xi64, name = "__annc_output_0">,
          !atir.tensor<?xi32, name = "__annc_output_1">,
          !atir.tensor<?x4xf32, name = "__annc_output_2">) {
    %axis = atir.constant "public" @"axis" ->
      <i32, name = "axis", data = dense<0> : tensor<i32>>
    %end = atir.constant "public" @"end" ->
      <2xi32, name = "end", data = dense<[0, 0]> : tensor<2xi32>>
    %stride = atir.constant "public" @"stride" ->
      <2xi32, name = "stride", data = dense<[1, 1]> : tensor<2xi32>>
    %slice_buffer = "atir.buffer"() : () -> !atir.tensor<?xi64>
    %slice = atir.StridedSlice %slice_buffer, %keys, %begin, %end, %stride {
      beginMask = 1 : i32, endMask = 1 : i32, ellipsisMask = 0 : i32,
      newAxisMask = 0 : i32, shrinkAxisMask = 2 : i32
    } : !atir.tensor<?xi64>, !atir.tensor<4x3xi64>, !atir.tensor<2xi32>,
        !atir.tensor<2xi32, name = "end", data = dense<0> : tensor<2xi32>>,
        !atir.tensor<2xi32, name = "stride", data = dense<1> : tensor<2xi32>>
        -> !atir.tensor<?xi64>
    %unique_values, %unique_indices = atir.Unique %slice :
      (!atir.tensor<?xi64>) -> (!atir.tensor<?xi64>, !atir.tensor<?xi32>)
    %unique2_values, %unique2_indices = atir.Unique %unique_values :
      (!atir.tensor<?xi64>) ->
      (!atir.tensor<?xi64>, !atir.tensor<?xi32>)
    %inner_buffer = "atir.buffer"() : () -> !atir.tensor<?x4xf32>
    %inner = atir.Gather %inner_buffer, %data, %unique2_values, %axis :
      !atir.tensor<?x4xf32>, !atir.tensor<2x4xf32>, !atir.tensor<?xi64>,
      !atir.tensor<i32, name = "axis", data = dense<0> : tensor<i32>>
      -> !atir.tensor<?x4xf32>
    %outer_buffer = "atir.buffer"() : () -> !atir.tensor<?x4xf32>
    %outer = atir.Gather %outer_buffer, %inner, %unique2_indices, %axis :
      !atir.tensor<?x4xf32>, !atir.tensor<?x4xf32>, !atir.tensor<?xi32>,
      !atir.tensor<i32, name = "axis", data = dense<0> : tensor<i32>>
      -> !atir.tensor<?x4xf32>
    %output0_buffer = "atir.buffer"() : () ->
      !atir.tensor<?xi64, name = "__annc_output_0">
    %output0 = atir.Identity %output0_buffer, %unique_values {
      metadata = {tf.output_tensor = "first_unique:0"}
    } : <?xi64, name = "__annc_output_0">, <?xi64>
      -> <?xi64, name = "__annc_output_0">
    %output1_buffer = "atir.buffer"() : () ->
      !atir.tensor<?xi32, name = "__annc_output_1">
    %output1 = atir.Identity %output1_buffer, %unique_indices {
      metadata = {tf.output_tensor = "first_unique:1"}
    } : <?xi32, name = "__annc_output_1">, <?xi32>
      -> <?xi32, name = "__annc_output_1">
    %output2_buffer = "atir.buffer"() : () ->
      !atir.tensor<?x4xf32, name = "__annc_output_2">
    %output2 = atir.Identity %output2_buffer, %outer {
      metadata = {tf.output_tensor = "outer_gather:0"}
    } : <?x4xf32, name = "__annc_output_2">, <?x4xf32>
      -> <?x4xf32, name = "__annc_output_2">
    return %output0, %output1, %output2 :
      !atir.tensor<?xi64, name = "__annc_output_0">,
      !atir.tensor<?xi32, name = "__annc_output_1">,
      !atir.tensor<?x4xf32, name = "__annc_output_2">
  }
}

// -----

// ss.end = [0, 1] (masked by end_mask=1): the end values are ignored by TF
// semantics and by the kernel — the value check must not block the match.
// CHECK-LABEL: func.func @main
// CHECK: atir.Gather
// CHECK: atir.Gather
// CHECK-SAME: annc.fusion_materialized
// CHECK-LABEL: func.func private @fused_
// CHECK-SAME: %arg0: !llvm.ptr,
// CHECK-SAME: fusion.metadata = {
module attributes {module.state = "atir"} {
  func.func @main(
      %data: !atir.tensor<2x4xf32>, %keys: !atir.tensor<4x3xi64>,
      %begin: !atir.tensor<2xi32>)
      -> (!atir.tensor<?xi64, name = "__annc_output_0">,
          !atir.tensor<?xi32, name = "__annc_output_1">,
          !atir.tensor<?x4xf32, name = "__annc_output_2">) {
    %axis = atir.constant "public" @"axis" ->
      <i32, name = "axis", data = dense<0> : tensor<i32>>
    %end = atir.constant "public" @"end" ->
      <2xi32, name = "end", data = dense<[0, 1]> : tensor<2xi32>>
    %stride = atir.constant "public" @"stride" ->
      <2xi32, name = "stride", data = dense<[1, 1]> : tensor<2xi32>>
    %slice_buffer = "atir.buffer"() : () -> !atir.tensor<?xi64>
    %slice = atir.StridedSlice %slice_buffer, %keys, %begin, %end, %stride {
      beginMask = 1 : i32, endMask = 1 : i32, ellipsisMask = 0 : i32,
      newAxisMask = 0 : i32, shrinkAxisMask = 2 : i32
    } : !atir.tensor<?xi64>, !atir.tensor<4x3xi64>, !atir.tensor<2xi32>,
        !atir.tensor<2xi32, name = "end", data = dense<[0, 1]> : tensor<2xi32>>,
        !atir.tensor<2xi32, name = "stride", data = dense<1> : tensor<2xi32>>
        -> !atir.tensor<?xi64>
    %unique_values, %unique_indices = atir.Unique %slice :
      (!atir.tensor<?xi64>) -> (!atir.tensor<?xi64>, !atir.tensor<?xi32>)
    %unique2_values, %unique2_indices = atir.Unique %unique_values :
      (!atir.tensor<?xi64>) ->
      (!atir.tensor<?xi64>, !atir.tensor<?xi32>)
    %inner_buffer = "atir.buffer"() : () -> !atir.tensor<?x4xf32>
    %inner = atir.Gather %inner_buffer, %data, %unique2_values, %axis :
      !atir.tensor<?x4xf32>, !atir.tensor<2x4xf32>, !atir.tensor<?xi64>,
      !atir.tensor<i32, name = "axis", data = dense<0> : tensor<i32>>
      -> !atir.tensor<?x4xf32>
    %outer_buffer = "atir.buffer"() : () -> !atir.tensor<?x4xf32>
    %outer = atir.Gather %outer_buffer, %inner, %unique2_indices, %axis :
      !atir.tensor<?x4xf32>, !atir.tensor<?x4xf32>, !atir.tensor<?xi32>,
      !atir.tensor<i32, name = "axis", data = dense<0> : tensor<i32>>
      -> !atir.tensor<?x4xf32>
    %output0_buffer = "atir.buffer"() : () ->
      !atir.tensor<?xi64, name = "__annc_output_0">
    %output0 = atir.Identity %output0_buffer, %unique_values {
      metadata = {tf.output_tensor = "first_unique:0"}
    } : <?xi64, name = "__annc_output_0">, <?xi64>
      -> <?xi64, name = "__annc_output_0">
    %output1_buffer = "atir.buffer"() : () ->
      !atir.tensor<?xi32, name = "__annc_output_1">
    %output1 = atir.Identity %output1_buffer, %unique_indices {
      metadata = {tf.output_tensor = "first_unique:1"}
    } : <?xi32, name = "__annc_output_1">, <?xi32>
      -> <?xi32, name = "__annc_output_1">
    %output2_buffer = "atir.buffer"() : () ->
      !atir.tensor<?x4xf32, name = "__annc_output_2">
    %output2 = atir.Identity %output2_buffer, %outer {
      metadata = {tf.output_tensor = "outer_gather:0"}
    } : <?x4xf32, name = "__annc_output_2">, <?x4xf32>
      -> <?x4xf32, name = "__annc_output_2">
    return %output0, %output1, %output2 :
      !atir.tensor<?xi64, name = "__annc_output_0">,
      !atir.tensor<?xi32, name = "__annc_output_1">,
      !atir.tensor<?x4xf32, name = "__annc_output_2">
  }
}

// -----
module attributes {module.state = "atir"} {
  func.func private @fused_kp_fused_gather(
      %execution: !llvm.ptr,
      %data: !atir.tensor<2x4xf32>, %keys: !atir.tensor<4x3xi64>,
      %begin: !atir.tensor<2xi32>)
      -> (!atir.tensor<?xi64>, !atir.tensor<?xi32>, !atir.tensor<?x4xf32>)
      attributes {
        annc.kernel,
        fusion.metadata = {abi = "annc_execution_v2"}
      } {
    %axis = atir.constant "public" @"axis" ->
      <i32, name = "axis", data = dense<0> : tensor<i32>>
    %end = atir.constant "public" @"end" ->
      <2xi32, name = "end", data = dense<[0, 0]> : tensor<2xi32>>
    %stride = atir.constant "public" @"stride" ->
      <2xi32, name = "stride", data = dense<[1, 1]> : tensor<2xi32>>
    %slice_buffer = "atir.buffer"() : () -> !atir.tensor<?xi64>
    %slice = atir.StridedSlice %slice_buffer, %keys, %begin, %end, %stride {
      beginMask = 1 : i32, endMask = 1 : i32, ellipsisMask = 0 : i32,
      newAxisMask = 0 : i32, shrinkAxisMask = 2 : i32
    } : !atir.tensor<?xi64>, !atir.tensor<4x3xi64>, !atir.tensor<2xi32>,
        !atir.tensor<2xi32, name = "end", data = dense<0> : tensor<2xi32>>,
        !atir.tensor<2xi32, name = "stride", data = dense<1> : tensor<2xi32>>
        -> !atir.tensor<?xi64>
    %unique_values, %unique_indices = atir.Unique %slice :
      (!atir.tensor<?xi64>) -> (!atir.tensor<?xi64>, !atir.tensor<?xi32>)
    %unique2_values, %unique2_indices = atir.Unique %unique_values :
      (!atir.tensor<?xi64>) ->
      (!atir.tensor<?xi64>, !atir.tensor<?xi32>)
    %inner_buffer = "atir.buffer"() : () -> !atir.tensor<?x4xf32>
    %inner = atir.Gather %inner_buffer, %data, %unique2_values, %axis :
      !atir.tensor<?x4xf32>, !atir.tensor<2x4xf32>, !atir.tensor<?xi64>,
      !atir.tensor<i32, name = "axis", data = dense<0> : tensor<i32>>
      -> !atir.tensor<?x4xf32>
    %outer_buffer = "atir.buffer"() : () -> !atir.tensor<?x4xf32>
    %outer = atir.Gather %outer_buffer, %inner, %unique2_indices, %axis :
      !atir.tensor<?x4xf32>, !atir.tensor<?x4xf32>, !atir.tensor<?xi32>,
      !atir.tensor<i32, name = "axis", data = dense<0> : tensor<i32>>
      -> !atir.tensor<?x4xf32>
    return %unique_values, %unique_indices, %outer : !atir.tensor<?xi64>,
      !atir.tensor<?xi32>, !atir.tensor<?x4xf32>
  }
}

// -----

// Fast-codegen with ss.end = [0, 1]: same shared match — folds to Customize.
// (Appended last so the FAST check order matches the split-input output
// order.)
// FAST-LABEL: func.func private @fused_kp_fused_gather_end_01
// FAST: atir.Customize
// FAST-SAME: custom.op_name = "KPFusedGather"
// FAST-NOT: atir.Unique
module attributes {module.state = "atir"} {
  func.func private @fused_kp_fused_gather_end_01(
      %execution: !llvm.ptr,
      %data: !atir.tensor<2x4xf32>, %keys: !atir.tensor<4x3xi64>,
      %begin: !atir.tensor<2xi32>)
      -> (!atir.tensor<?xi64>, !atir.tensor<?xi32>, !atir.tensor<?x4xf32>)
      attributes {
        annc.kernel,
        fusion.metadata = {abi = "annc_execution_v2"}
      } {
    %axis = atir.constant "public" @"axis" ->
      <i32, name = "axis", data = dense<0> : tensor<i32>>
    %end = atir.constant "public" @"end" ->
      <2xi32, name = "end", data = dense<[0, 1]> : tensor<2xi32>>
    %stride = atir.constant "public" @"stride" ->
      <2xi32, name = "stride", data = dense<[1, 1]> : tensor<2xi32>>
    %slice_buffer = "atir.buffer"() : () -> !atir.tensor<?xi64>
    %slice = atir.StridedSlice %slice_buffer, %keys, %begin, %end, %stride {
      beginMask = 1 : i32, endMask = 1 : i32, ellipsisMask = 0 : i32,
      newAxisMask = 0 : i32, shrinkAxisMask = 2 : i32
    } : !atir.tensor<?xi64>, !atir.tensor<4x3xi64>, !atir.tensor<2xi32>,
        !atir.tensor<2xi32, name = "end", data = dense<[0, 1]> : tensor<2xi32>>,
        !atir.tensor<2xi32, name = "stride", data = dense<1> : tensor<2xi32>>
        -> !atir.tensor<?xi64>
    %unique_values, %unique_indices = atir.Unique %slice :
      (!atir.tensor<?xi64>) -> (!atir.tensor<?xi64>, !atir.tensor<?xi32>)
    %unique2_values, %unique2_indices = atir.Unique %unique_values :
      (!atir.tensor<?xi64>) ->
      (!atir.tensor<?xi64>, !atir.tensor<?xi32>)
    %inner_buffer = "atir.buffer"() : () -> !atir.tensor<?x4xf32>
    %inner = atir.Gather %inner_buffer, %data, %unique2_values, %axis :
      !atir.tensor<?x4xf32>, !atir.tensor<2x4xf32>, !atir.tensor<?xi64>,
      !atir.tensor<i32, name = "axis", data = dense<0> : tensor<i32>>
      -> !atir.tensor<?x4xf32>
    %outer_buffer = "atir.buffer"() : () -> !atir.tensor<?x4xf32>
    %outer = atir.Gather %outer_buffer, %inner, %unique2_indices, %axis :
      !atir.tensor<?x4xf32>, !atir.tensor<?x4xf32>, !atir.tensor<?xi32>,
      !atir.tensor<i32, name = "axis", data = dense<0> : tensor<i32>>
      -> !atir.tensor<?x4xf32>
    return %unique_values, %unique_indices, %outer : !atir.tensor<?xi64>,
      !atir.tensor<?xi32>, !atir.tensor<?x4xf32>
  }
}
