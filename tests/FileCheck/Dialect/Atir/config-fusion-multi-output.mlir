// RUN: annc-opt %s -atir-config-fusion="config=%S/config-fusion-multi-output.json run-builtin-after=false" | FileCheck %s

// CHECK: func.func private @fused_topk_multi_output_
// CHECK-SAME: -> (!atir.tensor<4x3xf32, name = "TopK">, !atir.tensor<4x3xi32, name = "TopK:1">)
// CHECK: fusion.metadata
// CHECK: fusion.pattern = "topk_multi_output"
// CHECK: outputs = [{dtype = "f32"
// CHECK-SAME: tf_name = "TopK:0"}, {dtype = "i32"
// CHECK-SAME: tf_name = "TopK:1"}]
// CHECK-LABEL: func.func @main
// CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = call @fused_topk_multi_output_
// CHECK-NOT: "atir.TopK"
// CHECK: return %[[VALUES]], %[[INDICES]]
func.func @main(
    %input: !atir.tensor<4x8xf32, name = "input">,
    %k: !atir.tensor<i32, name = "k">)
    -> (!atir.tensor<4x3xf32, name = "TopK">,
        !atir.tensor<4x3xi32, name = "TopK:1">) {
  %values, %indices = "atir.TopK"(%input, %k) <{sorted = true}> : (
      !atir.tensor<4x8xf32, name = "input">,
      !atir.tensor<i32, name = "k">) -> (
      !atir.tensor<4x3xf32, name = "TopK">,
      !atir.tensor<4x3xi32, name = "TopK:1">)
  return %values, %indices : !atir.tensor<4x3xf32, name = "TopK">,
                             !atir.tensor<4x3xi32, name = "TopK:1">
}
