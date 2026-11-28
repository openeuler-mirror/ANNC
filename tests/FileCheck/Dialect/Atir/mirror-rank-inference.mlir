// RUN: annc-opt --split-input-file %s --atir-rank-inference | FileCheck %s

// Rank inference across control-flow mirror ops (atir.switch / atir.merge /
// atir.enter / atir.exit / atir.next_iteration / atir.loop_cond): the mirror
// verifiers keep data/output types identical, so the pass never sees mixed
// states inside an op. Its role is cross-op propagation: a consumer-side
// rank pin on one side must flow through the mirror op to the data input
// and sibling outputs.

// -----

// CHECK-LABEL: func.func @switch_consumer_pin(
// CHECK-SAME: !atir.tensor<?xf32>) -> (!atir.tensor<?xf32>, !atir.tensor<?xi32>)
func.func @switch_consumer_pin(%arg0: !atir.tensor<*xf32, rank_known = false>)
    -> (!atir.tensor<*xf32, rank_known = false>,
        !atir.tensor<*xi32, rank_known = false>) {
  %pred = "atir.buffer"() : () -> !atir.tensor<i32, encoding = <"bool">>
  // CHECK: atir.switch
  // CHECK-SAME: -> (!atir.tensor<?xf32>, !atir.tensor<?xf32>)
  %f, %t = atir.switch %arg0, %pred
      : (!atir.tensor<*xf32, rank_known = false>,
         !atir.tensor<i32, encoding = <"bool">>)
        -> (!atir.tensor<*xf32, rank_known = false>,
            !atir.tensor<*xf32, rank_known = false>)
  %y, %idx = atir.Unique %t
      : (!atir.tensor<*xf32, rank_known = false>)
        -> (!atir.tensor<*xf32, rank_known = false>,
            !atir.tensor<*xi32, rank_known = false>)
  return %y, %idx
      : !atir.tensor<*xf32, rank_known = false>,
        !atir.tensor<*xi32, rank_known = false>
}

// -----

// Merge equality spans all inputs and the value result; value_index (rank-0
// i32) is untouched.
// CHECK-LABEL: func.func @merge_consumer_pin(
// CHECK-SAME: !atir.tensor<?xf32>{{.*}}) -> !atir.tensor<i32>
func.func @merge_consumer_pin(%a: !atir.tensor<*xf32, rank_known = false>,
                              %b: !atir.tensor<*xf32, rank_known = false>)
    -> !atir.tensor<i32> {
  // CHECK: atir.merge
  // CHECK-SAME: -> (!atir.tensor<?xf32>, !atir.tensor<i32>)
  %v, %idx = atir.merge %a, %b
      : (!atir.tensor<*xf32, rank_known = false>,
         !atir.tensor<*xf32, rank_known = false>)
        -> (!atir.tensor<*xf32, rank_known = false>, !atir.tensor<i32>)
  %y, %u = atir.Unique %v
      : (!atir.tensor<*xf32, rank_known = false>)
        -> (!atir.tensor<*xf32, rank_known = false>,
            !atir.tensor<*xi32, rank_known = false>)
  return %idx : !atir.tensor<i32>
}

// -----

// Enter/NextIteration/Exit passthrough chain: one rank pin upgrades the
// whole chain and the function argument.
// CHECK-LABEL: func.func @loop_chain_pin(
// CHECK-SAME: !atir.tensor<?xf32>) -> (!atir.tensor<?xf32>, !atir.tensor<?xi32>)
func.func @loop_chain_pin(%arg0: !atir.tensor<*xf32, rank_known = false>)
    -> (!atir.tensor<*xf32, rank_known = false>,
        !atir.tensor<*xi32, rank_known = false>) {
  // CHECK: atir.enter
  // CHECK-SAME: -> !atir.tensor<?xf32>
  %en = atir.enter %arg0 : (!atir.tensor<*xf32, rank_known = false>) -> !atir.tensor<*xf32, rank_known = false>
  // CHECK: atir.next_iteration
  // CHECK-SAME: -> !atir.tensor<?xf32>
  %nx = atir.next_iteration %en : (!atir.tensor<*xf32, rank_known = false>) -> !atir.tensor<*xf32, rank_known = false>
  // CHECK: atir.exit
  // CHECK-SAME: -> !atir.tensor<?xf32>
  %ex = atir.exit %nx : (!atir.tensor<*xf32, rank_known = false>) -> !atir.tensor<*xf32, rank_known = false>
  %y, %u = atir.Unique %ex
      : (!atir.tensor<*xf32, rank_known = false>)
        -> (!atir.tensor<*xf32, rank_known = false>,
            !atir.tensor<*xi32, rank_known = false>)
  // CHECK: atir.Unique
  // CHECK-SAME: -> (!atir.tensor<?xf32>, !atir.tensor<?xi32>)
  return %y, %u
      : !atir.tensor<*xf32, rank_known = false>,
        !atir.tensor<*xi32, rank_known = false>
}

// -----

// No pin anywhere: everything conservatively stays unknown.
// CHECK-LABEL: func.func @mirror_no_pin_stays_unknown(
// CHECK-SAME: !atir.tensor<*xf32, rank_known = false>{{.*}} -> !atir.tensor<*xf32, rank_known = false>
func.func @mirror_no_pin_stays_unknown(
    %arg0: !atir.tensor<*xf32, rank_known = false>)
    -> !atir.tensor<*xf32, rank_known = false> {
  %pred = "atir.buffer"() : () -> !atir.tensor<i32, encoding = <"bool">>
  // CHECK: atir.switch
  // CHECK-SAME: -> (!atir.tensor<*xf32, rank_known = false>, !atir.tensor<*xf32, rank_known = false>)
  %f, %t = atir.switch %arg0, %pred
      : (!atir.tensor<*xf32, rank_known = false>,
         !atir.tensor<i32, encoding = <"bool">>)
        -> (!atir.tensor<*xf32, rank_known = false>,
            !atir.tensor<*xf32, rank_known = false>)
  return %f : !atir.tensor<*xf32, rank_known = false>
}
