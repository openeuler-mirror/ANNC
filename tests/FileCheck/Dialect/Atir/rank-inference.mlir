// RUN: annc-opt --split-input-file %s --atir-rank-inference | FileCheck %s

// Rank inference: upgrade rank_known=false tensors from TF op semantics
// (whitelist) and consumer-side constraints.  Upgraded types print as
// <?x...xT> (the * and rank_known = false disappear); values with no
// hard guarantee stay untouched.

// -----

// Unique pins its input and both outputs to rank 1 (TF shape fn asserts
// WithRank(input, 1)).
// CHECK-LABEL: func.func @unique(
// CHECK-SAME: !atir.tensor<?xf32>) -> (!atir.tensor<?xf32>, !atir.tensor<?xi32>)
func.func @unique(%arg0: !atir.tensor<*xf32, rank_known = false>)
    -> (!atir.tensor<*xf32, rank_known = false>,
        !atir.tensor<*xi32, rank_known = false>) {
  %y, %idx = atir.Unique %arg0
      : (!atir.tensor<*xf32, rank_known = false>)
        -> (!atir.tensor<*xf32, rank_known = false>,
            !atir.tensor<*xi32, rank_known = false>)
  // CHECK: atir.Unique %arg0
  // CHECK-SAME: (!atir.tensor<?xf32>) -> (!atir.tensor<?xf32>, !atir.tensor<?xi32>)
  return %y, %idx
      : !atir.tensor<*xf32, rank_known = false>,
        !atir.tensor<*xi32, rank_known = false>
  // CHECK: return
  // CHECK-NOT: rank_known = false
}

// -----

// StridedSlice rank equation propagates both ways: TF guarantees
// result = input + popcount(newAxisMask) - popcount(shrinkAxisMask), so a
// known rank-3 result pins the unknown input to 3 - 1 + 0 = 2.  The begin
// length (3 here, != 2) is deliberately ignored: TF implies an ellipsis at
// the end of the spec, so begin length is only a lower bound on the input
// rank.  The shared T dtype pins the input's `!atir.unknown` element type to
// i64.  The opaque input side (%arg0) has no constraint and stays unknown.
// CHECK-LABEL: func.func @strided_slice(
// CHECK-SAME: !atir.tensor<*xf32, rank_known = false>) -> !atir.tensor<?x?x?xi64>
func.func @strided_slice(%arg0: !atir.tensor<*xf32, rank_known = false>)
    -> !atir.tensor<?x?x?xi64> {
  %keys = atir.opaque (%arg0) {opType = "ParseActionAggregationAddTimeAndAttribute"}
      : (!atir.tensor<*xf32, rank_known = false>)
        -> !atir.tensor<*x!atir.unknown, rank_known = false>
  // CHECK: atir.opaque(%arg0)
  // CHECK-SAME: (!atir.tensor<*xf32, rank_known = false>) -> !atir.tensor<?x?xi64>
  %buf = "atir.buffer"() : () -> !atir.tensor<*xi64, rank_known = false>
  %begin = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "begin">
  %end = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "end">
  %strides = "atir.buffer"() : () -> !atir.tensor<3xi32, name = "strides">
  %slice = atir.StridedSlice %buf, %keys, %begin, %end, %strides
      {newAxisMask = 2 : i32}
      : <*xi64, rank_known = false>, <*x!atir.unknown, rank_known = false>,
        <3xi32, name = "begin">, <3xi32, name = "end">, <3xi32, name = "strides">
        -> <?x?x?xi64>
  // CHECK: atir.StridedSlice
  // CHECK-SAME: <?x?x?xi64>, <?x?xi64>,
  // CHECK-SAME: <3xi32, name = "begin">, <3xi32, name = "end">, <3xi32, name = "strides"> -> <?x?x?xi64>
  return %slice : !atir.tensor<?x?x?xi64>
}

// -----

// Begin shorter than the input rank is legal TF (implied trailing ellipsis:
// tf.strided_slice on a rank-3 tensor with begin=[1], end=[5] slices only
// dim 0), so the begin vector length is not a rank guarantee.  With both
// sides unknown nothing pins the ranks; the T dtype contract still pins the
// unknown result element type back to the input's f32.
// CHECK-LABEL: func.func @short_begin(
// CHECK-SAME: !atir.tensor<*xf32, rank_known = false>) -> !atir.tensor<*xf32, rank_known = false>
func.func @short_begin(%arg0: !atir.tensor<*xf32, rank_known = false>)
    -> !atir.tensor<*x!atir.unknown, rank_known = false> {
  %buf = "atir.buffer"() : () -> !atir.tensor<*x!atir.unknown, rank_known = false>
  %begin = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "begin">
  %end = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "end">
  %strides = "atir.buffer"() : () -> !atir.tensor<1xi32, name = "strides">
  %slice = atir.StridedSlice %buf, %arg0, %begin, %end, %strides
      : <*x!atir.unknown, rank_known = false>, <*xf32, rank_known = false>,
        <1xi32, name = "begin">, <1xi32, name = "end">, <1xi32, name = "strides">
        -> <*x!atir.unknown, rank_known = false>
  // CHECK: atir.StridedSlice
  // CHECK-SAME: <*x!atir.unknown, rank_known = false>, <*xf32, rank_known = false>,
  // CHECK-SAME: <1xi32, name = "begin">, <1xi32, name = "end">, <1xi32, name = "strides"> -> <*xf32, rank_known = false>
  return %slice : !atir.tensor<*x!atir.unknown, rank_known = false>
}

// -----

// SparseSegmentSum contract: TF accepts N-D data (rank 1, 2, 3 all verified
// legal on TF 2.20) and the result keeps the data rank, so a known rank-2
// result pins the data to rank 2 via the shared rank (never a hardcoded 2);
// indices=1, segmentIds=1, numSegments scalar (plus the output-buffer class).
// CHECK-LABEL: func.func @sparse_segment_sum(
// CHECK-SAME: %arg0: !atir.tensor<?x?xf32>, %arg1: !atir.tensor<?xi64>, %arg2: !atir.tensor<?xi32>) -> !atir.tensor<?x?xf32>
func.func @sparse_segment_sum(
    %data: !atir.tensor<*xf32, rank_known = false>,
    %indices: !atir.tensor<*xi64, rank_known = false>,
    %segids: !atir.tensor<*xi32, rank_known = false>)
    -> !atir.tensor<?x?xf32> {
  %buf = "atir.buffer"() : () -> !atir.tensor<*xf32, rank_known = false>
  %num = "atir.buffer"() : () -> !atir.tensor<*xi32, rank_known = false>
  %out = atir.SparseSegmentSum %buf, %data, %indices, %segids, %num
      : <*xf32, rank_known = false>, <*xf32, rank_known = false>,
        <*xi64, rank_known = false>, <*xi32, rank_known = false>,
        <*xi32, rank_known = false> -> <?x?xf32>
  // CHECK: atir.SparseSegmentSum
  // CHECK-SAME: <?x?xf32>, <?x?xf32>, <?xi64>, <?xi32>, <i32> -> <?x?xf32>
  return %out : !atir.tensor<?x?xf32>
  // CHECK: return
  // CHECK-NOT: rank_known = false
}

// -----

// SparseSegmentSum data is N-D in TF (rank-1 verified legal): the shared
// data/result rank pins rank 1 here, proving the rule does not hardcode 2.
// CHECK-LABEL: func.func @ssr_rank1(
// CHECK-SAME: %arg0: !atir.tensor<?xf32>) -> !atir.tensor<?xf32>
func.func @ssr_rank1(%data: !atir.tensor<*xf32, rank_known = false>)
    -> !atir.tensor<?xf32> {
  %buf = "atir.buffer"() : () -> !atir.tensor<*xf32, rank_known = false>
  %ind = "atir.buffer"() : () -> !atir.tensor<*xi64, rank_known = false>
  %seg = "atir.buffer"() : () -> !atir.tensor<*xi32, rank_known = false>
  %num = "atir.buffer"() : () -> !atir.tensor<*xi32, rank_known = false>
  %out = atir.SparseSegmentSum %buf, %data, %ind, %seg, %num
      : <*xf32, rank_known = false>, <*xf32, rank_known = false>,
        <*xi64, rank_known = false>, <*xi32, rank_known = false>,
        <*xi32, rank_known = false> -> <?xf32>
  // CHECK: atir.SparseSegmentSum
  // CHECK-SAME: <?xf32>, <?xf32>, <?xi64>, <?xi32>, <i32> -> <?xf32>
  return %out : !atir.tensor<?xf32>
}

// -----

// Pack: inputs share rank r and the result is r+1.  The output buffer
// operand carries the RESULT's rank and must not drag the inputs up to it -
// inputs are pinned from a known input or from result-1.
// CHECK-LABEL: func.func @pack(
// CHECK-SAME: %arg0: !atir.tensor<?xf32>) -> !atir.tensor<2x?xf32>
func.func @pack(%a: !atir.tensor<*xf32, rank_known = false>)
    -> !atir.tensor<2x?xf32> {
  %buf = "atir.buffer"() : () -> !atir.tensor<2x?xf32>
  %b = "atir.buffer"() : () -> !atir.tensor<?xf32>
  %p = atir.Pack %buf, (%a, %b) axis = 0
      : <2x?xf32>, !atir.tensor<*xf32, rank_known = false>,
        !atir.tensor<?xf32> -> <2x?xf32>
  // CHECK: atir.Pack
  // CHECK-SAME: <2x?xf32>, !atir.tensor<?xf32>, !atir.tensor<?xf32> -> <2x?xf32>
  return %p : !atir.tensor<2x?xf32>
}

// -----

// Conflicting constraints leave the value unknown: %v is pinned to rank 1 by
// Unique and to rank 2 by the SparseSegmentSum data/result shared rank (the
// result is a known serving fact here).  The other side effects still fire
// (Unique outputs, ss buffer/indices/segmentIds/numSegments).
// CHECK-LABEL: func.func @conflict
func.func @conflict()
    -> (!atir.tensor<?x?xf32>,
        !atir.tensor<*xi32, rank_known = false>) {
  %v = atir.opaque {opType = "Mystery"}
      : () -> !atir.tensor<*xf32, rank_known = false>
  // CHECK: atir.opaque
  // CHECK-SAME: !atir.tensor<*xf32, rank_known = false>
  %y, %idx = atir.Unique %v
      : (!atir.tensor<*xf32, rank_known = false>)
        -> (!atir.tensor<*xf32, rank_known = false>,
            !atir.tensor<*xi32, rank_known = false>)
  // CHECK: atir.Unique
  // CHECK-SAME: (!atir.tensor<*xf32, rank_known = false>) -> (!atir.tensor<?xf32>, !atir.tensor<?xi32>)
  %buf = "atir.buffer"() : () -> !atir.tensor<*xf32, rank_known = false>
  %ind = "atir.buffer"() : () -> !atir.tensor<*xi64, rank_known = false>
  %seg = "atir.buffer"() : () -> !atir.tensor<*xi32, rank_known = false>
  %num = "atir.buffer"() : () -> !atir.tensor<*xi32, rank_known = false>
  %out = atir.SparseSegmentSum %buf, %v, %ind, %seg, %num
      : <*xf32, rank_known = false>, <*xf32, rank_known = false>,
        <*xi64, rank_known = false>, <*xi32, rank_known = false>,
        <*xi32, rank_known = false> -> <?x?xf32>
  // CHECK: atir.SparseSegmentSum
  // CHECK-SAME: <?x?xf32>, <*xf32, rank_known = false>, <?xi64>, <?xi32>, <i32> -> <?x?xf32>
  return %out, %idx
      : !atir.tensor<?x?xf32>,
        !atir.tensor<*xi32, rank_known = false>
}

// -----

// Bidirectional propagation through the same-shape Cast chain: MatMul pins
// its lhs to rank 2, the equality classes walk the rank back to the func
// argument, and the func signature (args and result) is rebuilt.
// CHECK-LABEL: func.func @cast_chain(
// CHECK-SAME: !atir.tensor<?x?xf32>) -> !atir.tensor<?x?xf32>
func.func @cast_chain(%a: !atir.tensor<*xf32, rank_known = false>)
    -> !atir.tensor<*xf32, rank_known = false> {
  %b1 = "atir.buffer"() : () -> !atir.tensor<*xf32, rank_known = false>
  %c1 = atir.Cast %b1, %a
      : <*xf32, rank_known = false>, <*xf32, rank_known = false>
        -> <*xf32, rank_known = false>
  %b2 = "atir.buffer"() : () -> !atir.tensor<*xf32, rank_known = false>
  %c2 = atir.Cast %b2, %c1
      : <*xf32, rank_known = false>, <*xf32, rank_known = false>
        -> <*xf32, rank_known = false>
  %mb = "atir.buffer"() : () -> !atir.tensor<*xf32, rank_known = false>
  %w = "atir.buffer"() : () -> !atir.tensor<2x2xf32, name = "w">
  %m = "atir.MatMul"(%mb, %c2, %w) <{withBias = false}>
      : (!atir.tensor<*xf32, rank_known = false>,
         !atir.tensor<*xf32, rank_known = false>,
         !atir.tensor<2x2xf32, name = "w">)
        -> !atir.tensor<*xf32, rank_known = false>
  // CHECK: atir.Cast
  // CHECK-SAME: <?x?xf32>, <?x?xf32> -> <?x?xf32>
  // CHECK: "atir.MatMul"
  // CHECK-SAME: <?x?xf32>
  return %m : !atir.tensor<*xf32, rank_known = false>
  // CHECK: return
  // CHECK-NOT: rank_known = false
}

// -----

// Shape/Size/Rank whitelist outputs and Reshape's targetShape length pin the
// result.  The Shape/Rank/Reshape input itself has no constraint and stays
// unknown.
// CHECK-LABEL: func.func @misc
func.func @misc(%arg0: !atir.tensor<*xf32, rank_known = false>)
    -> (!atir.tensor<*xf32, rank_known = false>,
        !atir.tensor<*xi32, rank_known = false>,
        !atir.tensor<*xi32, rank_known = false>) {
  %shape = atir.Shape %arg0
      : (!atir.tensor<*xf32, rank_known = false>)
        -> !atir.tensor<*xi32, rank_known = false>
  // CHECK: atir.Shape %arg0
  // CHECK-SAME: (!atir.tensor<*xf32, rank_known = false>) -> !atir.tensor<?xi32>
  %rank = atir.Rank %arg0
      : (!atir.tensor<*xf32, rank_known = false>)
        -> !atir.tensor<*xi32, rank_known = false>
  // CHECK: atir.Rank %arg0
  // CHECK-SAME: (!atir.tensor<*xf32, rank_known = false>) -> !atir.tensor<i32>
  %rb = "atir.buffer"() : () -> !atir.tensor<*xf32, rank_known = false>
  %ts = "atir.buffer"() : () -> !atir.tensor<2xi32, name = "ts">
  %r = atir.Reshape %rb, %arg0, %ts
      : <*xf32, rank_known = false>, <*xf32, rank_known = false>,
        <2xi32, name = "ts"> -> <*xf32, rank_known = false>
  // CHECK: atir.Reshape
  // CHECK-SAME: <?x?xf32>, <*xf32, rank_known = false>, <2xi32, name = "ts"> -> <?x?xf32>
  return %r, %shape, %rank
      : !atir.tensor<*xf32, rank_known = false>,
        !atir.tensor<*xi32, rank_known = false>,
        !atir.tensor<*xi32, rank_known = false>
}
