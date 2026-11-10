// RUN: annc-tf2atir-v2 %s | FileCheck %s
// Residue sink (V2) — complement-only policy. The artifact carries
// annc.residue carrier ops (the post-rename residue form); each opType is
// what the TF provider matches against the complement map.
//
// kOpMap now contains ONLY stablehlo-unexpressible capability ops. Cases:
//   1. kept op convention hit (SparseToDense: snake_case -> camelCase attr)
//   2. kept op number_attr sentinel (SparseSegmentMean -> numSegments fill)
//   3. deleted ops (StridedSlice/Equal/DynamicPartition/ParallelDynamicStitch)
//      fall to atir.unsupported(unknown_op) with full source metadata
//   4. formerly-mapped op without its contract attr flips from
//      spec_mismatch to unknown_op (Transpose: no mapping, no contract)
//   5. never-mapped op (Einsum) unchanged
//
// All tensors are ranked; the sink has no rank gate (unranked residues map
// straight through, see sink-layer.md).

// CHECK-LABEL: func @kept_convention
// SparseToDense is in the complement map; snake_case validate_indices lands
// on the declared camelCase validateIndices; unmatched attrs ride along.
// CHECK: "atir.sparse_to_dense_v2"(%[[A:.*]], %[[B:.*]], %[[C:.*]], %[[D:.*]]) <{validateIndices = false}> {call_target_name = "tf.SparseToDense", device = "/device:CPU:0"}
func.func @kept_convention(
    %arg0: tensor<2xi64>, %arg1: tensor<2xi64>, %arg2: tensor<f32>,
    %arg3: tensor<2xf32>) -> tensor<2x2xf32> {
  %0 = "annc.residue"(%arg0, %arg1, %arg2, %arg3) {call_target_name = "tf.SparseToDense", device = "/device:CPU:0", validate_indices = false} : (tensor<2xi64>, tensor<2xi64>, tensor<f32>, tensor<2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @sentinel_fill
// SparseSegmentMean (base form, 3 operands): the mirror expects numSegments.
// The verifier-reported arity contract (4) drives a scalar-0 sentinel fill —
// atir semantics: num_segments <= 0 infers the row count.
// CHECK: %[[Z:.*]] = arith.constant dense<0> : tensor<i32>
// CHECK: "atir.sparse_segment_mean_v2"(%[[A:.*]], %[[B:.*]], %[[C:.*]], %[[Z]]) {call_target_name = "tf.SparseSegmentMean", device = "/device:CPU:0", sparse_gradient = false} : (tensor<4x8xf32>, tensor<4xi32>, tensor<2xi32>, tensor<i32>) -> tensor<2x8xf32>
func.func @sentinel_fill(
    %arg0: tensor<4x8xf32>, %arg1: tensor<4xi32>,
    %arg2: tensor<2xi32>) -> tensor<2x8xf32> {
  %0 = "annc.residue"(%arg0, %arg1, %arg2) {call_target_name = "tf.SparseSegmentMean", device = "/device:CPU:0", sparse_gradient = false} : (tensor<4x8xf32>, tensor<4xi32>, tensor<2xi32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

// -----

// CHECK-LABEL: func @deleted_convention_fallback
// StridedSlice was mapped to strided_slice_v2 before batch 4; the carrier is
// gone, so the residue falls back with reason=unknown_op and the full source
// metadata preserved for the GraphDef rewrite.
// CHECK: atir.unsupported {detail = "", metadata = {begin_mask = 1 : i64, call_target_name = "tf.StridedSlice", device = "/device:CPU:0", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 2 : i64}, opType = "StridedSlice", reason = "unknown_op"}
func.func @deleted_convention_fallback(
    %arg0: tensor<4x10xf32>) -> tensor<2x4xf32> {
  %b = stablehlo.constant dense<[0, 1]> : tensor<2xi32>
  %e = stablehlo.constant dense<[2, 5]> : tensor<2xi32>
  %s = stablehlo.constant dense<[1, 1]> : tensor<2xi32>
  %0 = "annc.residue"(%arg0, %b, %e, %s) {call_target_name = "tf.StridedSlice", begin_mask = 1 : i64, device = "/device:CPU:0", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 2 : i64} : (tensor<4x10xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func @deleted_alias_fallback
// Equal used to alias to compare_v2; batch 4 removed the alias — fallback.
// CHECK: atir.unsupported {detail = "", metadata = {call_target_name = "tf.Equal", device = "/device:CPU:0", incompatible_shape_error = false}, opType = "Equal", reason = "unknown_op"}
func.func @deleted_alias_fallback(
    %arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
  %0 = "annc.residue"(%arg0, %arg1) {call_target_name = "tf.Equal", device = "/device:CPU:0", incompatible_shape_error = false} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: func @deleted_number_attr_fallback
// DynamicPartition's 12-result number_attr repair is gone with its carrier;
// metadata rides along verbatim.
// CHECK: atir.unsupported {detail = "", metadata = {call_target_name = "tf.DynamicPartition", device = "/device:CPU:0"}, opType = "DynamicPartition", reason = "unknown_op"}
func.func @deleted_number_attr_fallback(
    %arg0: tensor<4xf32>, %arg1: tensor<4xi32>) -> (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) {
  %0:12 = "annc.residue"(%arg0, %arg1) {call_target_name = "tf.DynamicPartition", device = "/device:CPU:0"} : (tensor<4xf32>, tensor<4xi32>) -> (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>)
  return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7, %0#8, %0#9, %0#10, %0#11 : tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>
}

// -----

// CHECK-LABEL: func @deleted_variadic_fallback
// ParallelDynamicStitch: variadic segment repair retired with its carrier.
// CHECK: atir.unsupported {detail = "", metadata = {call_target_name = "tf.ParallelDynamicStitch", device = "/device:CPU:0"}, opType = "ParallelDynamicStitch", reason = "unknown_op"}
func.func @deleted_variadic_fallback(
    %i0: tensor<2xi32>, %i1: tensor<2xi32>,
    %d0: tensor<4xf32>, %d1: tensor<4xf32>) -> tensor<8xf32> {
  %0 = "annc.residue"(%i0, %i1, %d0, %d1) {call_target_name = "tf.ParallelDynamicStitch", device = "/device:CPU:0"} : (tensor<2xi32>, tensor<2xi32>, tensor<4xf32>, tensor<4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// CHECK-LABEL: func @contract_flip_to_unknown
// Transpose lost its mapping AND its contract check: formerly spec_mismatch
// (missing permutation attr), now the plain unknown_op disposition.
// CHECK: atir.unsupported {detail = "", metadata = {call_target_name = "tf.Transpose", device = "/device:CPU:0"}, opType = "Transpose", reason = "unknown_op"}
func.func @contract_flip_to_unknown(
    %arg0: tensor<4x8xf32>, %arg1: tensor<2xi32>) -> tensor<8x4xf32> {
  %0 = "annc.residue"(%arg0, %arg1) {call_target_name = "tf.Transpose", device = "/device:CPU:0"} : (tensor<4x8xf32>, tensor<2xi32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// -----

// CHECK-LABEL: func @unknown_op_fallback
// Never-mapped op: unchanged by batch 4.
// CHECK: atir.unsupported {detail = {{.*}}, metadata = {call_target_name = "tf.Einsum", device = "/device:CPU:0", equation = "ab,bc->ac"}, opType = "Einsum", reason = "unknown_op"}
func.func @unknown_op_fallback(
    %arg0: tensor<4x8xf32>, %arg1: tensor<8x2xf32>) -> tensor<4x2xf32> {
  %0 = "annc.residue"(%arg0, %arg1) {call_target_name = "tf.Einsum", device = "/device:CPU:0", equation = "ab,bc->ac"} : (tensor<4x8xf32>, tensor<8x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// -----

// Stats behavior: mapped/fallback counters go to stderr (the RUN pipe only
// feeds stdout to FileCheck, so stats stay outside the assertion surface);
// per-op attribution is covered by the IR assertions above.
