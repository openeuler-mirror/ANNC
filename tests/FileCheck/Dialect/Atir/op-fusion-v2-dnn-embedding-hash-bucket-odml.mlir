// RUN: annc-opt %s -atir-op-fusion-v2 | FileCheck %s
// RUN: annc-opt %s -atir-op-fusion-v2 | annc-asm - --annc-fast-codegen | FileCheck %s --check-prefix=CODEGEN
//
// op-fusion-v2 pattern: dnn_embedding_hash_bucket over the real odml-frontend
// shape (wide_and_deep). Unlike the handcrafted ideal form, the wheel lowers
// the pre-hash filtering into computes (GreaterEqual compare, GatherV2
// torch_index_select) and keeps Where/Reshape/Unique as residues; the hash
// input arrives through the string transport (encode -> dynamic_gather ->
// decode, i.e. tf.GatherNd); SparseFillEmptyRows is a v2 carrier whose
// indicator/reverse outputs feed out-of-site consumers; num_segments is an
// arith constant; the trailing identity Reshape is folded away, so the
// segment mean itself is the site output.

#loc0 = loc(unknown)

module {
  func.func @main() -> tensor<?x64xf32> {
    // Weight read: dynamic boundary, not a closure member.
    %w = "atir.variable_v2"() <{sym_name = "v2_0"}> {annc.source_node = "embedding_weights/part_0"} : () -> tensor<10000x64xf32> loc(fused["VariableV2:", "embedding_weights/part_0"])

    // String input placeholder (the V1 ABI boundary, rank 1).
    %s = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Placeholder", device = "", residue_reason = "non_numeric_type"}, opType = "Placeholder", reason = "unknown_op"} () : () -> tensor<?x!atir.string> loc(fused["Placeholder:", "input_str"])

    // to_sparse_input: ExpandDims over the transport, then GatherNd
    // (encode -> dynamic_gather -> decode) selecting the non-empty rows.
    %empty = "atir.constant_v2"() <{sym_name = "const_0", values = [""]}> : () -> tensor<!atir.string> loc(#loc0)
    %ne = atir.unsupported {detail = "", metadata = {annc.source_op = "tf.NotEqual", call_target_name = "tf.NotEqual"}, opType = "NotEqual", reason = "unknown_op"} (%s, %empty) : (tensor<?x!atir.string>, tensor<!atir.string>) -> tensor<?xi1> loc(fused["NotEqual:", "to_sparse_input/NotEqual"])
    %rows = atir.unsupported {detail = "", metadata = {annc.source_op = "tf.Where", call_target_name = "tf.Where"}, opType = "Where", reason = "unknown_op"} (%ne) : (tensor<?xi1>) -> tensor<?xi64> loc(fused["Where:", "to_sparse_input/values"])

    %enc = "atir.encode_string_v2"(%s) {call_target_name = "annc.encodeString"} : (tensor<?x!atir.string>) -> tensor<?x?xi8> loc(#loc0)
    %eshp = stablehlo.constant dense<[1, -1]> : tensor<2xi32> loc(#loc0)
    %enc2 = atir.unsupported {detail = "", metadata = {annc.source_op = "tf.ExpandDims", call_target_name = "tf.ExpandDims"}, opType = "ExpandDims", reason = "unknown_op"} (%enc, %eshp) : (tensor<?x?xi8>, tensor<2xi32>) -> tensor<?x1x?xi8> loc(fused["ExpandDims:", "ExpandDims"])
    %rshp = stablehlo.constant dense<[-1, 2]> : tensor<2xi32> loc(#loc0)
    %rows2 = atir.unsupported {detail = "", metadata = {annc.source_op = "tf.Reshape", call_target_name = "tf.Reshape"}, opType = "Reshape", reason = "unknown_op"} (%rows, %rshp) : (tensor<?xi64>, tensor<2xi32>) -> tensor<?x2xi64> loc(fused["Reshape:", "to_sparse_input/indices"])
    %c1i = arith.constant 1 : index loc(#loc0)
    %c2i = arith.constant 2 : index loc(#loc0)
    %dim = tensor.dim %enc2, %c2i : tensor<?x1x?xi8> loc(#loc0)
    %fe = tensor.from_elements %c1i, %c1i, %dim : tensor<3xindex> loc(#loc0)
    %gsz = arith.index_cast %fe : tensor<3xindex> to tensor<3xi64> loc(#loc0)
    %gnd = "stablehlo.dynamic_gather"(%enc2, %rows2, %gsz) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>}> : (tensor<?x1x?xi8>, tensor<?x2xi64>, tensor<3xi64>) -> tensor<?x?xi8> loc(fused["GatherNd:", "to_sparse_input/values"])
    %sin = "atir.decode_string_v2"(%gnd) {call_target_name = "annc.decodeString"} : (tensor<?x?xi8>) -> tensor<?x!atir.string> loc(#loc0)

    // Hash anchor.
    %h = "atir.string_to_hash_bucket_fast_v2"(%sin) <{numBuckets = 10000 : i64}> : (tensor<?x!atir.string>) -> tensor<?xi64> loc(fused["StringToHashBucketFast:", "lookup"])

    // Pre-hash filtering as lowers it: compare + Where + Reshape residues,
    // then the two row-wise gathers feeding SparseFillEmptyRows.
    %neg1 = stablehlo.constant dense<-1> : tensor<i64> loc(#loc0)
    %hshape = shape.shape_of %h : tensor<?xi64> -> tensor<1xindex> loc(fused["GreaterEqual:", "weights/GreaterEqual"])
    %hs32 = arith.index_cast %hshape : tensor<1xindex> to tensor<1xi32> loc(fused["GreaterEqual:", "weights/GreaterEqual"])
    %negb = stablehlo.dynamic_broadcast_in_dim %neg1, %hs32, dims = [] : (tensor<i64>, tensor<1xi32>) -> tensor<?xi64> loc(fused["GreaterEqual:", "weights/GreaterEqual"])
    %ge = stablehlo.compare  GE, %h, %negb : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi1> loc(fused["GreaterEqual:", "weights/GreaterEqual"])
    %wi = atir.unsupported {detail = "", metadata = {annc.source_op = "tf.Where", call_target_name = "tf.Where"}, opType = "Where", reason = "unknown_op"} (%ge) : (tensor<?xi1>) -> tensor<?x1xi64> loc(fused["Where:", "weights/Where"])
    %shp1 = stablehlo.constant dense<[-1]> : tensor<1xi32> loc(#loc0)
    %wi1 = atir.unsupported {detail = "", metadata = {annc.source_op = "tf.Reshape", call_target_name = "tf.Reshape"}, opType = "Reshape", reason = "unknown_op"} (%wi, %shp1) : (tensor<?x1xi64>, tensor<1xi32>) -> tensor<?xi64> loc(fused["Reshape:", "weights/Reshape"])

    %g1 = "stablehlo.torch_index_select"(%rows2, %wi1) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<?x2xi64>, tensor<?xi64>) -> tensor<?x2xi64> loc(fused["GatherV2:", "weights/GatherV2_1"])
    %g2 = "stablehlo.torch_index_select"(%h, %wi1) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64> loc(fused["GatherV2:", "weights/GatherV2_2"])

    %denseShape = stablehlo.constant dense<[3, 2]> : tensor<2xi64> loc(#loc0)
    %fi, %fv, %ind, %rev = "atir.sparse_fill_empty_rows_v2"(%g1, %g2, %denseShape, %neg1) : (tensor<?x2xi64>, tensor<?xi64>, tensor<2xi64>, tensor<i64>) -> (tensor<?x2xi64>, tensor<?xi64>, tensor<?xi1>, tensor<?xi64>) loc(fused["SparseFillEmptyRows:", "weights/SparseFillEmptyRows"])

    // The indicator feeds an out-of-site consumer (zeros_like scaffolding
    // stays in main; the outline clones the carrier for the kernel).
    %indshp = stablehlo.constant dense<[3, 1]> : tensor<2xi32> loc(#loc0)
    %ind2 = atir.unsupported {detail = "", metadata = {annc.source_op = "tf.Reshape", call_target_name = "tf.Reshape"}, opType = "Reshape", reason = "unknown_op"} (%ind, %indshp) : (tensor<?xi1>, tensor<2xi32>) -> tensor<3x1xi1> loc(fused["Reshape:", "weights/Reshape_1"])

    %u:2 = atir.unsupported {detail = "", metadata = {annc.source_op = "tf.Unique", call_target_name = "tf.Unique"}, opType = "Unique", reason = "unknown_op"} (%fv) : (tensor<?xi64>) -> (tensor<?xi64>, tensor<?xi32>) loc(fused["Unique:", "embedding_lookup_sparse/Unique"])

    %g = "stablehlo.torch_index_select"(%w, %u#0) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<10000x64xf32>, tensor<?xi64>) -> tensor<?x64xf32> loc(fused["GatherV2:", "embedding_lookup_sparse/embedding_lookup"])

    // Segment indices: cast chain from the filled indices (StridedSlice
    // collapses to a convert in the minimal form).
    %segidx = stablehlo.convert %fi : (tensor<?x2xi64>) -> tensor<?x2xi32> loc(fused["Cast:", "cast"])
    %segidx2:2 = "atir.sparse_reshape_v2"(%segidx, %denseShape, %denseShape) : (tensor<?x2xi32>, tensor<2xi64>, tensor<2xi64>) -> (tensor<?xi32>, tensor<2xi64>) loc(fused["SparseReshape:", "weights/SparseReshape"])

    // num_segments rides an arith constant in the odml output.
    %nseg = arith.constant dense<0> : tensor<i32> loc(#loc0)
    %seg = "atir.sparse_segment_mean_v2"(%g, %u#1, %segidx2#0, %nseg) : (tensor<?x64xf32>, tensor<?xi32>, tensor<?xi32>, tensor<i32>) -> tensor<?x64xf32> loc(fused["SparseSegmentMean:", "embedding_lookup_sparse"])

    // No trailing Reshape here: the odml frontend folds it away, the
    // segment mean result is the site output directly.
    return %seg : tensor<?x64xf32>
  }
}

// The main body keeps the call (rewriting the site in place) and no
// anchor/carrier ops survive in it.
// CHECK: call @KPFusedDnnEmbeddingWithHashBucket(
// CHECK-NOT: atir.string_to_hash_bucket_fast_v2
// CHECK-NOT: sparse_fill_empty_rows_v2
// CHECK: return

// The outlined kernel function follows the main body.
// CHECK: func.func private @KPFusedDnnEmbeddingWithHashBucket(
// CHECK-SAME: %arg0: tensor<?x!atir.string>, %arg1: tensor<10000x64xf32>
// CHECK-SAME: annc.kernel
// CHECK-SAME: abi = "annc_context_v3"
// CHECK-SAME: context_i64_attrs = [10000]
// CHECK-SAME: fusion.pattern = "dnn_embedding_hash_bucket"
// CHECK-SAME: kernel_name = "KPFusedDnnEmbeddingWithHashBucket"

// CODEGEN: atir.Customize
// CODEGEN-SAME: custom.op_name = "KPFusedDnnEmbeddingWithHashBucket"
// CODEGEN: return
