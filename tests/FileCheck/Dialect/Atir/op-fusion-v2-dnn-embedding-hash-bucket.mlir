// RUN: annc-opt %s -atir-op-fusion-v2 | FileCheck %s
// RUN: annc-opt %s -atir-op-fusion-v2 | annc-asm - --annc-fast-codegen | FileCheck %s --check-prefix=CODEGEN
//
// op-fusion-v2 pattern: dnn_embedding_hash_bucket (the V2 carrier port of
// the V1 KPFusedDnnEmbeddingWithHashBucket). The string domain behind the
// hash anchor stays opaque (the matcher only walks back to the string
// boundary); Unique -> embedding gather -> segment mean -> final reshape
// form the numeric tail. num_buckets rides context_i64_attrs.
// ABI = schema order (input, embedding_weight), output is the final
// reshape (block-order-last reshape of the segment result).
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 0 : i32}} {
  func.func @main() -> tensor<?x16xf32> attributes {tf.arg_names = [], tf.entry_function = {control_outputs = "", inputs = "", outputs = "out:0"}} {
    %s = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Placeholder", device = "", residue_reason = "non_numeric_type", shape = #tf_type.shape<*>}, opType = "Placeholder", reason = "unknown_op"} () : () -> tensor<?x!atir.string> loc(fused["Placeholder:", "input_str"])
    %w = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Placeholder", device = "", residue_reason = "no_legalize_pattern", shape = #tf_type.shape<[10000, 16]>}, opType = "Placeholder", reason = "unknown_op"} () : () -> tensor<10000x16xf32> loc(fused["Placeholder:", "embedding_weights"])
    %h = "atir.string_to_hash_bucket_fast_v2"(%s) <{numBuckets = 10000 : i64}> : (tensor<?x!atir.string>) -> tensor<?xi64> loc("hash_anchor")
    %u:2 = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Unique", device = "", residue_reason = "unregistered_custom_op"}, opType = "Unique", reason = "unknown_op"} (%h) : (tensor<?xi64>) -> (tensor<?xi64>, tensor<?xi64>) loc("uniq_node")
    %idx32 = stablehlo.convert %u#1 : (tensor<?xi64>) -> tensor<?xi32> loc("idx_conv")
    %emb = "stablehlo.torch_index_select"(%w, %u#0) <{batch_dims = 0 : i64, dim = 0 : i64}> : (tensor<10000x16xf32>, tensor<?xi64>) -> tensor<?x16xf32> loc("emb_gather")
    %nseg = stablehlo.constant dense<0> : tensor<i32> loc("nseg")
    %seg = "atir.sparse_segment_mean_v2"(%emb, %idx32, %idx32, %nseg) : (tensor<?x16xf32>, tensor<?xi32>, tensor<?xi32>, tensor<i32>) -> tensor<?x16xf32> loc("seg_mean")
    %shape = stablehlo.constant dense<[-1, 16]> : tensor<2xi32> loc("out_shape")
    %out = atir.unsupported {detail = "", metadata = {call_target_name = "tf.Reshape", device = "", residue_reason = "no_legalize_pattern"}, opType = "Reshape", reason = "unknown_op"} (%seg, %shape) : (tensor<?x16xf32>, tensor<2xi32>) -> tensor<?x16xf32> loc("out_reshape")
    return %out : tensor<?x16xf32>
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)

// CHECK: call @KPFusedDnnEmbeddingWithHashBucket(
// CHECK: func.func private @KPFusedDnnEmbeddingWithHashBucket
// CHECK-SAME: annc.kernel
// CHECK-SAME: abi = "annc_context_v3"
// CHECK-SAME: context_i64_attrs = [10000]
// CHECK-SAME: fusion.pattern = "dnn_embedding_hash_bucket"
// CHECK-SAME: kernel_name = "KPFusedDnnEmbeddingWithHashBucket"
// CHECK-SAME: tf_name = "out_reshape:0"
// CHECK: return

// CODEGEN: atir.Customize
// CODEGEN-SAME: custom.op_name = "KPFusedDnnEmbeddingWithHashBucket"
// CODEGEN: return
