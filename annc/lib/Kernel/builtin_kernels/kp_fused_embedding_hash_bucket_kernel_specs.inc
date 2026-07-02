ANNC_KERNEL(
    Name("KPFusedDnnEmbeddingWithHashBucket").Backend("aarch64").TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef1DString* input,
     AnncMemRef2DF32* embedding_weight,
     int64_t num_buckets),
    {
        const auto status = kp_fused_embedding_hash_bucket_aarch64_impl(
            ::annc::threadpool::getCurrentThreadPool(), output, input,
            embedding_weight, num_buckets);
        return ::annc::kernels::statusCode(status);
    })
