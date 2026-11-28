// Context ABI: the implementation consumes named tensors and runtime services.

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedDnnEmbeddingWithHashBucket_float,
    Name("KPFusedDnnEmbeddingWithHashBucket")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kp_fused_embedding_hash_bucket_aarch64_impl);
    })
