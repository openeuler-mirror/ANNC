// Context ABI: the implementation consumes named tensors and runtime services.

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedEmbeddingPaddingFast_float,
    Name("KPFusedEmbeddingPaddingFast")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kpFusedEmbeddingPaddingFastImpl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedEmbeddingPadding_float,
    Name("KPFusedEmbeddingPadding")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kpFusedEmbeddingPaddingImpl);
    })
