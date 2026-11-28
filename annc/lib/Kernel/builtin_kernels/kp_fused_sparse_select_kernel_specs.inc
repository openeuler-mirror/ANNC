// Context ABI: the implementation consumes named tensors and runtime services.

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedSparseSelect_int32_t,
    Name("KPFusedSparseSelect")
        .Backend("aarch64")
        .TypeConstraint<int32_t>("T")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kpFusedSparseSelectImpl);
    })
