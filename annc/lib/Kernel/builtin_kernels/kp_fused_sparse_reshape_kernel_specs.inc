// Context ABI: the implementation consumes named tensors and runtime services.

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedSparseReshapeI64_int64_t,
    Name("KPFusedSparseReshapeI64")
        .Backend("aarch64")
        .TypeConstraint<int64_t>("T")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kpFusedSparseReshapeImpl);
    })
