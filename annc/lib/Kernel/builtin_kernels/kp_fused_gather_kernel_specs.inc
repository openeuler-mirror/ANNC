// Context ABI: the implementation consumes named tensors and runtime services.

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedGather_float_int64_t_int32_t_int32_t,
    Name("KPFusedGather")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("Tkeys")
        .TypeConstraint<int32_t>("Tbegin")
        .TypeConstraint<int32_t>("Tindices")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kpFusedGatherImpl);
    })
