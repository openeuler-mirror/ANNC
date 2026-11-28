// Single context argument; tensor and attribute slots follow CustomOpSchema.

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_BatchedMatMulAdd_float,
    Name("BatchedMatMulAdd").Backend("aarch64").Abi("annc_context_v3")
        .TypeConstraint<float>("T"),
    {
      return ::annc::kernels::invokeContextKernel(context, batchedMatMulAddImpl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_BatchedMatMulAdd4_float,
    Name("BatchedMatMulAdd4").Backend("aarch64").Abi("annc_context_v3")
        .TypeConstraint<float>("T"),
    {
      return ::annc::kernels::invokeContextKernel(context,
                                                  batchedMatMulAddRank4Impl);
    })
