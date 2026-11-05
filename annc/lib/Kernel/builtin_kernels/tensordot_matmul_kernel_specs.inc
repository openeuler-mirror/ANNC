// Single context argument; tensor and attribute slots follow CustomOpSchema.

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_TensordotMatMul_float,
    Name("TensordotMatMul").Backend("aarch64").Abi("annc_context_v3")
        .TypeConstraint<float>("T"),
    {
      return ::annc::kernels::invokeContextKernel(context, tensordotMatMulImpl);
    })
