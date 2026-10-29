// Single context argument; tensor and attribute slots follow CustomOpSchema.

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_LayerNorm_float,
    Name("LayerNorm").Backend("aarch64").Abi("annc_context_v3")
        .TypeConstraint<float>("T"),
    {
      return ::annc::kernels::invokeContextKernel(context, layerNormImpl);
    })
