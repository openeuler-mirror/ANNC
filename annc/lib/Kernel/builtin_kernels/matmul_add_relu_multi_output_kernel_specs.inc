// Context ABI: the implementation consumes named tensors and runtime services.

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_MatMulAddReluWithAddOutput_float,
    Name("MatMulAddReluWithAddOutput")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, matmulAddReluWithAddOutputImpl);
    })
