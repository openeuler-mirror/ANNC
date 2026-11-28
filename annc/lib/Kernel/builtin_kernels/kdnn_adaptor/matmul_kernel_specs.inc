// Context ABI: the implementation consumes named tensors and runtime services.

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_MatMul_kdnn_float,
    Name("MatMul")
        .Backend("kdnn")
        .TypeConstraint<float>("T")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, ::annc::kernels::kdnn_adaptor::matmul_kdnn_impl);
    })
