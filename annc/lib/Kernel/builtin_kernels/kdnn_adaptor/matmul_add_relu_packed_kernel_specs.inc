// Context ABI: the implementation consumes named tensors and runtime services.

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_MatMulAddRelu_kdnn_packed_float,
    Name("MatMulAddRelu")
        .Backend("kdnn_packed")
        .TypeConstraint<float>("T")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, ::annc::kernels::kdnn_adaptor::matmul_add_relu_kdnn_packed_impl);
    })
