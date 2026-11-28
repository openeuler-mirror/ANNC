// Single context argument; tensor and attribute slots follow CustomOpSchema.

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_MaskedMul1_float,
    Name("MaskedMul1").Backend("aarch64").Abi("annc_context_v3")
        .TypeConstraint<float>("T").TypeConstraint<int64_t>("TMask"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, [](auto& view) {
            return maskedMulSiteImpl<1>(view, false);
          });
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_MaskedMul2_float,
    Name("MaskedMul2").Backend("aarch64").Abi("annc_context_v3")
        .TypeConstraint<float>("T").TypeConstraint<int64_t>("TMask"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, [](auto& view) {
            return maskedMulSiteImpl<2>(view, false);
          });
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_MaskedMul3_float,
    Name("MaskedMul3").Backend("aarch64").Abi("annc_context_v3")
        .TypeConstraint<float>("T").TypeConstraint<int64_t>("TMask"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, [](auto& view) {
            return maskedMulSiteImpl<3>(view, false);
          });
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_MaskedMul4_float,
    Name("MaskedMul4").Backend("aarch64").Abi("annc_context_v3")
        .TypeConstraint<float>("T").TypeConstraint<int64_t>("TMask"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, [](auto& view) {
            return maskedMulSiteImpl<4>(view, false);
          });
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_MaskedMul6_float,
    Name("MaskedMul6").Backend("aarch64").Abi("annc_context_v3")
        .TypeConstraint<float>("T").TypeConstraint<int64_t>("TMask"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, [](auto& view) {
            return maskedMulSiteImpl<6>(view, false);
          });
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_MaskedMul7_float,
    Name("MaskedMul7").Backend("aarch64").Abi("annc_context_v3")
        .TypeConstraint<float>("T").TypeConstraint<int64_t>("TMask"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, [](auto& view) {
            return maskedMulSiteImpl<7>(view, false);
          });
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_MaskedMul11_float,
    Name("MaskedMul11").Backend("aarch64").Abi("annc_context_v3")
        .TypeConstraint<float>("T").TypeConstraint<int64_t>("TMask"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, [](auto& view) {
            return maskedMulSiteImpl<11>(view, false);
          });
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_MaskedMul1Flat_float,
    Name("MaskedMul1Flat").Backend("aarch64").Abi("annc_context_v3")
        .TypeConstraint<float>("T").TypeConstraint<int64_t>("TMask"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, [](auto& view) {
            return maskedMulSiteImpl<1>(view, true);
          });
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_MaskedMul3Flat_float,
    Name("MaskedMul3Flat").Backend("aarch64").Abi("annc_context_v3")
        .TypeConstraint<float>("T").TypeConstraint<int64_t>("TMask"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, [](auto& view) {
            return maskedMulSiteImpl<3>(view, true);
          });
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_MaskedMul4Flat_float,
    Name("MaskedMul4Flat").Backend("aarch64").Abi("annc_context_v3")
        .TypeConstraint<float>("T").TypeConstraint<int64_t>("TMask"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, [](auto& view) {
            return maskedMulSiteImpl<4>(view, true);
          });
    })
