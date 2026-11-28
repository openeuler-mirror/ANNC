// Context ABI: the implementation consumes named tensors and runtime services.

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedSparseSegmentReduceI64Sum_float_int64_t,
    Name("KPFusedSparseSegmentReduceI64Sum")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("Tidx")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kpFusedSparseSegmentSumImpl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedSparseSegmentReduceI64Mean_float_int64_t,
    Name("KPFusedSparseSegmentReduceI64Mean")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("Tidx")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kpFusedSparseSegmentMeanImpl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedSparseSegmentReduceI32Sum_float_int32_t,
    Name("KPFusedSparseSegmentReduceI32Sum")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("Tidx")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kpFusedSparseSegmentSumImpl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedSparseSegmentReduceI32Mean_float_int32_t,
    Name("KPFusedSparseSegmentReduceI32Mean")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("Tidx")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kpFusedSparseSegmentMeanImpl);
    })
