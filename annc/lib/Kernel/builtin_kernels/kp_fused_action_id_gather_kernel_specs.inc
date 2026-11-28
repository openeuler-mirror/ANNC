// Context ABI: the implementation consumes named tensors and runtime services.

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedEmbeddingActionIdGatherR1x1_float_int64_t_int32_t,
    Name("KPFusedEmbeddingActionIdGatherR1x1")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("T1")
        .TypeConstraint<int32_t>("T2")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kp_fused_action_id_gather_impl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedEmbeddingActionIdGatherR2x1_float_int64_t_int32_t,
    Name("KPFusedEmbeddingActionIdGatherR2x1")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("T1")
        .TypeConstraint<int32_t>("T2")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kp_fused_action_id_gather_impl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedEmbeddingActionIdGatherR1x2_float_int64_t_int32_t,
    Name("KPFusedEmbeddingActionIdGatherR1x2")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("T1")
        .TypeConstraint<int32_t>("T2")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kp_fused_action_id_gather_impl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedEmbeddingActionIdGatherR2x2_float_int64_t_int32_t,
    Name("KPFusedEmbeddingActionIdGatherR2x2")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("T1")
        .TypeConstraint<int32_t>("T2")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kp_fused_action_id_gather_impl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedEmbeddingActionIdGatherR1x1_float_int32_t_int32_t,
    Name("KPFusedEmbeddingActionIdGatherR1x1")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("T1")
        .TypeConstraint<int32_t>("T2")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kp_fused_action_id_gather_impl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedEmbeddingActionIdGatherR2x1_float_int32_t_int32_t,
    Name("KPFusedEmbeddingActionIdGatherR2x1")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("T1")
        .TypeConstraint<int32_t>("T2")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kp_fused_action_id_gather_impl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedEmbeddingActionIdGatherR1x2_float_int32_t_int32_t,
    Name("KPFusedEmbeddingActionIdGatherR1x2")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("T1")
        .TypeConstraint<int32_t>("T2")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kp_fused_action_id_gather_impl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedEmbeddingActionIdGatherR2x2_float_int32_t_int32_t,
    Name("KPFusedEmbeddingActionIdGatherR2x2")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("T1")
        .TypeConstraint<int32_t>("T2")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kp_fused_action_id_gather_impl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedEmbeddingActionIdGatherR1x1_float_int64_t_int64_t,
    Name("KPFusedEmbeddingActionIdGatherR1x1")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("T1")
        .TypeConstraint<int64_t>("T2")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kp_fused_action_id_gather_impl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedEmbeddingActionIdGatherR2x1_float_int64_t_int64_t,
    Name("KPFusedEmbeddingActionIdGatherR2x1")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("T1")
        .TypeConstraint<int64_t>("T2")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kp_fused_action_id_gather_impl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedEmbeddingActionIdGatherR1x2_float_int64_t_int64_t,
    Name("KPFusedEmbeddingActionIdGatherR1x2")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("T1")
        .TypeConstraint<int64_t>("T2")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kp_fused_action_id_gather_impl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedEmbeddingActionIdGatherR2x2_float_int64_t_int64_t,
    Name("KPFusedEmbeddingActionIdGatherR2x2")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("T1")
        .TypeConstraint<int64_t>("T2")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kp_fused_action_id_gather_impl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedEmbeddingActionIdGatherR1x1_float_int32_t_int64_t,
    Name("KPFusedEmbeddingActionIdGatherR1x1")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("T1")
        .TypeConstraint<int64_t>("T2")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kp_fused_action_id_gather_impl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedEmbeddingActionIdGatherR2x1_float_int32_t_int64_t,
    Name("KPFusedEmbeddingActionIdGatherR2x1")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("T1")
        .TypeConstraint<int64_t>("T2")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kp_fused_action_id_gather_impl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedEmbeddingActionIdGatherR1x2_float_int32_t_int64_t,
    Name("KPFusedEmbeddingActionIdGatherR1x2")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("T1")
        .TypeConstraint<int64_t>("T2")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kp_fused_action_id_gather_impl);
    })

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedEmbeddingActionIdGatherR2x2_float_int32_t_int64_t,
    Name("KPFusedEmbeddingActionIdGatherR2x2")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("T1")
        .TypeConstraint<int64_t>("T2")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kp_fused_action_id_gather_impl);
    })
