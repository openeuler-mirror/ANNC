// Context ABI: the implementation consumes named tensors and runtime services.

ANNC_CONTEXT_KERNEL_SPEC(
    ContextV3_KPFusedTargetBehaviorInteraction_float,
    Name("KPFusedTargetBehaviorInteraction")
        .Backend("aarch64")
        .TypeConstraint<float>("T")
        .Abi("annc_context_v3"),
    {
      return ::annc::kernels::invokeContextKernel(
          context, kp_fused_target_behavior_interaction_aarch64_impl);
    })
