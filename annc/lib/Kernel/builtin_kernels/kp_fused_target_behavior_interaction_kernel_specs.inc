ANNC_KERNEL(
    Name("KPFusedTargetBehaviorInteraction").Backend("aarch64")
        .TypeConstraint<float>("T"),
    (AnncMemRef3DF32* output,
     AnncMemRef3DF32* batch_input,
     AnncMemRef2DF32* weight,
     AnncMemRef1DF32* bias,
     AnncMemRef3DF32* tile_input),
    {
        const auto status = kp_fused_target_behavior_interaction_aarch64_impl(
            ::annc::threadpool::getCurrentThreadPool(), output, batch_input,
            weight, bias, tile_input);
        return ::annc::kernels::statusCode(status);
    })
