// Single Context entry: tensor slots (x, variables...) and i64 attribute N.
ANNC_CONTEXT_KERNEL_SPEC(ContextV3_KPFusedSparseDynamicStitch_float,
                         Name("KPFusedSparseDynamicStitch")
                             .Backend("aarch64")
                             .TypeConstraint<float>("T")
                             .Abi("annc_context_v3"),
                         {
                           return ::annc::kernels::invokeContextKernel(
                               context, kp_fused_sparse_dynamic_stitch_impl);
                         })
