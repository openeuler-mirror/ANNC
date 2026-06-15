ANNC_KERNEL(
    Name("MatMul").Backend("aarch64").TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* lhs,
     AnncMemRef2DF32* rhs),
    { matmul_aarch64_impl(::annc::threadpool::getCurrentThreadPool(), lhs, rhs, output); })
