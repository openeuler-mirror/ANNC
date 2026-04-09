ANNC_KERNEL(
    Name("MatMul").Backend("aarch64"),
    (AnncMemRef2DF32* lhs,
     AnncMemRef2DF32* rhs,
     AnncMemRef2DF32* output),
    { matmul_aarch64_impl(::annc::runtime::getCurrentThreadPoolCtx(), lhs, rhs, output); })
