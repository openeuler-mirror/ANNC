ANNC_KERNEL(
    Name("MatMul").Backend("kdnn").TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* lhs,
     AnncMemRef2DF32* rhs),
    {
        ::annc::kernels::kdnn_adaptor::matmul_kdnn_impl(
            ::annc::threadpool::getCurrentThreadPool(),
            lhs,
            rhs,
            output);
    })
