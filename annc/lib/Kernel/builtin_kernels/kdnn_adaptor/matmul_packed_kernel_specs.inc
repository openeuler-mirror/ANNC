ANNC_KERNEL(
    Name("MatMul").Backend("kdnn_packed").TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* lhs,
     AnncMemRef2DF32* rhs),
    {
        ::annc::kernels::kdnn_adaptor::matmul_kdnn_packed_impl(
            ::annc::threadpool::getCurrentThreadPool(),
            output,
            lhs,
            rhs);
    })
