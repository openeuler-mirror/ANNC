ANNC_KERNEL(
    Name("MatMulAdd").Backend("kdnn").TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* lhs,
     AnncMemRef2DF32* rhs,
     AnncMemRef1DF32* bias),
    {
        ::annc::kernels::kdnn_adaptor::matmul_add_kdnn_impl(
            ::annc::threadpool::getCurrentThreadPool(),
            output,
            lhs,
            rhs,
            bias);
    })
