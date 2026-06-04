ANNC_KERNEL(
    Name("MatMul").Backend("kdnn"),
    (AnncMemRef2DF32* lhs,
     AnncMemRef2DF32* rhs,
     AnncMemRef2DF32* output),
    {
        ::annc::kernels::kdnn_adaptor::matmul_kdnn_impl(
            ::annc::threadpool::getCurrentThreadPool(),
            lhs,
            rhs,
            output);
    })
