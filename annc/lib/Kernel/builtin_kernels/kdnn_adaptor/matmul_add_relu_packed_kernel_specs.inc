ANNC_KERNEL(
    Name("MatMulAddRelu").Backend("kdnn_packed").TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* lhs,
     AnncMemRef2DF32* rhs,
     AnncMemRef1DF32* bias),
    {
        const auto status =
            ::annc::kernels::kdnn_adaptor::matmul_add_relu_kdnn_packed_impl(
                ::annc::threadpool::getCurrentThreadPool(),
                output,
                lhs,
                rhs,
                bias);
        return ::annc::kernels::statusCode(status);
    })
