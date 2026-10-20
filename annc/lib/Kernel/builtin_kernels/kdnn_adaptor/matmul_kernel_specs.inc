ANNC_KERNEL(
    Name("MatMul").Backend("kdnn").TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* lhs,
     AnncMemRef2DF32* rhs),
    {
        const auto status = ::annc::kernels::kdnn_adaptor::matmul_kdnn_impl(
            ::annc::threadpool::getCurrentThreadPool(),
            output,
            lhs,
            rhs);
        return ::annc::kernels::statusCode(status);
    })

// Execution-v2 variant: output allocated via the execution context, thread
// pool resolved from the context (ONNX) or TLS (TF).
ANNC_KERNEL(
    Name("MatMul")
        .Backend("kdnn")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T"),
    (const AnncExecutionContext* execution,
     AnncMemRef2DF32* lhs,
     AnncMemRef2DF32* rhs),
    {
        const auto status =
            ::annc::kernels::kdnn_adaptor::matmul_kdnn_exec_v2_impl(
                execution, lhs, rhs);
        return ::annc::kernels::statusCode(status);
    })
