ANNC_KERNEL(
    Name("MatMulAddReluWithAddOutput")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T"),
    (const AnncExecutionContext* execution, AnncMemRef2DF32* lhs,
     AnncMemRef2DF32* rhs, AnncMemRef1DF32* bias),
    {
      const auto status = matmulAddReluWithAddOutputImpl(
          ::annc::threadpool::getCurrentThreadPool(), execution, lhs, rhs,
          bias);
      return ::annc::kernels::statusCode(status);
    })
