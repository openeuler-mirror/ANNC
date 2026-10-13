// FusedTensordotMatMul (task 20260924-004): the ONNX Tensordot block
// Reshape(MatMul(Reshape(x, [prod(free), K]), W), free ++ [N]) folds into this
// single custom op.  Only f32 is registered — every site in the four target
// models is float32, matching ORT's patch kernel registration.
//
// Wrapper ABI (annc_execution_v2): (execution, input, weight), ranks 3/2.  The
// kernel allocates its output (slot 0, shape [d0, d1, N]) through the
// execution context and takes the thread pool from that same context
// (resolveThreadPool: the pool the context carries, falling back to the pool
// injected into this library's TLS).

ANNC_KERNEL(
    Name("TensordotMatMul")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T"),
    (const AnncExecutionContext* execution, AnncMemRef3DF32* input,
     AnncMemRef2DF32* weight),
    {
        const auto status = tensordotMatMulImpl(
            ::annc::kernels::execution::resolveThreadPool(execution),
            execution, input, weight);
        return ::annc::kernels::statusCode(status);
    })
