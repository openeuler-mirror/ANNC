// FusedBatchedMatMulAdd (ONNX rank-3 MatMul + broadcast bias Add): the target
// models' attention QKV projections.  The weight's and bias's leading dims are
// broadcast (batch 1), so the fused kernel is a small GEMM per row plus a bias
// initializer, with the bias added after the full k sum (the source chain's
// order).
//
// Wrapper ABI (annc_execution_v2): (execution, data, weight, bias).  The output
// is allocated by the kernel through the execution context (slot 0, shape
// [d0, d1, N] read off the data descriptor) and the thread pool comes from that
// same context (resolveThreadPool: the pool the context carries, falling back
// to the pool injected into this library's TLS).
//
// Ranks 3/3/3/3; the leading dims are runtime values (the batch dim is
// dynamic), so sizes come from the memref descriptors.
//
// Only f32 is registered -- every site in the target models is float32.

ANNC_KERNEL(
    Name("BatchedMatMulAdd")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T"),
    (const AnncExecutionContext* execution, AnncMemRef3DF32* data,
     AnncMemRef3DF32* weight, AnncMemRef3DF32* bias),
    {
        const auto status = batchedMatMulAddImpl(
            ::annc::kernels::execution::resolveThreadPool(execution), execution,
            data, weight, bias);
        return ::annc::kernels::statusCode(status);
    })
