// FusedBatchedMatMulAdd (ONNX MatMul + broadcast bias Add): the target models'
// attention QKV projections (rank-3 [B, K, N] weights) and their per-domain
// projections (rank-4 [W0, W1, K, N] weights). Data and weights may broadcast
// their batch dims; bias has one vector per output batch. Bias is added after
// the full k sum (the source chain's order).
//
// Wrapper ABI (annc_execution_v2): (execution, data, weight, bias).  The output
// is allocated by the kernel through the execution context (slot 0, shape
// [batch..., M, N] read off the data descriptor) and the thread pool comes from
// that same context (resolveThreadPool: the pool the context carries, falling
// back to the pool injected into this library's TLS).
//
// One symbol per rank (BatchedMatMulAdd for rank 3, BatchedMatMulAdd4 for rank
// 4): the execution-v2 C signature is fixed per symbol, so the memref arity
// cannot be a parameter -- the same scheme the MaskedMul1..4 arities use. The
// leading dims are runtime values (the batch dim is dynamic), so sizes come
// from the memref descriptors.
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

ANNC_KERNEL(
    Name("BatchedMatMulAdd4")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T"),
    (const AnncExecutionContext* execution, AnncMemRef4DF32* data,
     AnncMemRef4DF32* weight, AnncMemRef4DF32* bias),
    {
        const auto status = batchedMatMulAddRank4Impl(
            ::annc::kernels::execution::resolveThreadPool(execution), execution,
            data, weight, bias);
        return ::annc::kernels::statusCode(status);
    })
