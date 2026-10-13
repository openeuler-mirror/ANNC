// FusedMaskedMul (ONNX mask chain): Equal(ids, pad) -> Not -> Cast(f32) ->
// Unsqueeze -> [Tile] -> Mul(data, mask), collapsed to one kernel that reads
// ids row by row (the Tile is pure broadcast and moves no information).
//
// Wrapper ABI (annc_execution_v2): (execution, data..., ids, padding).  The
// outputs are allocated by the kernel through the execution context -- one
// slot per site, in the order the pattern lists them -- and the thread pool
// comes from that same context (resolveThreadPool: the pool the context
// carries, falling back to the pool injected into this library's TLS).
//
// One chain feeds 1..4 Muls and all of them die with the chain, so a site's
// arity is the chain's Mul count; the execution-v2 C signature is fixed per
// symbol, hence one symbol per arity (MaskedMul1..MaskedMul4), sharing one
// implementation.  Ranks 3 (data) / 2 (ids); the leading dims are runtime
// values (the batch dim is dynamic), so sizes come from the memref
// descriptors.
//
// Only f32 data / i64 ids are registered -- every site in the target models
// has that pair (task 20260924-016).

ANNC_KERNEL(
    Name("MaskedMul1")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("TMask"),
    (const AnncExecutionContext* execution, AnncMemRef3DF32* data0,
     AnncMemRef2DI64* ids, int64_t padding),
    {
        const auto status = maskedMul1Impl(
            ::annc::kernels::execution::resolveThreadPool(execution), execution,
            data0, ids, padding);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("MaskedMul2")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("TMask"),
    (const AnncExecutionContext* execution, AnncMemRef3DF32* data0,
     AnncMemRef3DF32* data1, AnncMemRef2DI64* ids, int64_t padding),
    {
        const auto status = maskedMul2Impl(
            ::annc::kernels::execution::resolveThreadPool(execution), execution,
            data0, data1, ids, padding);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("MaskedMul3")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("TMask"),
    (const AnncExecutionContext* execution, AnncMemRef3DF32* data0,
     AnncMemRef3DF32* data1, AnncMemRef3DF32* data2, AnncMemRef2DI64* ids,
     int64_t padding),
    {
        const auto status = maskedMul3Impl(
            ::annc::kernels::execution::resolveThreadPool(execution), execution,
            data0, data1, data2, ids, padding);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("MaskedMul4")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("TMask"),
    (const AnncExecutionContext* execution, AnncMemRef3DF32* data0,
     AnncMemRef3DF32* data1, AnncMemRef3DF32* data2, AnncMemRef3DF32* data3,
     AnncMemRef2DI64* ids, int64_t padding),
    {
        const auto status = maskedMul4Impl(
            ::annc::kernels::execution::resolveThreadPool(execution), execution,
            data0, data1, data2, data3, ids, padding);
        return ::annc::kernels::statusCode(status);
    })
