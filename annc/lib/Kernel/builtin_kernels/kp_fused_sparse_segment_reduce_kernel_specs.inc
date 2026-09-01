// KPFusedSparseSegmentReduce Execution V2 specializations: Tidx
// (int64/int32) x combiner (Sum/Mean).  Multi-name scheme — one op name per
// specialization because KernelRegistry matching has no attrs channel.
// Wrapper ABI: (execution, keys, begin, data, indices, begin_1); outputs are
// allocated via the execution API (slot 0 = output, slot 1 = slice_output).

ANNC_KERNEL(
    Name("KPFusedSparseSegmentReduceI64Sum")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("Tidx"),
    (const AnncExecutionContext* execution, AnncMemRef2DI64* keys,
     AnncMemRef1DI32* begin, AnncMemRef2DF32* data,
     AnncMemRef1DI64* indices, AnncMemRef1DI32* begin_1),
    {
        const auto status = kpFusedSparseSegmentReduceI64Sum(
            ::annc::threadpool::getCurrentThreadPool(), execution, keys,
            begin, data, indices, begin_1);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseSegmentReduceI64Mean")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("Tidx"),
    (const AnncExecutionContext* execution, AnncMemRef2DI64* keys,
     AnncMemRef1DI32* begin, AnncMemRef2DF32* data,
     AnncMemRef1DI64* indices, AnncMemRef1DI32* begin_1),
    {
        const auto status = kpFusedSparseSegmentReduceI64Mean(
            ::annc::threadpool::getCurrentThreadPool(), execution, keys,
            begin, data, indices, begin_1);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseSegmentReduceI32Sum")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("Tidx"),
    (const AnncExecutionContext* execution, AnncMemRef2DI64* keys,
     AnncMemRef1DI32* begin, AnncMemRef2DF32* data,
     AnncMemRef1DI32* indices, AnncMemRef1DI32* begin_1),
    {
        const auto status = kpFusedSparseSegmentReduceI32Sum(
            ::annc::threadpool::getCurrentThreadPool(), execution, keys,
            begin, data, indices, begin_1);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseSegmentReduceI32Mean")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("Tidx"),
    (const AnncExecutionContext* execution, AnncMemRef2DI64* keys,
     AnncMemRef1DI32* begin, AnncMemRef2DF32* data,
     AnncMemRef1DI32* indices, AnncMemRef1DI32* begin_1),
    {
        const auto status = kpFusedSparseSegmentReduceI32Mean(
            ::annc::threadpool::getCurrentThreadPool(), execution, keys,
            begin, data, indices, begin_1);
        return ::annc::kernels::statusCode(status);
    })
