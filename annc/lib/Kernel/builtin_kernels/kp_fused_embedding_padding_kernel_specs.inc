// KPFusedEmbeddingPaddingFast: two scalar outputs (padding_rows,
// reshape_rows).  KPFusedEmbeddingPadding: scalar padding_rows + padded
// [reshape_rows, reshape_cols] data output.  Execution V2: outputs are
// allocated through the execution API (slot 0 = padding_rows, slot 1 =
// reshape_rows / out_data).

ANNC_KERNEL(
    Name("KPFusedEmbeddingPaddingFast")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T"),
    (const AnncExecutionContext* execution, AnncMemRef1DI64* origin_shape,
     AnncMemRef0DI32* input_rows, AnncMemRef0DI32* pack,
     AnncMemRef2DF32* data, AnncMemRef1DI32* reshape_sizes),
    {
        const auto status = kpFusedEmbeddingPaddingFast(
            ::annc::threadpool::getCurrentThreadPool(), execution,
            origin_shape, input_rows, pack, data, reshape_sizes);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedEmbeddingPadding")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T"),
    (const AnncExecutionContext* execution, AnncMemRef1DI64* origin_shape,
     AnncMemRef0DI32* input_rows, AnncMemRef0DI32* pack,
     AnncMemRef2DF32* data, AnncMemRef1DI32* reshape_sizes),
    {
        const auto status = kpFusedEmbeddingPaddingFull(
            ::annc::threadpool::getCurrentThreadPool(), execution,
            origin_shape, input_rows, pack, data, reshape_sizes);
        return ::annc::kernels::statusCode(status);
    })
