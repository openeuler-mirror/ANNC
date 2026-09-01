// KPFusedEmbeddingActionIdGather specializations: indices1/indices2 are 1-D
// or 2-D; dtype combos mirror the JD registration — (i64,i32) [v1 default],
// (i32,i32), (i64,i64), (i32,i64).  Each (R1,R2) rank combo registers under
// its own op type (KPFusedEmbeddingActionIdGatherR1x1 .. R2x2); the registry
// selects among same-name entries by the T1/T2 type constraints, no registry
// attr channel needed.

// R1=1, R2=1
ANNC_KERNEL(
    Name("KPFusedEmbeddingActionIdGatherR1x1").Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("T1")
        .TypeConstraint<int32_t>("T2"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* params,
     AnncMemRef1DI64* indices1,
     AnncMemRef1DI32* indices2,
     AnncMemRef0DI32* pack_dim,
     AnncMemRef0DI32* pack),
    {
        const auto status = kActionIdGather11(
            ::annc::threadpool::getCurrentThreadPool(), output,
            ANNC_MEMREF_DATA(*indices1), indices1->sizes[0], 1, params,
            ANNC_MEMREF_DATA(*indices2), 1, indices2->sizes[0],
            *ANNC_MEMREF_DATA(*pack_dim), *ANNC_MEMREF_DATA(*pack));
        return ::annc::kernels::statusCode(status);
    })

// R1=2, R2=1
ANNC_KERNEL(
    Name("KPFusedEmbeddingActionIdGatherR2x1").Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("T1")
        .TypeConstraint<int32_t>("T2"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* params,
     AnncMemRef2DI64* indices1,
     AnncMemRef1DI32* indices2,
     AnncMemRef0DI32* pack_dim,
     AnncMemRef0DI32* pack),
    {
        const auto status = kActionIdGather21(
            ::annc::threadpool::getCurrentThreadPool(), output,
            ANNC_MEMREF_DATA(*indices1), indices1->sizes[0],
            indices1->sizes[1], params,
            ANNC_MEMREF_DATA(*indices2), 1, indices2->sizes[0],
            *ANNC_MEMREF_DATA(*pack_dim), *ANNC_MEMREF_DATA(*pack));
        return ::annc::kernels::statusCode(status);
    })

// R1=1, R2=2
ANNC_KERNEL(
    Name("KPFusedEmbeddingActionIdGatherR1x2").Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("T1")
        .TypeConstraint<int32_t>("T2"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* params,
     AnncMemRef1DI64* indices1,
     AnncMemRef2DI32* indices2,
     AnncMemRef0DI32* pack_dim,
     AnncMemRef0DI32* pack),
    {
        const auto status = kActionIdGather12(
            ::annc::threadpool::getCurrentThreadPool(), output,
            ANNC_MEMREF_DATA(*indices1), indices1->sizes[0], 1, params,
            ANNC_MEMREF_DATA(*indices2), indices2->sizes[0],
            indices2->sizes[1],
            *ANNC_MEMREF_DATA(*pack_dim), *ANNC_MEMREF_DATA(*pack));
        return ::annc::kernels::statusCode(status);
    })

// R1=2, R2=2
ANNC_KERNEL(
    Name("KPFusedEmbeddingActionIdGatherR2x2").Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("T1")
        .TypeConstraint<int32_t>("T2"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* params,
     AnncMemRef2DI64* indices1,
     AnncMemRef2DI32* indices2,
     AnncMemRef0DI32* pack_dim,
     AnncMemRef0DI32* pack),
    {
        const auto status = kActionIdGather22(
            ::annc::threadpool::getCurrentThreadPool(), output,
            ANNC_MEMREF_DATA(*indices1), indices1->sizes[0],
            indices1->sizes[1], params,
            ANNC_MEMREF_DATA(*indices2), indices2->sizes[0],
            indices2->sizes[1],
            *ANNC_MEMREF_DATA(*pack_dim), *ANNC_MEMREF_DATA(*pack));
        return ::annc::kernels::statusCode(status);
    })

// ── (i32,i32) ──

ANNC_KERNEL(
    Name("KPFusedEmbeddingActionIdGatherR1x1").Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("T1")
        .TypeConstraint<int32_t>("T2"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* params,
     AnncMemRef1DI32* indices1,
     AnncMemRef1DI32* indices2,
     AnncMemRef0DI32* pack_dim,
     AnncMemRef0DI32* pack),
    {
        const auto status = kActionIdGather11_I32I32(
            ::annc::threadpool::getCurrentThreadPool(), output,
            ANNC_MEMREF_DATA(*indices1), indices1->sizes[0], 1, params,
            ANNC_MEMREF_DATA(*indices2), 1, indices2->sizes[0],
            *ANNC_MEMREF_DATA(*pack_dim), *ANNC_MEMREF_DATA(*pack));
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedEmbeddingActionIdGatherR2x1").Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("T1")
        .TypeConstraint<int32_t>("T2"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* params,
     AnncMemRef2DI32* indices1,
     AnncMemRef1DI32* indices2,
     AnncMemRef0DI32* pack_dim,
     AnncMemRef0DI32* pack),
    {
        const auto status = kActionIdGather21_I32I32(
            ::annc::threadpool::getCurrentThreadPool(), output,
            ANNC_MEMREF_DATA(*indices1), indices1->sizes[0],
            indices1->sizes[1], params,
            ANNC_MEMREF_DATA(*indices2), 1, indices2->sizes[0],
            *ANNC_MEMREF_DATA(*pack_dim), *ANNC_MEMREF_DATA(*pack));
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedEmbeddingActionIdGatherR1x2").Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("T1")
        .TypeConstraint<int32_t>("T2"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* params,
     AnncMemRef1DI32* indices1,
     AnncMemRef2DI32* indices2,
     AnncMemRef0DI32* pack_dim,
     AnncMemRef0DI32* pack),
    {
        const auto status = kActionIdGather12_I32I32(
            ::annc::threadpool::getCurrentThreadPool(), output,
            ANNC_MEMREF_DATA(*indices1), indices1->sizes[0], 1, params,
            ANNC_MEMREF_DATA(*indices2), indices2->sizes[0],
            indices2->sizes[1],
            *ANNC_MEMREF_DATA(*pack_dim), *ANNC_MEMREF_DATA(*pack));
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedEmbeddingActionIdGatherR2x2").Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("T1")
        .TypeConstraint<int32_t>("T2"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* params,
     AnncMemRef2DI32* indices1,
     AnncMemRef2DI32* indices2,
     AnncMemRef0DI32* pack_dim,
     AnncMemRef0DI32* pack),
    {
        const auto status = kActionIdGather22_I32I32(
            ::annc::threadpool::getCurrentThreadPool(), output,
            ANNC_MEMREF_DATA(*indices1), indices1->sizes[0],
            indices1->sizes[1], params,
            ANNC_MEMREF_DATA(*indices2), indices2->sizes[0],
            indices2->sizes[1],
            *ANNC_MEMREF_DATA(*pack_dim), *ANNC_MEMREF_DATA(*pack));
        return ::annc::kernels::statusCode(status);
    })

// ── (i64,i64) ──

ANNC_KERNEL(
    Name("KPFusedEmbeddingActionIdGatherR1x1").Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("T1")
        .TypeConstraint<int64_t>("T2"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* params,
     AnncMemRef1DI64* indices1,
     AnncMemRef1DI64* indices2,
     AnncMemRef0DI32* pack_dim,
     AnncMemRef0DI32* pack),
    {
        const auto status = kActionIdGather11_I64I64(
            ::annc::threadpool::getCurrentThreadPool(), output,
            ANNC_MEMREF_DATA(*indices1), indices1->sizes[0], 1, params,
            ANNC_MEMREF_DATA(*indices2), 1, indices2->sizes[0],
            *ANNC_MEMREF_DATA(*pack_dim), *ANNC_MEMREF_DATA(*pack));
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedEmbeddingActionIdGatherR2x1").Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("T1")
        .TypeConstraint<int64_t>("T2"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* params,
     AnncMemRef2DI64* indices1,
     AnncMemRef1DI64* indices2,
     AnncMemRef0DI32* pack_dim,
     AnncMemRef0DI32* pack),
    {
        const auto status = kActionIdGather21_I64I64(
            ::annc::threadpool::getCurrentThreadPool(), output,
            ANNC_MEMREF_DATA(*indices1), indices1->sizes[0],
            indices1->sizes[1], params,
            ANNC_MEMREF_DATA(*indices2), 1, indices2->sizes[0],
            *ANNC_MEMREF_DATA(*pack_dim), *ANNC_MEMREF_DATA(*pack));
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedEmbeddingActionIdGatherR1x2").Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("T1")
        .TypeConstraint<int64_t>("T2"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* params,
     AnncMemRef1DI64* indices1,
     AnncMemRef2DI64* indices2,
     AnncMemRef0DI32* pack_dim,
     AnncMemRef0DI32* pack),
    {
        const auto status = kActionIdGather12_I64I64(
            ::annc::threadpool::getCurrentThreadPool(), output,
            ANNC_MEMREF_DATA(*indices1), indices1->sizes[0], 1, params,
            ANNC_MEMREF_DATA(*indices2), indices2->sizes[0],
            indices2->sizes[1],
            *ANNC_MEMREF_DATA(*pack_dim), *ANNC_MEMREF_DATA(*pack));
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedEmbeddingActionIdGatherR2x2").Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("T1")
        .TypeConstraint<int64_t>("T2"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* params,
     AnncMemRef2DI64* indices1,
     AnncMemRef2DI64* indices2,
     AnncMemRef0DI32* pack_dim,
     AnncMemRef0DI32* pack),
    {
        const auto status = kActionIdGather22_I64I64(
            ::annc::threadpool::getCurrentThreadPool(), output,
            ANNC_MEMREF_DATA(*indices1), indices1->sizes[0],
            indices1->sizes[1], params,
            ANNC_MEMREF_DATA(*indices2), indices2->sizes[0],
            indices2->sizes[1],
            *ANNC_MEMREF_DATA(*pack_dim), *ANNC_MEMREF_DATA(*pack));
        return ::annc::kernels::statusCode(status);
    })

// ── (i32,i64) ──

ANNC_KERNEL(
    Name("KPFusedEmbeddingActionIdGatherR1x1").Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("T1")
        .TypeConstraint<int64_t>("T2"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* params,
     AnncMemRef1DI32* indices1,
     AnncMemRef1DI64* indices2,
     AnncMemRef0DI32* pack_dim,
     AnncMemRef0DI32* pack),
    {
        const auto status = kActionIdGather11_I32I64(
            ::annc::threadpool::getCurrentThreadPool(), output,
            ANNC_MEMREF_DATA(*indices1), indices1->sizes[0], 1, params,
            ANNC_MEMREF_DATA(*indices2), 1, indices2->sizes[0],
            *ANNC_MEMREF_DATA(*pack_dim), *ANNC_MEMREF_DATA(*pack));
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedEmbeddingActionIdGatherR2x1").Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("T1")
        .TypeConstraint<int64_t>("T2"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* params,
     AnncMemRef2DI32* indices1,
     AnncMemRef1DI64* indices2,
     AnncMemRef0DI32* pack_dim,
     AnncMemRef0DI32* pack),
    {
        const auto status = kActionIdGather21_I32I64(
            ::annc::threadpool::getCurrentThreadPool(), output,
            ANNC_MEMREF_DATA(*indices1), indices1->sizes[0],
            indices1->sizes[1], params,
            ANNC_MEMREF_DATA(*indices2), 1, indices2->sizes[0],
            *ANNC_MEMREF_DATA(*pack_dim), *ANNC_MEMREF_DATA(*pack));
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedEmbeddingActionIdGatherR1x2").Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("T1")
        .TypeConstraint<int64_t>("T2"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* params,
     AnncMemRef1DI32* indices1,
     AnncMemRef2DI64* indices2,
     AnncMemRef0DI32* pack_dim,
     AnncMemRef0DI32* pack),
    {
        const auto status = kActionIdGather12_I32I64(
            ::annc::threadpool::getCurrentThreadPool(), output,
            ANNC_MEMREF_DATA(*indices1), indices1->sizes[0], 1, params,
            ANNC_MEMREF_DATA(*indices2), indices2->sizes[0],
            indices2->sizes[1],
            *ANNC_MEMREF_DATA(*pack_dim), *ANNC_MEMREF_DATA(*pack));
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedEmbeddingActionIdGatherR2x2").Backend("aarch64")
        .TypeConstraint<float>("T")
        .TypeConstraint<int32_t>("T1")
        .TypeConstraint<int64_t>("T2"),
    (AnncMemRef2DF32* output,
     AnncMemRef2DF32* params,
     AnncMemRef2DI32* indices1,
     AnncMemRef2DI64* indices2,
     AnncMemRef0DI32* pack_dim,
     AnncMemRef0DI32* pack),
    {
        const auto status = kActionIdGather22_I32I64(
            ::annc::threadpool::getCurrentThreadPool(), output,
            ANNC_MEMREF_DATA(*indices1), indices1->sizes[0],
            indices1->sizes[1], params,
            ANNC_MEMREF_DATA(*indices2), indices2->sizes[0],
            indices2->sizes[1],
            *ANNC_MEMREF_DATA(*pack_dim), *ANNC_MEMREF_DATA(*pack));
        return ::annc::kernels::statusCode(status);
    })
