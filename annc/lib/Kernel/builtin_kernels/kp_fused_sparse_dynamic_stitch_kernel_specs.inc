// KPFusedSparseDynamicStitch specializations: N = number of partitions
// (2..16).  Multi-name scheme: each N specialization registers under its own
// op type (KPFusedSparseDynamicStitchN2..N16); the rewrite pattern builds the
// matching name from the table count. No registry attr channel needed.
// impl (template argument lists cannot appear in the ANNC_KERNEL macro
// body — the preprocessor splits them on the comma).

ANNC_KERNEL(
    Name("KPFusedSparseDynamicStitchN2").Backend("aarch64")
        .TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef1DI64* x,
     AnncMemRef2DF32* v0,
     AnncMemRef2DF32* v1),
    {
        const auto status = kp_fused_sparse_dynamic_stitch_n2(
            ::annc::threadpool::getCurrentThreadPool(), output, x,
            v0, v1);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseDynamicStitchN3").Backend("aarch64")
        .TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef1DI64* x,
     AnncMemRef2DF32* v0,
     AnncMemRef2DF32* v1,
     AnncMemRef2DF32* v2),
    {
        const auto status = kp_fused_sparse_dynamic_stitch_n3(
            ::annc::threadpool::getCurrentThreadPool(), output, x,
            v0, v1, v2);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseDynamicStitchN4").Backend("aarch64")
        .TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef1DI64* x,
     AnncMemRef2DF32* v0,
     AnncMemRef2DF32* v1,
     AnncMemRef2DF32* v2,
     AnncMemRef2DF32* v3),
    {
        const auto status = kp_fused_sparse_dynamic_stitch_n4(
            ::annc::threadpool::getCurrentThreadPool(), output, x,
            v0, v1, v2, v3);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseDynamicStitchN5").Backend("aarch64")
        .TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef1DI64* x,
     AnncMemRef2DF32* v0,
     AnncMemRef2DF32* v1,
     AnncMemRef2DF32* v2,
     AnncMemRef2DF32* v3,
     AnncMemRef2DF32* v4),
    {
        const auto status = kp_fused_sparse_dynamic_stitch_n5(
            ::annc::threadpool::getCurrentThreadPool(), output, x,
            v0, v1, v2, v3, v4);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseDynamicStitchN6").Backend("aarch64")
        .TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef1DI64* x,
     AnncMemRef2DF32* v0,
     AnncMemRef2DF32* v1,
     AnncMemRef2DF32* v2,
     AnncMemRef2DF32* v3,
     AnncMemRef2DF32* v4,
     AnncMemRef2DF32* v5),
    {
        const auto status = kp_fused_sparse_dynamic_stitch_n6(
            ::annc::threadpool::getCurrentThreadPool(), output, x,
            v0, v1, v2, v3, v4, v5);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseDynamicStitchN7").Backend("aarch64")
        .TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef1DI64* x,
     AnncMemRef2DF32* v0,
     AnncMemRef2DF32* v1,
     AnncMemRef2DF32* v2,
     AnncMemRef2DF32* v3,
     AnncMemRef2DF32* v4,
     AnncMemRef2DF32* v5,
     AnncMemRef2DF32* v6),
    {
        const auto status = kp_fused_sparse_dynamic_stitch_n7(
            ::annc::threadpool::getCurrentThreadPool(), output, x,
            v0, v1, v2, v3, v4, v5, v6);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseDynamicStitchN8").Backend("aarch64")
        .TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef1DI64* x,
     AnncMemRef2DF32* v0,
     AnncMemRef2DF32* v1,
     AnncMemRef2DF32* v2,
     AnncMemRef2DF32* v3,
     AnncMemRef2DF32* v4,
     AnncMemRef2DF32* v5,
     AnncMemRef2DF32* v6,
     AnncMemRef2DF32* v7),
    {
        const auto status = kp_fused_sparse_dynamic_stitch_n8(
            ::annc::threadpool::getCurrentThreadPool(), output, x,
            v0, v1, v2, v3, v4, v5, v6, v7);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseDynamicStitchN9").Backend("aarch64")
        .TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef1DI64* x,
     AnncMemRef2DF32* v0,
     AnncMemRef2DF32* v1,
     AnncMemRef2DF32* v2,
     AnncMemRef2DF32* v3,
     AnncMemRef2DF32* v4,
     AnncMemRef2DF32* v5,
     AnncMemRef2DF32* v6,
     AnncMemRef2DF32* v7,
     AnncMemRef2DF32* v8),
    {
        const auto status = kp_fused_sparse_dynamic_stitch_n9(
            ::annc::threadpool::getCurrentThreadPool(), output, x,
            v0, v1, v2, v3, v4, v5, v6, v7, v8);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseDynamicStitchN10").Backend("aarch64")
        .TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef1DI64* x,
     AnncMemRef2DF32* v0,
     AnncMemRef2DF32* v1,
     AnncMemRef2DF32* v2,
     AnncMemRef2DF32* v3,
     AnncMemRef2DF32* v4,
     AnncMemRef2DF32* v5,
     AnncMemRef2DF32* v6,
     AnncMemRef2DF32* v7,
     AnncMemRef2DF32* v8,
     AnncMemRef2DF32* v9),
    {
        const auto status = kp_fused_sparse_dynamic_stitch_n10(
            ::annc::threadpool::getCurrentThreadPool(), output, x,
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseDynamicStitchN11").Backend("aarch64")
        .TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef1DI64* x,
     AnncMemRef2DF32* v0,
     AnncMemRef2DF32* v1,
     AnncMemRef2DF32* v2,
     AnncMemRef2DF32* v3,
     AnncMemRef2DF32* v4,
     AnncMemRef2DF32* v5,
     AnncMemRef2DF32* v6,
     AnncMemRef2DF32* v7,
     AnncMemRef2DF32* v8,
     AnncMemRef2DF32* v9,
     AnncMemRef2DF32* v10),
    {
        const auto status = kp_fused_sparse_dynamic_stitch_n11(
            ::annc::threadpool::getCurrentThreadPool(), output, x,
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseDynamicStitchN12").Backend("aarch64")
        .TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef1DI64* x,
     AnncMemRef2DF32* v0,
     AnncMemRef2DF32* v1,
     AnncMemRef2DF32* v2,
     AnncMemRef2DF32* v3,
     AnncMemRef2DF32* v4,
     AnncMemRef2DF32* v5,
     AnncMemRef2DF32* v6,
     AnncMemRef2DF32* v7,
     AnncMemRef2DF32* v8,
     AnncMemRef2DF32* v9,
     AnncMemRef2DF32* v10,
     AnncMemRef2DF32* v11),
    {
        const auto status = kp_fused_sparse_dynamic_stitch_n12(
            ::annc::threadpool::getCurrentThreadPool(), output, x,
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseDynamicStitchN13").Backend("aarch64")
        .TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef1DI64* x,
     AnncMemRef2DF32* v0,
     AnncMemRef2DF32* v1,
     AnncMemRef2DF32* v2,
     AnncMemRef2DF32* v3,
     AnncMemRef2DF32* v4,
     AnncMemRef2DF32* v5,
     AnncMemRef2DF32* v6,
     AnncMemRef2DF32* v7,
     AnncMemRef2DF32* v8,
     AnncMemRef2DF32* v9,
     AnncMemRef2DF32* v10,
     AnncMemRef2DF32* v11,
     AnncMemRef2DF32* v12),
    {
        const auto status = kp_fused_sparse_dynamic_stitch_n13(
            ::annc::threadpool::getCurrentThreadPool(), output, x,
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseDynamicStitchN14").Backend("aarch64")
        .TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef1DI64* x,
     AnncMemRef2DF32* v0,
     AnncMemRef2DF32* v1,
     AnncMemRef2DF32* v2,
     AnncMemRef2DF32* v3,
     AnncMemRef2DF32* v4,
     AnncMemRef2DF32* v5,
     AnncMemRef2DF32* v6,
     AnncMemRef2DF32* v7,
     AnncMemRef2DF32* v8,
     AnncMemRef2DF32* v9,
     AnncMemRef2DF32* v10,
     AnncMemRef2DF32* v11,
     AnncMemRef2DF32* v12,
     AnncMemRef2DF32* v13),
    {
        const auto status = kp_fused_sparse_dynamic_stitch_n14(
            ::annc::threadpool::getCurrentThreadPool(), output, x,
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseDynamicStitchN15").Backend("aarch64")
        .TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef1DI64* x,
     AnncMemRef2DF32* v0,
     AnncMemRef2DF32* v1,
     AnncMemRef2DF32* v2,
     AnncMemRef2DF32* v3,
     AnncMemRef2DF32* v4,
     AnncMemRef2DF32* v5,
     AnncMemRef2DF32* v6,
     AnncMemRef2DF32* v7,
     AnncMemRef2DF32* v8,
     AnncMemRef2DF32* v9,
     AnncMemRef2DF32* v10,
     AnncMemRef2DF32* v11,
     AnncMemRef2DF32* v12,
     AnncMemRef2DF32* v13,
     AnncMemRef2DF32* v14),
    {
        const auto status = kp_fused_sparse_dynamic_stitch_n15(
            ::annc::threadpool::getCurrentThreadPool(), output, x,
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13,
            v14);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseDynamicStitchN16").Backend("aarch64")
        .TypeConstraint<float>("T"),
    (AnncMemRef2DF32* output,
     AnncMemRef1DI64* x,
     AnncMemRef2DF32* v0,
     AnncMemRef2DF32* v1,
     AnncMemRef2DF32* v2,
     AnncMemRef2DF32* v3,
     AnncMemRef2DF32* v4,
     AnncMemRef2DF32* v5,
     AnncMemRef2DF32* v6,
     AnncMemRef2DF32* v7,
     AnncMemRef2DF32* v8,
     AnncMemRef2DF32* v9,
     AnncMemRef2DF32* v10,
     AnncMemRef2DF32* v11,
     AnncMemRef2DF32* v12,
     AnncMemRef2DF32* v13,
     AnncMemRef2DF32* v14,
     AnncMemRef2DF32* v15),
    {
        const auto status = kp_fused_sparse_dynamic_stitch_n16(
            ::annc::threadpool::getCurrentThreadPool(), output, x,
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13,
            v14, v15);
        return ::annc::kernels::statusCode(status);
    })
