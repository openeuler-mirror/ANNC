// KPFusedSparseReshape specializations: T (int64/int32) — the element type
// of pack_const.  new_shape is always i64 (812 SparseReshape semantics).
// Multi-name scheme (no attrs channel in the B KernelRegistry).
// Wrapper ABI (Execution V2): (execution, slice_input, begin, pack_const,
// new_shape); outputs allocated via execution API (slot 0 = out_indices,
// slot 1 = out_shape).

ANNC_KERNEL(
    Name("KPFusedSparseReshapeI64")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<int64_t>("T"),
    (const AnncExecutionContext* execution, AnncMemRef2DI64* slice_input,
     AnncMemRef1DI32* begin, AnncMemRef0DI64* pack_const,
     AnncMemRef1DI64* new_shape),
    {
        const auto status = kpFusedSparseReshapeI64(
            ::annc::threadpool::getCurrentThreadPool(), execution,
            slice_input, begin, pack_const, new_shape);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseReshapeI32")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<int32_t>("T"),
    (const AnncExecutionContext* execution, AnncMemRef2DI64* slice_input,
     AnncMemRef1DI32* begin, AnncMemRef0DI32* pack_const,
     AnncMemRef1DI64* new_shape),
    {
        const auto status = kpFusedSparseReshapeI32(
            ::annc::threadpool::getCurrentThreadPool(), execution,
            slice_input, begin, pack_const, new_shape);
        return ::annc::kernels::statusCode(status);
    })

// DynLimit variants: trailing 0-D i32 range_limit (dynamic Range.limit from
// varlen-batch graphs) — unused by the kernel, present for the ABI wrapper.
ANNC_KERNEL(
    Name("KPFusedSparseReshapeI64DynLimit")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<int64_t>("T"),
    (const AnncExecutionContext* execution, AnncMemRef2DI64* slice_input,
     AnncMemRef1DI32* begin, AnncMemRef0DI64* pack_const,
     AnncMemRef1DI64* new_shape, AnncMemRef0DI32* range_limit),
    {
        const auto status = kpFusedSparseReshapeI64DynLimit(
            ::annc::threadpool::getCurrentThreadPool(), execution,
            slice_input, begin, pack_const, new_shape, range_limit);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("KPFusedSparseReshapeI32DynLimit")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<int32_t>("T"),
    (const AnncExecutionContext* execution, AnncMemRef2DI64* slice_input,
     AnncMemRef1DI32* begin, AnncMemRef0DI32* pack_const,
     AnncMemRef1DI64* new_shape, AnncMemRef0DI32* range_limit),
    {
        const auto status = kpFusedSparseReshapeI32DynLimit(
            ::annc::threadpool::getCurrentThreadPool(), execution,
            slice_input, begin, pack_const, new_shape, range_limit);
        return ::annc::kernels::statusCode(status);
    })
