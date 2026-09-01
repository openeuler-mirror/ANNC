// KPFusedSparseSelect specializations: T (int32) — the element type of
// a/b/c and the 4 comparison scalars (812 kernel hardcodes int32).  Outputs
// x is i32, y/w are f32.  Execution V2: outputs allocated through the
// execution API (slot 0 = out_x, slot 1 = out_y, slot 2 = out_w).
// Wrapper ABI: (execution, a, b, c, gt, eq1, eq2, eq3) — 7 inputs, exactly
// the V2 input limit.

ANNC_KERNEL(
    Name("KPFusedSparseSelect")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<int32_t>("T"),
    (const AnncExecutionContext* execution, AnncMemRef1DI32* a,
     AnncMemRef1DI32* b, AnncMemRef1DI32* c, AnncMemRef0DI32* gt,
     AnncMemRef0DI32* eq1, AnncMemRef0DI32* eq2, AnncMemRef0DI32* eq3),
    {
        const auto status = kpFusedSparseSelect(
            ::annc::threadpool::getCurrentThreadPool(), execution, a, b, c,
            gt, eq1, eq2, eq3);
        return ::annc::kernels::statusCode(status);
    })
