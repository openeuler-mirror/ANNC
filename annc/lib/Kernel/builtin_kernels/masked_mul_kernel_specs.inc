// FusedMaskedMul (ONNX mask chain): Equal(ids, pad) -> Not -> Cast(f32) ->
// Unsqueeze -> [Tile] -> Mul(data_i, mask) -> y_i, collapsed to one kernel per
// *site* -- and a site is the whole subgraph: the chain, every Mul it feeds,
// and the element-wise path between those Muls (model1's Sum -> Div(by a
// constant)) -- see masked_mul_aarch64.cpp for the full contract.
//
// Wrapper ABI (annc_execution_v2): the thread pool comes from the execution
// context (resolveThreadPool) and every output is allocated by the kernel
// through that context, in the slot order the pattern declares: the raw Mul
// results, then the Muls fed by the internal d, then the optional flat mask.
//
//   (execution, arg..., ids,
//    dataArgs, rawMuls, fedMuls, argMap, divisorBits, padding)
//
// `dataArgs` counts the leading arguments that are data (the rest are the Sum's
// other inputs); `argMap` holds one 4-bit data-argument index per raw Mul;
// `divisorBits` is the f32 bit pattern of the Div's constant (the ABI carries
// i64 attributes only).  One symbol per argument count (the execution-v2 C
// signature is fixed per symbol), plus a Flat variant that also writes the
// pre-unsqueeze mask when the graph still reads it.
//
// Only f32 data / i64 ids are registered -- every site in the target models
// has that pair (task 20260924-016).

ANNC_KERNEL(
    Name("MaskedMul1")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("TMask"),
    (const AnncExecutionContext* execution, AnncMemRef3DF32* a0,
     AnncMemRef2DI64* ids, int64_t dataArgs, int64_t rawMuls, int64_t fedMuls,
     int64_t argMap, int64_t sumMask, int64_t interiorKind, int64_t divisorBits,
     int64_t padding),
    {
        const auto status = maskedMul1Impl(
            ::annc::kernels::execution::resolveThreadPool(execution), execution,
            a0, ids, dataArgs, rawMuls, fedMuls, argMap, sumMask, interiorKind, divisorBits,
            padding);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("MaskedMul2")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("TMask"),
    (const AnncExecutionContext* execution, AnncMemRef3DF32* a0,
     AnncMemRef3DF32* a1, AnncMemRef2DI64* ids, int64_t dataArgs,
     int64_t rawMuls, int64_t fedMuls, int64_t argMap, int64_t sumMask,
     int64_t interiorKind, int64_t divisorBits, int64_t padding),
    {
        const auto status = maskedMul2Impl(
            ::annc::kernels::execution::resolveThreadPool(execution), execution,
            a0, a1, ids, dataArgs, rawMuls, fedMuls, argMap, sumMask, interiorKind,
            divisorBits, padding);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("MaskedMul3")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("TMask"),
    (const AnncExecutionContext* execution, AnncMemRef3DF32* a0,
     AnncMemRef3DF32* a1, AnncMemRef3DF32* a2, AnncMemRef2DI64* ids,
     int64_t dataArgs, int64_t rawMuls, int64_t fedMuls, int64_t argMap,
     int64_t sumMask, int64_t interiorKind, int64_t divisorBits,
     int64_t padding),
    {
        const auto status = maskedMul3Impl(
            ::annc::kernels::execution::resolveThreadPool(execution), execution,
            a0, a1, a2, ids, dataArgs, rawMuls, fedMuls, argMap, sumMask, interiorKind,
            divisorBits, padding);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("MaskedMul4")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("TMask"),
    (const AnncExecutionContext* execution, AnncMemRef3DF32* a0,
     AnncMemRef3DF32* a1, AnncMemRef3DF32* a2, AnncMemRef3DF32* a3,
     AnncMemRef2DI64* ids, int64_t dataArgs, int64_t rawMuls, int64_t fedMuls,
     int64_t argMap, int64_t sumMask, int64_t interiorKind, int64_t divisorBits,
     int64_t padding),
    {
        const auto status = maskedMul4Impl(
            ::annc::kernels::execution::resolveThreadPool(execution), execution,
            a0, a1, a2, a3, ids, dataArgs, rawMuls, fedMuls, argMap, sumMask, interiorKind,
            divisorBits, padding);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("MaskedMul6")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("TMask"),
    (const AnncExecutionContext* execution, AnncMemRef3DF32* a0,
     AnncMemRef3DF32* a1, AnncMemRef3DF32* a2, AnncMemRef3DF32* a3,
     AnncMemRef3DF32* a4, AnncMemRef3DF32* a5, AnncMemRef2DI64* ids,
     int64_t dataArgs, int64_t rawMuls, int64_t fedMuls, int64_t argMap,
     int64_t sumMask, int64_t interiorKind, int64_t divisorBits,
     int64_t padding),
    {
        const auto status = maskedMul6Impl(
            ::annc::kernels::execution::resolveThreadPool(execution), execution,
            a0, a1, a2, a3, a4, a5, ids, dataArgs, rawMuls, fedMuls, argMap,
            sumMask, interiorKind, divisorBits, padding);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("MaskedMul7")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("TMask"),
    (const AnncExecutionContext* execution, AnncMemRef3DF32* a0,
     AnncMemRef3DF32* a1, AnncMemRef3DF32* a2, AnncMemRef3DF32* a3,
     AnncMemRef3DF32* a4, AnncMemRef3DF32* a5, AnncMemRef3DF32* a6,
     AnncMemRef2DI64* ids, int64_t dataArgs, int64_t rawMuls, int64_t fedMuls,
     int64_t argMap, int64_t sumMask, int64_t interiorKind, int64_t divisorBits,
     int64_t padding),
    {
        const auto status = maskedMul7Impl(
            ::annc::kernels::execution::resolveThreadPool(execution), execution,
            a0, a1, a2, a3, a4, a5, a6, ids, dataArgs, rawMuls, fedMuls,
            argMap, sumMask, interiorKind, divisorBits, padding);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("MaskedMul11")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("TMask"),
    (const AnncExecutionContext* execution, AnncMemRef3DF32* a0,
     AnncMemRef3DF32* a1, AnncMemRef3DF32* a2, AnncMemRef3DF32* a3,
     AnncMemRef3DF32* a4, AnncMemRef3DF32* a5, AnncMemRef3DF32* a6,
     AnncMemRef3DF32* a7, AnncMemRef3DF32* a8, AnncMemRef3DF32* a9,
     AnncMemRef3DF32* a10, AnncMemRef2DI64* ids, int64_t dataArgs,
     int64_t rawMuls, int64_t fedMuls, int64_t argMap, int64_t sumMask,
     int64_t interiorKind, int64_t divisorBits, int64_t padding),
    {
        const auto status = maskedMul11Impl(
            ::annc::kernels::execution::resolveThreadPool(execution), execution,
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, ids, dataArgs,
            rawMuls, fedMuls, argMap, sumMask, interiorKind, divisorBits, padding);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("MaskedMul1Flat")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("TMask"),
    (const AnncExecutionContext* execution, AnncMemRef3DF32* a0,
     AnncMemRef2DI64* ids, int64_t dataArgs, int64_t rawMuls, int64_t fedMuls,
     int64_t argMap, int64_t sumMask, int64_t interiorKind, int64_t divisorBits,
     int64_t padding),
    {
        const auto status = maskedMul1FlatImpl(
            ::annc::kernels::execution::resolveThreadPool(execution), execution,
            a0, ids, dataArgs, rawMuls, fedMuls, argMap, sumMask, interiorKind, divisorBits,
            padding);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("MaskedMul3Flat")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("TMask"),
    (const AnncExecutionContext* execution, AnncMemRef3DF32* a0,
     AnncMemRef3DF32* a1, AnncMemRef3DF32* a2, AnncMemRef2DI64* ids,
     int64_t dataArgs, int64_t rawMuls, int64_t fedMuls, int64_t argMap,
     int64_t sumMask, int64_t interiorKind, int64_t divisorBits,
     int64_t padding),
    {
        const auto status = maskedMul3FlatImpl(
            ::annc::kernels::execution::resolveThreadPool(execution), execution,
            a0, a1, a2, ids, dataArgs, rawMuls, fedMuls, argMap, sumMask, interiorKind,
            divisorBits, padding);
        return ::annc::kernels::statusCode(status);
    })

ANNC_KERNEL(
    Name("MaskedMul4Flat")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T")
        .TypeConstraint<int64_t>("TMask"),
    (const AnncExecutionContext* execution, AnncMemRef3DF32* a0,
     AnncMemRef3DF32* a1, AnncMemRef3DF32* a2, AnncMemRef3DF32* a3,
     AnncMemRef2DI64* ids, int64_t dataArgs, int64_t rawMuls, int64_t fedMuls,
     int64_t argMap, int64_t sumMask, int64_t interiorKind, int64_t divisorBits,
     int64_t padding),
    {
        const auto status = maskedMul4FlatImpl(
            ::annc::kernels::execution::resolveThreadPool(execution), execution,
            a0, a1, a2, a3, ids, dataArgs, rawMuls, fedMuls, argMap, sumMask, interiorKind,
            divisorBits, padding);
        return ::annc::kernels::statusCode(status);
    })
