ANNC_KERNEL(
    Name("AutoDefaultOp").Backend("aarch64"),
    (AnncMemRef1DF32* output, AnncMemRef1DF32* input),
    { auto_default_impl(output, input); })

ANNC_KERNEL(
    Name("AutoSpecOp")
        .Backend("aarch64")
        .TypeConstraint<float>("Tdata"),
    (),
    { auto_spec_f32_impl(); })

ANNC_KERNEL(
    Name("AutoSpecOp")
        .Backend("aarch64")
        .TypeConstraint<int32_t>("Tdata"),
    (),
    { auto_spec_i32_impl(); })
