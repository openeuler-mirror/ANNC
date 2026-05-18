ANNC_KERNEL(
    Name("AutoDefaultOp").Backend("aarch64"),
    ANNC_FORWARD(
        auto_default_impl,
        (AnncMemRef1DF32* output, AnncMemRef1DF32* input),
        (output, input)))

ANNC_KERNEL(
    Name("AutoSpecOp")
        .Backend("aarch64")
        .TypeConstraint<float>("Tdata"),
    ANNC_FORWARD(
        auto_spec_f32_impl,
        (),
        ()))

ANNC_KERNEL(
    Name("AutoSpecOp")
        .Backend("aarch64")
        .TypeConstraint<int32_t>("Tdata"),
    ANNC_FORWARD(
        auto_spec_i32_impl,
        (),
        ()))
