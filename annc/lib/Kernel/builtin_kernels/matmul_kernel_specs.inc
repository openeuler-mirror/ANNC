ANNC_KERNEL(
    Name("MatMul").Backend("aarch64"),
    ANNC_FORWARD(
        matmul_aarch64_impl,
        (AnncMemRef2DF32* lhs, AnncMemRef2DF32* rhs, AnncMemRef2DF32* output),
        (lhs, rhs, output)))
