// FusedLayerNorm (ONNX reciprocal-multiply LayerNorm): the whole 11-node chain
//
//   mean  = ReduceMean(x, last axis, keepdims)
//   d     = Sub(x, mean)
//   var   = ReduceMean(Mul(d, d), last axis, keepdims)
//   rsqrt = Reciprocal(Sqrt(Add(var, eps)))
//   g     = Mul(rsqrt, gamma)
//   y     = Add(Mul(x, g), Sub(beta, Mul(mean, g)))
//
// becomes one row-wise kernel.  gamma/beta are rank-1 [cols]; eps is the rank-0
// scalar constant the graph adds to the variance (both are constants in the
// ONNX models, so they arrive as memref arguments, not attributes).
//
// Wrapper ABI (annc_execution_v2): (execution, input, gamma, beta, eps).  The
// output is allocated by the kernel through the execution context (slot 0,
// shape [rows, cols] read off the input descriptor) and the thread pool comes
// from that same context (resolveThreadPool: the pool the context carries,
// falling back to the pool injected into this library's TLS).
//
// Only f32 is registered -- every site in the target models is float32.

ANNC_KERNEL(
    Name("LayerNorm")
        .Backend("aarch64")
        .Abi("annc_execution_v2")
        .TypeConstraint<float>("T"),
    (const AnncExecutionContext* execution, AnncMemRef2DF32* input,
     AnncMemRef1DF32* gamma, AnncMemRef1DF32* beta, AnncMemRef0DF32* eps),
    {
        const auto status = layerNormImpl(
            ::annc::kernels::execution::resolveThreadPool(execution), execution,
            input, gamma, beta, eps);
        return ::annc::kernels::statusCode(status);
    })
