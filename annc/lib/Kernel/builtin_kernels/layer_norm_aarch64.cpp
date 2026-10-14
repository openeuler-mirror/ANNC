#include <cmath>
#include <cstdint>

#include "Kernel/ExecutionContextUtils.h"
#include "parallel_grain.h"
#include "Kernel/KernelStatus.h"
#include "Kernel/MemRefTypes.h"
#include "Support/ThreadPool/Parallel.h"
#include "Support/ThreadPool/ThreadPool.h"

namespace {

// FusedLayerNorm, execution-v2 form: the ONNX (TF-exported)
// reciprocal-multiply LayerNorm chain
//
//   mean = ReduceMean(x, last axis, keepdims)
//   d    = Sub(x, mean)
//   var  = ReduceMean(Mul(d, d), last axis, keepdims)
//   rsqrt= Reciprocal(Sqrt(Add(var, eps)))
//   g    = Mul(rsqrt, gamma)
//   y    = Add(Mul(x, g), Sub(beta, Mul(mean, g)))
//
// collapsed into one kernel.  The last expression is algebraically
// (x - mean) * rsqrt * gamma + beta; the implementation keeps the source's
// factorisation (g applied to both x and mean) so the fp32 rounding matches
// the unfused graph as closely as the accumulation order allows.
//
//   x     : rank-2 [rows, cols] f32, contiguous row-major
//   gamma : rank-1 [cols] f32
//   beta  : rank-1 [cols] f32
//   eps   : rank-0 f32 (the scalar the graph adds to the variance)
//   output: rank-2 [rows, cols] f32 -- allocated here through the execution
//           context (slot 0); the row count is a runtime value, so the shape
//           is only known after reading the input descriptor.
annc::kernels::KernelStatus layerNormImpl(
    annc::threadpool::AnncThreadPool* thread_pool,
    const AnncExecutionContext* execution, AnncMemRef2DF32* input,
    AnncMemRef1DF32* gamma, AnncMemRef1DF32* beta, AnncMemRef0DF32* eps) {
  if (execution == nullptr || !input || !gamma || !beta || !eps) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  const int64_t rows = input->sizes[0];
  const int64_t cols = input->sizes[1];
  if (rows < 0 || cols <= 0) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  if (gamma->sizes[0] != cols || beta->sizes[0] != cols) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  // Contiguous row-major operands: the host builds the descriptors that way
  // (Kernel/ExecutionContextUtils.h -> makeMemRef), so this guards the
  // host/kernel contract rather than an arbitrary input layout.
  if (input->strides[1] != 1 || input->strides[0] != cols ||
      gamma->strides[0] != 1 || beta->strides[0] != 1) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  // Allocate in every case -- also when rows is zero -- because the host
  // checks that the call allocated every declared output slot.
  auto output = annc::kernels::execution::allocateOutput2DF32(
      execution, /*slot=*/0, rows, cols, ANNC_ALLOCATION_NONE);
  if (!output) {
    llvm::consumeError(output.takeError());
    return annc::kernels::KernelStatus::RuntimeError;
  }
  if (output->strides[1] != 1 || output->strides[0] != cols) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  if (rows == 0) return annc::kernels::KernelStatus::Success;

  const float* x = ANNC_MEMREF_DATA(*input);
  const float* g = ANNC_MEMREF_DATA(*gamma);
  const float* b = ANNC_MEMREF_DATA(*beta);
  const float epsilon = *ANNC_MEMREF_DATA(*eps);
  float* out = ANNC_MEMREF_DATA(*output);

  // The moments accumulate in f32, like the graph's own ReduceMean (ORT
  // accumulates in the tensor's element type): a double accumulator would be
  // more accurate than the source but would also break bit-fidelity with it.
  const float cols_f = static_cast<float>(cols);

  // Rows are independent; each row is read twice (mean, variance) and written
  // once, all within one cache-resident line set at the sizes this targets.
  // Each row is read twice and written once, so its cost is ~3*cols element
  // ops; the grain keeps a shard on the right side of the host pool's
  // dispatch cost.
  annc::threadpool::ParallelForOptions options;
  options.grain_size = annc::kernels::grainForRowWork(3 * cols);
  ::annc::threadpool::parallel_for(
      thread_pool, rows, options,
      [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
          const float* xRow = x + row * cols;
          float* outRow = out + row * cols;

          float sum = 0.0f;
          for (int64_t col = 0; col < cols; ++col) sum += xRow[col];
          const float mean = sum / cols_f;

          float sq = 0.0f;
          for (int64_t col = 0; col < cols; ++col) {
            const float d = xRow[col] - mean;
            sq += d * d;
          }
          const float variance = sq / cols_f;
          const float rsqrt = 1.0f / std::sqrt(variance + epsilon);

          for (int64_t col = 0; col < cols; ++col) {
            const float factor = rsqrt * g[col];
            outRow[col] = xRow[col] * factor + (b[col] - mean * factor);
          }
        }
      });

  return annc::kernels::KernelStatus::Success;
}

}  // namespace
