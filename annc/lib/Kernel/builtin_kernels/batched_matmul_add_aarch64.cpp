#include <cstdint>

#include "Kernel/ExecutionContextUtils.h"
#include "parallel_grain.h"
#include "Kernel/KernelStatus.h"
#include "Kernel/MemRefTypes.h"
#include "Support/ThreadPool/Parallel.h"
#include "Support/ThreadPool/ThreadPool.h"

namespace {

// FusedBatchedMatMulAdd, execution-v2 form: the ONNX pattern
//
//   MatMul(input [..., K], W [1, K, N]) -> Add(., bias [1, 1, N])
//
// collapsed into one kernel (the target models' attention QKV projections:
// sites of [1, 50, 12] x [1, 12, 16] + [1, 1, 16]).  The weight's and bias's
// leading dims are broadcast (batch 1), so the whole site is a small rank-2
// GEMM per row plus a bias initializer:
//
//   out[i, j, n] = bias[0, 0, n] + sum_k data[i, j, k] * W[0, k, n]
//
//   data  : rank-3 [d0, d1, K] f32, contiguous row-major
//   weight: rank-3 [1, K, N]  f32, contiguous row-major
//   bias  : rank-3 [1, 1, N]  f32, contiguous row-major
//   output: rank-3 [d0, d1, N] f32 -- allocated here through the execution
//           context (slot 0); the leading dims are runtime values, so the
//           shape is only known after reading the data descriptor.
//
// The epilogue order is the source chain's: accumulate the full k sum first,
// then add the bias (a different order would change the fp32 rounding versus
// the unfused graph).
annc::kernels::KernelStatus batchedMatMulAddImpl(
    annc::threadpool::AnncThreadPool* thread_pool,
    const AnncExecutionContext* execution, AnncMemRef3DF32* data,
    AnncMemRef3DF32* weight, AnncMemRef3DF32* bias) {
  if (execution == nullptr || !data || !weight || !bias) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  const int64_t d0 = data->sizes[0];
  const int64_t d1 = data->sizes[1];
  const int64_t k = data->sizes[2];
  const int64_t n = weight->sizes[2];
  if (d0 < 0 || d1 < 0 || k < 0 || n < 0) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  // The pattern only fuses sites whose leading dims are broadcast (1) and
  // whose K/N line up; re-check here so a mis-specialized call fails loudly.
  if (weight->sizes[0] != 1 || weight->sizes[1] != k ||
      bias->sizes[0] != 1 || bias->sizes[1] != 1 || bias->sizes[2] != n) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  // The kernel indexes all operands as contiguous row-major; the host builds
  // the descriptors that way (Kernel/ExecutionContextUtils.h -> makeMemRef).
  if (data->strides[2] != 1 || data->strides[1] != k ||
      data->strides[0] != d1 * k || weight->strides[2] != 1 ||
      weight->strides[1] != n || weight->strides[0] != k * n ||
      bias->strides[2] != 1 || bias->strides[1] != n ||
      bias->strides[0] != n) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  // Allocate in every case -- also when rows is zero -- because the host
  // checks that the call allocated every declared output slot.
  auto output = annc::kernels::execution::allocateOutput3DF32(
      execution, /*slot=*/0, d0, d1, n, ANNC_ALLOCATION_NONE);
  if (!output) {
    llvm::consumeError(output.takeError());
    return annc::kernels::KernelStatus::RuntimeError;
  }
  if (output->strides[2] != 1 || output->strides[1] != n ||
      output->strides[0] != d1 * n) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  const int64_t rows = d0 * d1;
  if (rows == 0 || n == 0) {
    return annc::kernels::KernelStatus::Success;
  }

  float* out = ANNC_MEMREF_DATA(*output);
  const float* x = ANNC_MEMREF_DATA(*data);
  const float* w = ANNC_MEMREF_DATA(*weight);
  const float* b = ANNC_MEMREF_DATA(*bias);

  // Rows are independent.  The n dimension is blocked to 16 elements (the
  // sites this targets have N = 12-16) so the accumulators live in registers
  // for the whole k loop: two memory reads per four FMAs instead of a
  // load+store per FMA.  The tail (n % 16) uses the plain element-at-a-time
  // path, which keeps the kernel correct for any N.
  constexpr int64_t kBlock = 16;
  // A row costs k*n multiply-accumulates, so the grain is much smaller than
  // for the elementwise kernels; small sites still stay serial.
  annc::threadpool::ParallelForOptions options;
  options.grain_size = annc::kernels::grainForRowWork(k * n);
  ::annc::threadpool::parallel_for(
      thread_pool, rows, options,
      [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
          const float* xRow = x + row * k;
          float* outRow = out + row * n;
          int64_t col = 0;
          for (; col + kBlock <= n; col += kBlock) {
            float acc[kBlock];
            for (int64_t i = 0; i < kBlock; ++i) acc[i] = 0.0f;
            for (int64_t t = 0; t < k; ++t) {
              const float factor = xRow[t];
              const float* wRow = w + t * n + col;
              for (int64_t i = 0; i < kBlock; ++i) {
                acc[i] += factor * wRow[i];
              }
            }
            // The bias is added after the full k sum, matching the source
            // chain (MatMul then Add); seeding the accumulator with it would
            // round differently.
            for (int64_t i = 0; i < kBlock; ++i) {
              outRow[col + i] = acc[i] + b[col + i];
            }
          }
          for (; col < n; ++col) {
            float acc = 0.0f;
            for (int64_t t = 0; t < k; ++t) {
              acc += xRow[t] * w[t * n + col];
            }
            outRow[col] = acc + b[col];
          }
        }
      });

  return annc::kernels::KernelStatus::Success;
}

}  // namespace
