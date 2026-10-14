#include <cstdint>

#include "Kernel/ExecutionContextUtils.h"
#include "Support/ThreadPool/RowParallel.h"
#include "Kernel/KernelStatus.h"
#include "Kernel/MemRefTypes.h"
#include "Support/ThreadPool/Parallel.h"
#include "Support/ThreadPool/ThreadPool.h"

namespace {

// One output row: out[0..n) = bias[0..n) + sum_k xRow[k] * wRow[k * n + 0..n).
// The 16-wide blocking keeps the accumulators in registers for the whole k
// loop (two memory reads per four FMAs instead of a load+store per FMA); the
// tail (n % 16) uses the plain element-at-a-time path, which keeps the kernel
// correct for any N.  The bias is added after the full k sum, matching the
// source chain (MatMul then Add) -- seeding the accumulator with it would
// round differently.
inline void batchedMatMulAddRow(const float* xRow, const float* wRow,
                                const float* biasRow, float* outRow, int64_t k,
                                int64_t n) {
  constexpr int64_t kBlock = 16;
  int64_t col = 0;
  for (; col + kBlock <= n; col += kBlock) {
    float acc[kBlock];
    for (int64_t i = 0; i < kBlock; ++i) acc[i] = 0.0f;
    for (int64_t t = 0; t < k; ++t) {
      const float factor = xRow[t];
      const float* wTail = wRow + t * n + col;
      for (int64_t i = 0; i < kBlock; ++i) acc[i] += factor * wTail[i];
    }
    for (int64_t i = 0; i < kBlock; ++i) {
      outRow[col + i] = acc[i] + biasRow[col + i];
    }
  }
  for (; col < n; ++col) {
    float acc = 0.0f;
    for (int64_t t = 0; t < k; ++t) acc += xRow[t] * wRow[t * n + col];
    outRow[col] = acc + biasRow[col];
  }
}

// FusedBatchedMatMulAdd, rank-4 form: the target models' per-domain projection
// sites, e.g. data [2, 1, M, 32] x W [2, 1, 32, 2] + bias [2, 1, 1, 2].
//
//   out[i, j, t, n] = bias[i, j, 0, n] + sum_k data[i, j, t, k] * W[i, j, k, n]
//
// data and W broadcast their leading two dims (size 1 means "reuse batch 0",
// numpy-style, as ONNX MatMul does); the output batch comes from the bias,
// which the matcher pinned to the graph's own batch dims.  Rank-4 memrefs are
// structs of their own (MemRefTypes.h), so this driver wraps the shared row
// routine rather than the rank-3 one.
annc::kernels::KernelStatus batchedMatMulAddRank4Impl(
    annc::threadpool::AnncThreadPool* thread_pool,
    const AnncExecutionContext* execution, AnncMemRef4DF32* data,
    AnncMemRef4DF32* weight, AnncMemRef4DF32* bias) {
  if (execution == nullptr || !data || !weight || !bias) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  const int64_t d0 = data->sizes[0];
  const int64_t d1 = data->sizes[1];
  const int64_t m = data->sizes[2];
  const int64_t k = data->sizes[3];
  const int64_t w0 = weight->sizes[0];
  const int64_t w1 = weight->sizes[1];
  const int64_t n = weight->sizes[3];
  const int64_t out0 = bias->sizes[0];
  const int64_t out1 = bias->sizes[1];
  if (d0 <= 0 || d1 <= 0 || m < 0 || k < 0 || w0 <= 0 || w1 <= 0 || n < 0 ||
      out0 <= 0 || out1 <= 0) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  // Data/W broadcast to the output batch per dim (size 1 or the full extent);
  // bias must carry the output batch exactly. Re-check here so a
  // mis-specialized call fails loudly instead of reading out of bounds.
  if ((d0 != 1 && d0 != out0) || (d1 != 1 && d1 != out1) ||
      (w0 != 1 && w0 != out0) || (w1 != 1 && w1 != out1) ||
      weight->sizes[2] != k || bias->sizes[2] != 1 || bias->sizes[3] != n) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  // The kernel indexes all operands as contiguous row-major; a broadcast dim
  // is always indexed at 0, so only the dims the loop strides through are
  // checked. The host builds the descriptors (ExecutionContextUtils.h ->
  // makeMemRef).
  if (data->strides[3] != 1 || data->strides[2] != k ||
      (d1 > 1 && data->strides[1] != m * k) ||
      (d0 > 1 && data->strides[0] != d1 * m * k) ||
      weight->strides[3] != 1 || weight->strides[2] != n ||
      (w1 > 1 && weight->strides[1] != k * n) ||
      (w0 > 1 && weight->strides[0] != w1 * k * n) ||
      bias->strides[3] != 1 || bias->strides[2] != n ||
      (out1 > 1 && bias->strides[1] != n) ||
      (out0 > 1 && bias->strides[0] != out1 * n)) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  // Allocate in every case -- also when rows is zero -- because the host
  // checks that the call allocated every declared output slot.
  auto output = annc::kernels::execution::allocateOutput4DF32(
      execution, /*slot=*/0, out0, out1, m, n, ANNC_ALLOCATION_NONE);
  if (!output) {
    llvm::consumeError(output.takeError());
    return annc::kernels::KernelStatus::RuntimeError;
  }
  if (output->strides[3] != 1 || output->strides[2] != n ||
      output->strides[1] != m * n || output->strides[0] != out1 * m * n) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  const int64_t rows = out0 * out1 * m;
  if (rows == 0 || n == 0) {
    return annc::kernels::KernelStatus::Success;
  }

  float* out = ANNC_MEMREF_DATA(*output);
  const float* x = ANNC_MEMREF_DATA(*data);
  const float* w = ANNC_MEMREF_DATA(*weight);
  const float* b = ANNC_MEMREF_DATA(*bias);

  // Same row contract as the rank-3 driver: one row is a k-length dot per
  // output element, and the row work k*n stays well below the elementwise
  // kernels' -- small sites keep running serial.
  auto options = annc::threadpool::rowOptions(
      annc::threadpool::cost::dotRow(k, n), /*elements_per_row=*/k * n);
  ::annc::threadpool::parallel_for(
      thread_pool, rows, options, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
          const int64_t t = row % m;
          const int64_t batch = row / m;
          const int64_t i = batch / out1;
          const int64_t j = batch % out1;
          const int64_t di = d0 == 1 ? 0 : i;
          const int64_t dj = d1 == 1 ? 0 : j;
          const int64_t wi = w0 == 1 ? 0 : i;
          const int64_t wj = w1 == 1 ? 0 : j;
          batchedMatMulAddRow(x + ((di * d1 + dj) * m + t) * k,
                              w + (wi * w1 + wj) * k * n, b + batch * n,
                              out + row * n, k, n);
        }
      });
  return annc::kernels::KernelStatus::Success;
}

// FusedBatchedMatMulAdd, rank-3 form: the ONNX pattern
//
//   MatMul(input [..., K], W [B, K, N]) -> Add(., bias [B, 1, N])
//
// collapsed into one kernel (the target models' attention QKV projections:
// sites of [1, 50, 12] x [1, 12, 16] + [1, 1, 16]). Data and weights can
// broadcast their batch dimension; bias carries the actual output batch.
//
//   out[i, j, n] = bias[i, 0, n] + sum_k data[i, j, k] * W[i, k, n]
//
//   data  : rank-3 [d0, d1, K] f32, contiguous row-major
//   weight: rank-3 [B, K, N]  f32, contiguous row-major
//   bias  : rank-3 [B, 1, N]  f32, contiguous row-major
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

  const int64_t dataBatch = data->sizes[0];
  const int64_t weightBatch = weight->sizes[0];
  const int64_t d0 = dataBatch == 1 ? weightBatch : dataBatch;
  const int64_t d1 = data->sizes[1];
  const int64_t k = data->sizes[2];
  const int64_t n = weight->sizes[2];
  if (d0 < 0 || dataBatch < 0 || weightBatch < 0 || d1 < 0 || k < 0 || n < 0) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  const int64_t biasBatch = bias->sizes[0];
  // Data/weight batches must broadcast to the output batch; bias must match
  // it exactly. Re-check here so a mis-specialized call fails loudly.
  if ((dataBatch != 1 && dataBatch != d0) ||
      (weightBatch != 1 && weightBatch != d0) || biasBatch != d0 ||
      weight->sizes[1] != k || bias->sizes[1] != 1 || bias->sizes[2] != n) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  // The kernel indexes all operands as contiguous row-major; the host builds
  // the descriptors that way (Kernel/ExecutionContextUtils.h -> makeMemRef).
  if (data->strides[2] != 1 || data->strides[1] != k ||
      data->strides[0] != d1 * k || weight->strides[2] != 1 ||
      weight->strides[1] != n ||
      bias->strides[2] != 1 || bias->strides[1] != n ||
      bias->strides[0] != n || weight->strides[0] != k * n) {
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

  // Rows are independent; the row work k*n stays well below the elementwise
  // kernels', so small sites keep running serial.
  auto options = annc::threadpool::rowOptions(
      annc::threadpool::cost::dotRow(k, n), /*elements_per_row=*/k * n);
  ::annc::threadpool::parallel_for(
      thread_pool, rows, options,
      [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
          const int64_t batchIndex = row / d1;
          batchedMatMulAddRow(x + (dataBatch == 1 ? row % d1 : row) * k,
                              w + (weightBatch == 1 ? 0 : batchIndex) * k * n,
                              b + batchIndex * n, out + row * n, k, n);
        }
      });

  return annc::kernels::KernelStatus::Success;
}

}  // namespace
