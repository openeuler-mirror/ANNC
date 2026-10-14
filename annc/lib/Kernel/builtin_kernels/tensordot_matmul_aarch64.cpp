#include <cstdint>

#include "Kernel/ExecutionContextUtils.h"
#include "parallel_grain.h"
#include "Kernel/KernelStatus.h"
#include "Kernel/MemRefTypes.h"
#include "Support/ThreadPool/Parallel.h"
#include "Support/ThreadPool/ThreadPool.h"

namespace {

// FusedTensordotMatMul, execution-v2 form.
//
//   input : rank-3 [d0, d1, K] (contiguous row-major)
//   weight: rank-2 [K, N]
//   output: rank-3 [d0, d1, N] -- allocated here through the execution context
//           (slot 0) because the fused output's sizes follow the input's
//           leading dims, which are runtime values; the shape is only known
//           after reading the input descriptor.
//
//   out[i, j, n] = sum_k input[i, j, k] * weight[k, n]
//
// The fused region being replaced is the ONNX Tensordot block
// Reshape(MatMul(Reshape(x, [d0*d1, K]), W), [d0, d1, N]) plus the shape
// arithmetic that computes the two Reshape targets — 16 ops in ATIR, of which
// 15 are metadata (Transpose, the twelve shape-chain ops, and the two
// Reshapes).  In row-major layout collapsing the leading dimensions moves no
// data, so all of them are pure metadata: the fused op is the same matmul
// over the unchanged input buffer, with m = d0 * d1.  This mirrors ORT's
// FusedTensordotMatMul kernel
// (onnxruntime/contrib_ops/cpu/fused_tensordot_matmul.cc).
annc::kernels::KernelStatus tensordotMatMulImpl(
    annc::threadpool::AnncThreadPool* thread_pool,
    const AnncExecutionContext* execution, AnncMemRef3DF32* input,
    AnncMemRef2DF32* weight) {
  if (!execution || !input || !weight) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  const int64_t d0 = input->sizes[0];
  const int64_t d1 = input->sizes[1];
  const int64_t k = input->sizes[2];
  const int64_t n = weight->sizes[1];
  if (d0 < 0 || d1 < 0 || k < 0 || n < 0) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  // The pattern only fuses sites whose contract dim matches the weight's
  // first dim; re-check here so a mis-specialized call fails loudly instead
  // of producing a wrong result.
  if (weight->sizes[0] != k) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  // The kernel indexes both operands as contiguous row-major (x + row * k,
  // w + t * n). The host builds the descriptors as contiguous row-major
  // (Kernel/ExecutionContextUtils.h -> makeMemRef), so this check guards
  // the host/kernel contract rather than an arbitrary input layout:
  // non-contiguous inputs are not supported and must be rejected here.
  if (input->strides[2] != 1 || input->strides[1] != k ||
      input->strides[0] != d1 * k || weight->strides[1] != 1 ||
      weight->strides[0] != n) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  const int64_t m = d0 * d1;
  // Output shape [d0, d1, n] is data-dependent (the leading dims come from the
  // input descriptor), so it is requested from the execution context rather
  // than received as a preallocated argument.  Allocate in every case -- also
  // when m or n is zero -- because the host checks that the call allocated
  // every declared output slot.
  auto output = annc::kernels::execution::allocateOutput3DF32(
      execution, /*slot=*/0, d0, d1, n, ANNC_ALLOCATION_NONE);
  if (!output) {
    llvm::consumeError(output.takeError());
    return annc::kernels::KernelStatus::RuntimeError;
  }
  if (m == 0 || n == 0) {
    return annc::kernels::KernelStatus::Success;
  }

  float* out = ANNC_MEMREF_DATA(*output);
  if (k == 0) {
    for (int64_t i = 0; i < m * n; ++i) {
      out[i] = 0.0f;
    }
    return annc::kernels::KernelStatus::Success;
  }

  const float* x = ANNC_MEMREF_DATA(*input);
  const float* w = ANNC_MEMREF_DATA(*weight);

  // Rows of the collapsed [m, K] view are independent, so the row loop is the
  // natural parallel split (same shape as the builtin MatMul kernel in
  // matmul_aarch64.cpp).  M is tiny at the sites this fusion targets
  // (batch * sequence), so the grain stays at a whole row.
  // A row costs k*n multiply-accumulates; the grain keeps one thread on the
  // sites whose k*n is small instead of paying the host pool's dispatch.
  annc::threadpool::ParallelForOptions options;
  options.grain_size = annc::kernels::grainForRowWork(k * n);
  ::annc::threadpool::parallel_for(
      thread_pool, m, options,
      [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
          const float* xRow = x + row * k;
          float* outRow = out + row * n;
          for (int64_t col = 0; col < n; ++col) {
            const float* wCol = w + col;
            float sum = 0.0f;
            for (int64_t t = 0; t < k; ++t) {
              sum += xRow[t] * wCol[t * n];
            }
            outRow[col] = sum;
          }
        }
      });

  return annc::kernels::KernelStatus::Success;
}

}  // namespace
