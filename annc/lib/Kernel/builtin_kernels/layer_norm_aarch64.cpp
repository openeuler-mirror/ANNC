#include <cmath>
#include <cstdint>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "Kernel/ExecutionContextUtils.h"
#include "Context/RowParallel.h"
#include "Kernel/KernelStatus.h"
#include "Kernel/MemRefTypes.h"
#include "Context/Parallel.h"
#include "Context/ThreadPool.h"

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
#if defined(__aarch64__)
// Vectorised row body, used once a row is long enough to amortise the setup --
// the same gate ORT's own layer-norm kernel uses for its NEON path (256, see
// annc_opt 0001 -> layer_norm_impl.cc ComputeJobFloatNeon, whose structure
// this follows: four 128-bit accumulators per pass, 16-element steps, scalar
// tail).  Same three passes and the same final factorisation as the scalar
// path below, so the only difference is the summation order inside the vector
// accumulators.
constexpr int64_t kNeonMinCols = 256;

void layerNormRowNeon(const float* xRow, const float* gamma, const float* beta,
                      float* outRow, int64_t cols, float cols_f,
                      float epsilon) {
  float32x4_t sum0 = vdupq_n_f32(0.0f);
  float32x4_t sum1 = vdupq_n_f32(0.0f);
  float32x4_t sum2 = vdupq_n_f32(0.0f);
  float32x4_t sum3 = vdupq_n_f32(0.0f);
  int64_t col = 0;
  for (; col + 16 <= cols; col += 16) {
    sum0 = vaddq_f32(sum0, vld1q_f32(xRow + col));
    sum1 = vaddq_f32(sum1, vld1q_f32(xRow + col + 4));
    sum2 = vaddq_f32(sum2, vld1q_f32(xRow + col + 8));
    sum3 = vaddq_f32(sum3, vld1q_f32(xRow + col + 12));
  }
  float sum = vaddvq_f32(
      vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
  for (; col < cols; ++col) sum += xRow[col];
  const float mean = sum / cols_f;
  const float32x4_t mean4 = vdupq_n_f32(mean);

  float32x4_t sq0 = vdupq_n_f32(0.0f);
  float32x4_t sq1 = vdupq_n_f32(0.0f);
  float32x4_t sq2 = vdupq_n_f32(0.0f);
  float32x4_t sq3 = vdupq_n_f32(0.0f);
  col = 0;
  for (; col + 16 <= cols; col += 16) {
    const float32x4_t d0 = vsubq_f32(vld1q_f32(xRow + col), mean4);
    const float32x4_t d1 = vsubq_f32(vld1q_f32(xRow + col + 4), mean4);
    const float32x4_t d2 = vsubq_f32(vld1q_f32(xRow + col + 8), mean4);
    const float32x4_t d3 = vsubq_f32(vld1q_f32(xRow + col + 12), mean4);
    sq0 = vmlaq_f32(sq0, d0, d0);
    sq1 = vmlaq_f32(sq1, d1, d1);
    sq2 = vmlaq_f32(sq2, d2, d2);
    sq3 = vmlaq_f32(sq3, d3, d3);
  }
  float sq = vaddvq_f32(vaddq_f32(vaddq_f32(sq0, sq1), vaddq_f32(sq2, sq3)));
  for (; col < cols; ++col) {
    const float d = xRow[col] - mean;
    sq += d * d;
  }
  const float rsqrt = 1.0f / std::sqrt(sq / cols_f + epsilon);
  const float32x4_t rsqrt4 = vdupq_n_f32(rsqrt);

  col = 0;
  for (; col + 4 <= cols; col += 4) {
    const float32x4_t factor = vmulq_f32(rsqrt4, vld1q_f32(gamma + col));
    const float32x4_t y = vaddq_f32(
        vmulq_f32(vld1q_f32(xRow + col), factor),
        vsubq_f32(vld1q_f32(beta + col), vmulq_f32(mean4, factor)));
    vst1q_f32(outRow + col, y);
  }
  for (; col < cols; ++col) {
    const float factor = rsqrt * gamma[col];
    outRow[col] = xRow[col] * factor + (beta[col] - mean * factor);
  }
}
#endif  // __aarch64__

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
  // One row is an elementwise normalize: a few cycles per element for the two
  // accumulation passes and the normalize pass, the row read twice, plus the
  // rsqrt that ends each row.
  auto options = annc::threadpool::rowOptions(
      annc::threadpool::cost::elementRow(cols, /*cycles=*/5,
                                         /*bytes_read=*/8) +
          /*row epilogue*/ 20,
      /*elements_per_row=*/3 * cols);
  ::annc::threadpool::parallel_for(
      thread_pool, rows, options,
      [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
          const float* xRow = x + row * cols;
          float* outRow = out + row * cols;
#if defined(__aarch64__)
          if (cols >= kNeonMinCols) {
            layerNormRowNeon(xRow, g, b, outRow, cols, cols_f, epsilon);
            continue;
          }
#endif

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
