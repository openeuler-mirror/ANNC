#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "Context/KernelContextUtils.h"
#include "Context/RowParallel.h"

namespace {

// FusedMaskedMul, context-v3 form: the ONNX mask chain
//
//   Equal(ids, padding) -> Not -> Cast(f32) -> Unsqueeze(last axis)
//     -> [Tile(broadcast to N)] -> Mul(data_i, mask) -> y_i
//
// collapsed into one kernel per *site*, and a site is the whole subgraph: the
// chain plus every Mul it feeds, plus the element-wise path between those Muls
// (model1's Sum -> Div(by a constant), which the graph puts between two masked
// Muls -- a plugin-EP fused node cannot have an unsupported node on a path
// between two of its members, so those nodes come along and are computed here).
// The fused node's outputs are everything that subgraph provides to the rest of
// the graph: every Mul's result, plus the chain values still read outside
// (model3's Shape/Expand read the Cast output).
//
// Per element (p = rawMuls, q = fedMuls):
//
//   f      = (ids[b, t] != padding) ? 1.0f : 0.0f
//   y_i    = x_i * f            i < p, x_i = data 实参 argMap 的第 i 个 nibble
//   S      = sum of the p results above, then the remaining arguments (the
//            inputs the Sum node also carried)
//   d      = S / divisor
//   y_fed  = d * f
//   [flat] = f                  the pre-unsqueeze mask, when the graph reads it
//
// Every element is multiplied, including the kept ones (x1.0f) -- no memset
// shortcut -- so the fused result is bit-identical to the source chain on
// +-0 / NaN / inf inputs (inf * 0.0f -> NaN must survive).  mask*(mask*x) and
// mask*x are the same value (the factor is exactly 0 or 1), which is why two
// Muls on the same source share one argument.
//
// The arithmetic order of the Sum differs from the graph's when the Sum node
// interleaved the masked results with its other inputs; the difference is
// rounding only (recorded in the task's data dir).

// f32 的位模式还原（Div 的除数经属性传入；ABI 只有 i64 属性）。
float divisorFromBits(int64_t bits) {
  const uint32_t value32 = static_cast<uint32_t>(bits);
  float value = 0.0f;
  std::memcpy(&value, &value32, sizeof(value));
  return value;
}

// 第 i 个 data 实参（argMap 里 4 bit 一个）。
int64_t maskedMulArgIndex(int64_t argMap, int64_t i) {
  return (argMap >> (4 * i)) & 0xF;
}

// Allocates the site's outputs (slot order: raw results, fed results, flat) and
// runs the row loop.  `dataArgs` says how many of the arguments are data (the
// rest are the Sum's other inputs); the first `rawMuls` results are the ones
// multiplied by the row factor.
template <size_t Arity>
AnncStatusCode maskedMulSiteImpl(annc::kernels::KernelContext& context,
                                bool writeFlat) {
  std::array<const AnncTensorView*, Arity> data;
  for (size_t i = 0; i < Arity; ++i) data[i] = context.getInput(i);
  const auto* ids = context.getInput(Arity);
  std::array<int64_t, 8> attrs;
  for (size_t i = 0; i < attrs.size(); ++i) {
    auto value = context.getI64Attr(i);
    if (!value) return context.status();
    attrs[i] = *value;
  }
  const auto dataArgs = attrs[0];
  const auto rawMuls = attrs[1];
  const auto fedMuls = attrs[2];
  const auto argMap = attrs[3];
  const auto sumMask = attrs[4];
  const auto interiorKind = attrs[5];
  const auto divisorBits = attrs[6];
  const auto padding = attrs[7];
  if (!ids) return context.status();
  auto thread_pool = context.getThreadPool();
  // 实参个数是 data + Sum 的其它输入；输出个数是 Mul 的个数（raw + fed），两者
  // 不必相等——同一份 data 被多个 Mul 读时实参只出现一次（见 argMap）。
  if (dataArgs < 1 || rawMuls < 1 || rawMuls > 16 || fedMuls < 0 ||
      dataArgs > static_cast<int64_t>(Arity)) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }
  for (int64_t i = 0; i < rawMuls; ++i) {
    if (maskedMulArgIndex(argMap, i) >= dataArgs) {
      return ANNC_STATUS_INVALID_ARGUMENT;
    }
  }
  for (size_t i = 0; i < Arity; ++i) {
    if (data[i] == nullptr) return ANNC_STATUS_INVALID_ARGUMENT;
  }

  // 所有实参（data 与 Sum 的其它输入）必须同形，且前缀与 ids 对齐——kernel
  // 按行索引 ids、按行读写 data。
  const int64_t d0 = data[0]->sizes[0];
  const int64_t d1 = data[0]->sizes[1];
  const int64_t n = data[0]->sizes[2];
  if (d0 < 0 || d1 < 0 || n < 0) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }
  if (ids->sizes[0] != d0 || ids->sizes[1] != d1) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }
  for (size_t i = 0; i < Arity; ++i) {
    if (data[i]->sizes[0] != d0 || data[i]->sizes[1] != d1 ||
        data[i]->sizes[2] != n) {
      return ANNC_STATUS_INVALID_ARGUMENT;
    }
    if (n != 0 && (data[i]->strides[2] != 1 || data[i]->strides[1] != n ||
                   data[i]->strides[0] != d1 * n)) {
      return ANNC_STATUS_INVALID_ARGUMENT;
    }
  }
  if (ids->strides[1] != 1 || ids->strides[0] != d1) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }

  constexpr int64_t kMaxSiteOutputs = 64;
  constexpr int64_t kMaxSiteInputs = 32;
  // Validate generated metadata before sizing any allocation. This is normally
  // guaranteed by the spec generator, but keeps a malformed ABI attribute from
  // turning into an unbounded vector allocation.
  if (fedMuls > kMaxSiteOutputs ||
      rawMuls > kMaxSiteOutputs - fedMuls ||
      rawMuls + Arity - dataArgs > kMaxSiteInputs) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }
  // Allocate in every case -- also when rows is zero -- because the host checks
  // that the call allocated every declared output slot.
  std::vector<float *> out(static_cast<size_t>(rawMuls + fedMuls), nullptr);
  for (int64_t i = 0; i < rawMuls + fedMuls; ++i) {
    auto buffer = context.allocateOutput<float>(static_cast<uint32_t>(i),
                                                 {d0, d1, n});
    if (!buffer) {
      return ANNC_STATUS_ALLOCATION_FAILED;
    }
    if (n != 0 && (buffer->strides[2] != 1 || buffer->strides[1] != n ||
                   buffer->strides[0] != d1 * n)) {
      return ANNC_STATUS_INVALID_ARGUMENT;
    }
    out[i] = annc::kernels::contextData<float>(*buffer);
  }
  float *flat = nullptr;
  if (writeFlat) {
    auto buffer = context.allocateOutput<float>(
        static_cast<uint32_t>(rawMuls + fedMuls), {d0, d1});
    if (!buffer) {
      return ANNC_STATUS_ALLOCATION_FAILED;
    }
    if (buffer->strides[1] != 1 || buffer->strides[0] != d1) {
      return ANNC_STATUS_INVALID_ARGUMENT;
    }
    flat = annc::kernels::contextData<float>(*buffer);
  }

  const int64_t rows = d0 * d1;
  if (rows == 0 || n == 0) return ANNC_STATUS_OK;

  const float divisor = divisorFromBits(divisorBits);
  if (fedMuls > 0 && divisor == 0.0f) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }
  std::array<const float*, Arity> in{};
  for (size_t i = 0; i < Arity; ++i)
    in[i] = annc::kernels::contextData<float>(*data[i]);
  const int64_t *idValues = annc::kernels::contextData<int64_t>(*ids);
  // Decode site metadata once, outside the row/column loops. The sum inputs
  // retain their original order, including the initial +0 (or product 1).
  std::array<const float*, kMaxSiteInputs> rawInputs{};
  std::array<const float*, kMaxSiteInputs> sumInputs{};
  size_t sumInputCount = 0;
  for (int64_t i = 0; i < rawMuls; ++i) {
    rawInputs[i] = in[maskedMulArgIndex(argMap, i)];
    if (fedMuls > 0 && ((sumMask >> i) & 1)) sumInputs[sumInputCount++] = out[i];
  }
  if (fedMuls > 0) {
    for (int64_t j = dataArgs; j < static_cast<int64_t>(Arity); ++j)
      sumInputs[sumInputCount++] = in[j];
  }
  const bool product = interiorKind == 1;

  // One row is an elementwise masked multiply: a compare and a multiply per
  // element, reading f32 data and an i64 id per element.
  // The host's cost model must see the fused row. Charging one
  // multiply underestimates a site with several raw/fed outputs and makes
  // ORT keep it on too few workers; the patched elementwise kernels charge
  // every output's memory traffic in the same way.
  const int64_t outputCount = rawMuls + fedMuls + (writeFlat ? 1 : 0);
  const int64_t sumAdds = fedMuls > 0 ? static_cast<int64_t>(sumInputCount) : 0;
  const int64_t cyclesPerElement =
      2 * rawMuls + sumAdds + (fedMuls > 0 ? 1 + fedMuls : 0);
  const int64_t rawSumReads =
      fedMuls > 0 ? __builtin_popcountll(static_cast<unsigned long long>(sumMask)) : 0;
  const int64_t bytesReadPerElement =
      8 + 4 * rawMuls + 4 * (Arity - dataArgs) + 4 * rawSumReads;
  const int64_t bytesWrittenPerElement = 4 * outputCount;
  auto options = annc::threadpool::rowOptions(
      annc::threadpool::cost::elementRow(n, cyclesPerElement,
                                          bytesReadPerElement,
                                          bytesWrittenPerElement),
      n);
  annc::kernels::Shard(
      thread_pool.numThreads(), thread_pool, rows, options.cost_per_unit.value_or(0),
      [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
          const float factor = (idValues[row] != padding) ? 1.0f : 0.0f;
          for (int64_t i = 0; i < rawMuls; ++i) {
            const float *src = rawInputs[i] + row * n;
            float *dst = out[i] + row * n;
            int64_t c = 0;
#if defined(__aarch64__)
            const float32x4_t factors = vdupq_n_f32(factor);
            for (; c + 4 <= n; c += 4)
              vst1q_f32(dst + c, vmulq_f32(vld1q_f32(src + c), factors));
#endif
            for (; c < n; ++c) dst[c] = src[c] * factor;
          }
          if (fedMuls > 0) {
            // Column-wise so the row stays in cache: S needs the Sum's inputs
            // (the results just written, plus the remaining arguments), then
            // the fed results are written.
            int64_t c = 0;
#if defined(__aarch64__)
            const float32x4_t divisors = vdupq_n_f32(divisor);
            const float32x4_t factors = vdupq_n_f32(factor);
            for (; c + 4 <= n; c += 4) {
              float32x4_t sum = vdupq_n_f32(product ? 1.0f : 0.0f);
              for (size_t inputIndex = 0; inputIndex < sumInputCount; ++inputIndex) {
                const float* input = sumInputs[inputIndex];
                const float32x4_t value = vld1q_f32(input + row * n + c);
                sum = product ? vmulq_f32(sum, value) : vaddq_f32(sum, value);
              }
              // Exact division, not reciprocal multiplication: preserve the
              // source Div rounding and IEEE NaN/inf/signed-zero semantics.
              const float32x4_t scaled =
                  vmulq_f32(vdivq_f32(sum, divisors), factors);
              for (int64_t l = 0; l < fedMuls; ++l)
                vst1q_f32(out[rawMuls + l] + row * n + c, scaled);
            }
#endif
            for (; c < n; ++c) {
              float sum = product ? 1.0f : 0.0f;
              for (size_t inputIndex = 0; inputIndex < sumInputCount; ++inputIndex) {
                const float* input = sumInputs[inputIndex];
                sum = product ? sum * input[row * n + c]
                              : sum + input[row * n + c];
              }
              const float scaled = (sum / divisor) * factor;
              for (int64_t l = 0; l < fedMuls; ++l) {
                out[rawMuls + l][row * n + c] = scaled;
              }
            }
          }
          if (writeFlat) {
            if (flat != nullptr) flat[row] = factor;
          }
        }
      });
  return ANNC_STATUS_OK;
}

// Entry points for the kernel symbols (annc/lib/Kernel/builtin_kernels/
// masked_mul_kernel_specs.inc): one per argument count, plus a variant that also
// writes the pre-unsqueeze mask when the graph still reads it.  The
// context-v3 C signature is fixed per symbol, hence the repetition.

}  // namespace
