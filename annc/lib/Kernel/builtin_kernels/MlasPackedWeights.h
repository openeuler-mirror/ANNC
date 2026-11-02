#ifndef ANNC_BUILTIN_MLAS_PACKED_WEIGHTS_H
#define ANNC_BUILTIN_MLAS_PACKED_WEIGHTS_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

#include "Context/KernelContextUtils.h"
#include "mlas.h"

namespace annc::kernels {

// MLAS-specific scratch, owned by the host's per-call temporary allocator.
// Pack the effective input view, including constants substituted by lowering.
class MlasPackedWeights {
 public:
  AnncStatusCode prepare(KernelContext& context, const AnncTensorView& weight) {
    const char* enabled = std::getenv("ANNC_EP_PACKED_B");
    if (!enabled || !*enabled || std::strcmp(enabled, "0") == 0)
      return ANNC_STATUS_OK;
    const size_t k = weight.sizes[weight.rank - 2];
    const size_t n = weight.sizes[weight.rank - 1];
    if (!k || !n) return ANNC_STATUS_OK;
    size_t slices = 1;
    for (uint32_t i = 0; i + 2 < weight.rank; ++i) {
      const size_t dim = weight.sizes[i];
      if (!dim) return ANNC_STATUS_OK;
      if (slices > std::numeric_limits<size_t>::max() / dim)
        return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                             "packed weight slice count overflow");
      slices *= dim;
    }
    const size_t alignment = MlasGetPreferredBufferAlignment();
    const size_t packed =
        MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, n, k, nullptr);
    if (!packed) return ANNC_STATUS_OK;  // MLAS has no packed implementation.
    const size_t limit = std::numeric_limits<size_t>::max();
    if (!alignment || (alignment & (alignment - 1)) ||
        packed > limit - (alignment - 1))
      return context.error(ANNC_STATUS_INTERNAL, "invalid MLAS packing size");
    stride_ = (packed + alignment - 1) & ~(alignment - 1);
    if (slices > (limit - alignment) / stride_)
      return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                           "packed weight allocation overflow");
    const size_t bytes = slices * stride_ + alignment;
    const size_t words =
        bytes / sizeof(int64_t) + (bytes % sizeof(int64_t) != 0);
    if (words > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
      return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                           "packed weight allocation too large");
    const int64_t dims[] = {static_cast<int64_t>(words)};
    AnncTensorView storage{};
    const auto code = context.allocateTemp<int64_t, 1>(dims, &storage);
    if (code != ANNC_STATUS_OK) return code;
    void* pointer = contextData<int64_t>(storage);
    size_t space = bytes;
    if (!std::align(alignment, slices * stride_, pointer, space))
      return context.error(ANNC_STATUS_INTERNAL, "cannot align packed weights");
    data_ = static_cast<unsigned char*>(pointer);
    const float* raw = contextData<float>(weight);
    for (size_t slice = 0; slice < slices; ++slice)
      MlasGemmPackB(CblasNoTrans, CblasNoTrans, n, k, raw + slice * k * n, n,
                    data_ + slice * stride_, nullptr);
    const char* debug = std::getenv("ANNC_EP_DEBUG");
    if (debug && *debug && std::strcmp(debug, "0") != 0)
      std::fprintf(stderr, "[annc-kernel] packed-B temp: rank=%u slices=%zu\n",
                   weight.rank, slices);
    return context.status();
  }

  const float* slice(size_t index) const {
    return data_ ? reinterpret_cast<const float*>(data_ + index * stride_)
                 : nullptr;
  }

 private:
  unsigned char* data_ = nullptr;
  size_t stride_ = 0;
};

}  // namespace annc::kernels

#endif
