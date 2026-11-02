#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>

#include "Context/KernelContextUtils.h"
#include "fingerprint64.h"

namespace {

AnncStatusCode kp_fused_embedding_hash_bucket_aarch64_impl(
    annc::kernels::KernelContext& context) {
  const auto* input = context.getInput(0);
  const auto* embedding_weight = context.getInput(1);
  if (!input || !embedding_weight) return context.status();
  auto num_buckets = context.getI64Attr(0);
  if (!num_buckets)
    return context.status();
  auto* output = context.allocateOutput<float>(
      0, {input->sizes[0], embedding_weight->sizes[1]});
  if (!output) return context.status();

  const int64_t batch = output->sizes[0];
  const int64_t embedding_size = output->sizes[1];
  const int64_t input_size = input->sizes[0];
  const int64_t weight_rows = embedding_weight->sizes[0];
  const int64_t weight_cols = embedding_weight->sizes[1];

  if (batch < 0 || embedding_size < 0 || input_size < 0 || weight_rows < 0 ||
      weight_cols < 0 || *num_buckets < 0) {
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        "[ANNC Kernel] KPFusedDnnEmbeddingWithHashBucket failed: "
        "negative shape or bucket count\n");
  }
  if (output->offset < 0 || input->offset < 0 || embedding_weight->offset < 0) {
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        "[ANNC Kernel] KPFusedDnnEmbeddingWithHashBucket failed: "
        "negative memref offset\n");
  }
  if (input_size != batch || weight_cols != embedding_size) {
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        "[ANNC Kernel] KPFusedDnnEmbeddingWithHashBucket failed: "
        "shape mismatch\n");
  }
  if (batch > 0 && embedding_size > 0 &&
      batch > std::numeric_limits<int64_t>::max() / embedding_size) {
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        "[ANNC Kernel] KPFusedDnnEmbeddingWithHashBucket failed: "
        "output element count overflow\n");
  }

  const int64_t output_elements = batch * embedding_size;
  if (output_elements > std::numeric_limits<int64_t>::max() /
                            static_cast<int64_t>(sizeof(float))) {
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        "[ANNC Kernel] KPFusedDnnEmbeddingWithHashBucket failed: "
        "output byte count overflow\n");
  }
  const int64_t usable_buckets = std::min(*num_buckets, weight_rows);

  if ((batch > 0 && !input->aligned) ||
      (output_elements > 0 && !output->aligned) ||
      (usable_buckets > 0 && embedding_size > 0 &&
       !embedding_weight->aligned)) {
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        "[ANNC Kernel] KPFusedDnnEmbeddingWithHashBucket failed: "
        "null memref data\n");
  }
  if ((batch > 0 && input->strides[0] != 1) ||
      (embedding_size > 0 &&
       (output->strides[0] != embedding_size || output->strides[1] != 1 ||
        embedding_weight->strides[0] != weight_cols ||
        embedding_weight->strides[1] != 1))) {
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        "[ANNC Kernel] KPFusedDnnEmbeddingWithHashBucket failed: "
        "non-contiguous memref is unsupported\n");
  }

  if (output_elements > 0) {
    std::memset(annc::kernels::contextData<float>(*output), 0,
                output_elements * sizeof(float));
  }
  if (batch <= 0 || embedding_size <= 0 || usable_buckets <= 0) {
    return context.status();
  }

  float* out = annc::kernels::contextData<float>(*output);
  const AnncStringRef* strings =
      annc::kernels::contextData<AnncStringRef>(*input);
  const float* weight = annc::kernels::contextData<float>(*embedding_weight);

  try {
    auto workers = context.getThreadPool();
    workers.parallelFor(batch, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        const AnncStringRef& item = strings[i];
        if (!item.data || item.size <= 0) {
          continue;
        }

        const uint64_t hash_value = annc::kernels::Fingerprint64(
            item.data, static_cast<size_t>(item.size));
        const uint64_t bucket =
            hash_value % static_cast<uint64_t>(usable_buckets);
        const float* src = weight + bucket * embedding_size;
        float* dst = out + i * embedding_size;
        if (embedding_size == 1) {
          dst[0] = src[0];
        } else {
          std::memcpy(dst, src, embedding_size * sizeof(float));
        }
      }
    });
    return context.status();
  } catch (const std::exception& ex) {
    return context.error(ANNC_STATUS_INTERNAL, ex.what());
  } catch (...) {
    return context.error(
        ANNC_STATUS_INTERNAL,
        "[ANNC Kernel] KPFusedDnnEmbeddingWithHashBucket failed: "
        "unknown exception\n");
  }
}

}  // namespace
