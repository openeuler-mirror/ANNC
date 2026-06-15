#include "Kernel/MemRefTypes.h"
#include "Kernel/threadpool/Parallel.h"
#include "Kernel/threadpool/ThreadPool.h"
#include "fingerprint64.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace {

void kp_fused_embedding_hash_bucket_aarch64_impl(
    annc::threadpool::AnncThreadPool* thread_pool,
    AnncMemRef2DF32* output,
    AnncMemRef1DString* input,
    AnncMemRef2DF32* embedding_weight,
    int64_t num_buckets) {
    float* out = ANNC_MEMREF_DATA(*output);
    const AnncStringRef* strings = ANNC_MEMREF_DATA(*input);
    const float* weight = ANNC_MEMREF_DATA(*embedding_weight);

    const int64_t batch = output->sizes[0];
    const int64_t embedding_size = output->sizes[1];
    const int64_t weight_rows = embedding_weight->sizes[0];
    const int64_t usable_buckets = std::min(num_buckets, weight_rows);

    std::memset(out, 0, batch * embedding_size * sizeof(float));
    if (batch <= 0 || embedding_size <= 0 || usable_buckets <= 0) {
        return;
    }

    annc::kernels::parallel_for(thread_pool, batch, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const AnncStringRef& item = strings[i];
            if (!item.data || item.size <= 0) {
                continue;
            }

            const uint64_t hash_value =
                annc::kernels::Fingerprint64(
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
}

} // namespace
