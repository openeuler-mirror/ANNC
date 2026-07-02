#include "Kernel/MemRefTypes.h"
#include "Kernel/KernelStatus.h"
#include "Kernel/threadpool/Parallel.h"
#include "Kernel/threadpool/ThreadPool.h"
#include "fingerprint64.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>

#include "llvm/Support/raw_ostream.h"

namespace {

annc::kernels::KernelStatus kp_fused_embedding_hash_bucket_aarch64_impl(
    annc::threadpool::AnncThreadPool* thread_pool,
    AnncMemRef2DF32* output,
    AnncMemRef1DString* input,
    AnncMemRef2DF32* embedding_weight,
    int64_t num_buckets) {
    if (!output || !input || !embedding_weight) {
        llvm::errs() << "[ANNC Kernel] KPFusedDnnEmbeddingWithHashBucket failed: "
                     "null memref argument\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }

    const int64_t batch = output->sizes[0];
    const int64_t embedding_size = output->sizes[1];
    const int64_t input_size = input->sizes[0];
    const int64_t weight_rows = embedding_weight->sizes[0];
    const int64_t weight_cols = embedding_weight->sizes[1];

    if (batch < 0 || embedding_size < 0 || input_size < 0 ||
        weight_rows < 0 || weight_cols < 0 || num_buckets < 0) {
        llvm::errs() << "[ANNC Kernel] KPFusedDnnEmbeddingWithHashBucket failed: "
                     "negative shape or bucket count\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    if (output->offset < 0 || input->offset < 0 ||
        embedding_weight->offset < 0) {
        llvm::errs() << "[ANNC Kernel] KPFusedDnnEmbeddingWithHashBucket failed: "
                     "negative memref offset\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    if (input_size != batch || weight_cols != embedding_size) {
        llvm::errs() << "[ANNC Kernel] KPFusedDnnEmbeddingWithHashBucket failed: "
                     "shape mismatch\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    if (batch > 0 && embedding_size > 0 &&
        batch > std::numeric_limits<int64_t>::max() / embedding_size) {
        llvm::errs() << "[ANNC Kernel] KPFusedDnnEmbeddingWithHashBucket failed: "
                     "output element count overflow\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }

    const int64_t output_elements = batch * embedding_size;
    if (output_elements > std::numeric_limits<int64_t>::max() /
                              static_cast<int64_t>(sizeof(float))) {
        llvm::errs() << "[ANNC Kernel] KPFusedDnnEmbeddingWithHashBucket failed: "
                     "output byte count overflow\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    const int64_t usable_buckets = std::min(num_buckets, weight_rows);

    if ((batch > 0 && !input->aligned) ||
        (output_elements > 0 && !output->aligned) ||
        (usable_buckets > 0 && embedding_size > 0 &&
         !embedding_weight->aligned)) {
        llvm::errs() << "[ANNC Kernel] KPFusedDnnEmbeddingWithHashBucket failed: "
                     "null memref data\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    if ((batch > 0 && input->strides[0] != 1) ||
        (embedding_size > 0 &&
         (output->strides[0] != embedding_size || output->strides[1] != 1 ||
          embedding_weight->strides[0] != weight_cols ||
          embedding_weight->strides[1] != 1))) {
        llvm::errs() << "[ANNC Kernel] KPFusedDnnEmbeddingWithHashBucket failed: "
                     "non-contiguous memref is unsupported\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }

    if (output_elements > 0) {
        std::memset(ANNC_MEMREF_DATA(*output), 0,
                    output_elements * sizeof(float));
    }
    if (batch <= 0 || embedding_size <= 0 || usable_buckets <= 0) {
        return annc::kernels::KernelStatus::Success;
    }

    float* out = ANNC_MEMREF_DATA(*output);
    const AnncStringRef* strings = ANNC_MEMREF_DATA(*input);
    const float* weight = ANNC_MEMREF_DATA(*embedding_weight);

    try {
        annc::kernels::parallel_for(
            thread_pool, batch, [&](int64_t begin, int64_t end) {
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
        return annc::kernels::KernelStatus::Success;
    } catch (const std::exception& ex) {
        llvm::errs() << "[ANNC Kernel] KPFusedDnnEmbeddingWithHashBucket failed: "
                     << ex.what() << '\n';
        return annc::kernels::KernelStatus::RuntimeError;
    } catch (...) {
        llvm::errs() << "[ANNC Kernel] KPFusedDnnEmbeddingWithHashBucket failed: "
                     "unknown exception\n";
        return annc::kernels::KernelStatus::UnknownError;
    }
}

} // namespace
