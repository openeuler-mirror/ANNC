#include "Support/ThreadPool/ThreadPool.h"
#include "Kernel/MemRefTypes.h"
#include "Support/ThreadPool/Parallel.h"
#include "Support/ThreadPool/RowParallel.h"

namespace {

void matmul_aarch64_impl(annc::threadpool::AnncThreadPool* thread_pool,
                         AnncMemRef2DF32* output,
                         AnncMemRef2DF32* lhs,
                         AnncMemRef2DF32* rhs) {
    float* out = ANNC_MEMREF_DATA(*output);
    const float* A = ANNC_MEMREF_DATA(*lhs);
    const float* B = ANNC_MEMREF_DATA(*rhs);

    int64_t M = output->sizes[0];
    int64_t N = output->sizes[1];
    int64_t K = lhs->sizes[1];

    // A row is N dot products of length K, reported so the hosts that size
    // blocks from cost treat this kernel like the rest; the in-tree pool keeps
    // its own default.
    auto options = annc::threadpool::rowOptions(
        annc::threadpool::cost::dotRow(K, N), /*elements_per_row=*/K * N);
    annc::threadpool::parallel_for(
        thread_pool, M, options, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                out[i * N + j] = sum;
            }
        }
    });
}

} // namespace
