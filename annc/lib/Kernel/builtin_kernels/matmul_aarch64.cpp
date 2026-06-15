#include "Kernel/threadpool/ThreadPool.h"
#include "Kernel/MemRefTypes.h"
#include "Kernel/threadpool/Parallel.h"

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

    annc::kernels::parallel_for(thread_pool, M, [&](int64_t begin, int64_t end) {
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
