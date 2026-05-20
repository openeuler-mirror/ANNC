#include "Kernel/threadpool/Parallel.h"

#include <algorithm>

namespace annc {
namespace kernels {

namespace {

void runSerial(std::int64_t total,
               const std::function<void(std::int64_t, std::int64_t)>& fn) {
    if (total <= 0) {
        return;
    }
    fn(0, total);
}

} // namespace

int num_threads(threadpool::AnncThreadPool* thread_pool) {
    if (thread_pool == nullptr) {
        return 1;
    }

    const int threads = thread_pool->num_threads();
    return threads > 0 ? threads : 1;
}

bool in_parallel_region(threadpool::AnncThreadPool* thread_pool) {
    if (thread_pool == nullptr) {
        return false;
    }
    return thread_pool->in_parallel_region();
}

void parallel_for(
    threadpool::AnncThreadPool* thread_pool,
    std::int64_t total,
    const threadpool::ParallelForOptions& options,
    const std::function<void(std::int64_t, std::int64_t)>& fn) {
    if (total <= 0) {
        return;
    }

    const std::int64_t grain_size =
        std::max<std::int64_t>(options.grain_size.value_or(1), 1);
    const bool should_run_parallel =
        thread_pool != nullptr &&
        total > grain_size &&
        num_threads(thread_pool) > 1 &&
        !in_parallel_region(thread_pool);

    if (!should_run_parallel) {
        runSerial(total, fn);
        return;
    }

    thread_pool->parallel_for(total, options, fn);
}

void parallel_for(
    threadpool::AnncThreadPool* thread_pool,
    std::int64_t total,
    const std::function<void(std::int64_t, std::int64_t)>& fn) {
    parallel_for(thread_pool, total, threadpool::ParallelForOptions{}, fn);
}

} // namespace kernels
} // namespace annc
