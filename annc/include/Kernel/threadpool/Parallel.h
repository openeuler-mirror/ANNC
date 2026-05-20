#ifndef ANNC_PARALLEL_H
#define ANNC_PARALLEL_H

#include <cstdint>
#include <functional>

#include "Kernel/threadpool/ThreadPool.h"

namespace annc {
namespace kernels {

int num_threads(threadpool::AnncThreadPool* thread_pool);
bool in_parallel_region(threadpool::AnncThreadPool* thread_pool);

void parallel_for(
    threadpool::AnncThreadPool* thread_pool,
    std::int64_t total,
    const threadpool::ParallelForOptions& options,
    const std::function<void(std::int64_t, std::int64_t)>& fn);

void parallel_for(
    threadpool::AnncThreadPool* thread_pool,
    std::int64_t total,
    const std::function<void(std::int64_t, std::int64_t)>& fn);

} // namespace kernels
} // namespace annc

#endif // ANNC_PARALLEL_H
