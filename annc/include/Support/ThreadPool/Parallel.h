#ifndef ANNC_PARALLEL_H
#define ANNC_PARALLEL_H

#include <cstdint>
#include <functional>

#include "Support/ThreadPool/ThreadPool.h"

namespace annc {
namespace threadpool {

using ParallelForCallback = void (*)(void* context, std::int64_t begin,
                                     std::int64_t end);

int num_threads(AnncThreadPool* thread_pool);
bool in_parallel_region(AnncThreadPool* thread_pool);

void parallel_for(
    AnncThreadPool* thread_pool,
    std::int64_t total,
    const ParallelForOptions& options,
    const std::function<void(std::int64_t, std::int64_t)>& fn);

void parallel_for(
    AnncThreadPool* thread_pool,
    std::int64_t total,
    const std::function<void(std::int64_t, std::int64_t)>& fn);

} // namespace threadpool
} // namespace annc

extern "C" std::int64_t annc_threadpool_num_threads();
extern "C" std::int32_t annc_threadpool_in_parallel();
extern "C" void annc_threadpool_parallel_for(
    std::int64_t total, std::int64_t cost_per_unit,
    annc::threadpool::ParallelForCallback callback, void* context);

#endif // ANNC_PARALLEL_H
