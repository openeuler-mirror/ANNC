#ifndef ANNC_PARALLEL_H
#define ANNC_PARALLEL_H

#include <cstdint>
#include <functional>

#include "Kernel/threadpool/ThreadPoolContext.h"

namespace annc {
namespace kernels {

int num_threads(runtime::AnncThreadPoolCtx* ctx);
bool in_parallel_region(runtime::AnncThreadPoolCtx* ctx);

void parallel_for(
    runtime::AnncThreadPoolCtx* ctx,
    std::int64_t begin,
    std::int64_t end,
    std::int64_t grain_size,
    const std::function<void(std::int64_t, std::int64_t)>& fn);

} // namespace kernels
} // namespace annc

#endif // ANNC_PARALLEL_H
