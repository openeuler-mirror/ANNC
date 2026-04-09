#include "Kernel/threadpool/Parallel.h"

#include <algorithm>

namespace annc {
namespace kernels {

namespace {

void runSerial(std::int64_t begin,
               std::int64_t end,
               const std::function<void(std::int64_t, std::int64_t)>& fn) {
    if (begin >= end) {
        return;
    }
    fn(begin, end);
}

struct ParallelForThunk {
    const std::function<void(std::int64_t, std::int64_t)>* fn = nullptr;
};

void invokeParallelForThunk(std::int64_t begin,
                            std::int64_t end,
                            int /*worker_id*/,
                            void* user_data) {
    auto* thunk = static_cast<ParallelForThunk*>(user_data);
    (*thunk->fn)(begin, end);
}

} // namespace

int num_threads(runtime::AnncThreadPoolCtx* ctx) {
    if (ctx == nullptr || ctx->num_threads == nullptr) {
        return 1;
    }

    const int threads = ctx->num_threads(ctx->impl);
    return threads > 0 ? threads : 1;
}

bool in_parallel_region(runtime::AnncThreadPoolCtx* ctx) {
    if (ctx == nullptr || ctx->in_parallel_region == nullptr) {
        return false;
    }
    return ctx->in_parallel_region(ctx->impl);
}

void parallel_for(
    runtime::AnncThreadPoolCtx* ctx,
    std::int64_t begin,
    std::int64_t end,
    std::int64_t grain_size,
    const std::function<void(std::int64_t, std::int64_t)>& fn) {
    if (begin >= end) {
        return;
    }

    grain_size = std::max<std::int64_t>(grain_size, 1);
    const bool should_run_parallel =
        ctx != nullptr &&
        ctx->parallel_for != nullptr &&
        (end - begin) > grain_size &&
        num_threads(ctx) > 1 &&
        !in_parallel_region(ctx);

    if (!should_run_parallel) {
        runSerial(begin, end, fn);
        return;
    }

    ParallelForThunk thunk{&fn};
    ctx->parallel_for(
        ctx->impl,
        begin,
        end,
        grain_size,
        &invokeParallelForThunk,
        &thunk);
}

} // namespace kernels
} // namespace annc
