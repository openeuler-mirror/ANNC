#ifndef ANNC_THREAD_POOL_CONTEXT_H
#define ANNC_THREAD_POOL_CONTEXT_H

#include <cstdint>

namespace annc {
namespace runtime {

struct AnncThreadPoolCtx {
    void* impl = nullptr;

    int (*num_threads)(void* impl) = nullptr;
    bool (*in_parallel_region)(void* impl) = nullptr;
    void (*parallel_for)(
        void* impl,
        std::int64_t begin,
        std::int64_t end,
        std::int64_t grain_size,
        void (*fn)(std::int64_t, std::int64_t, int, void*),
        void* user_data) = nullptr;
};

void setCurrentThreadPoolCtx(AnncThreadPoolCtx* ctx);
AnncThreadPoolCtx* getCurrentThreadPoolCtx();
void clearCurrentThreadPoolCtx();

} // namespace runtime
} // namespace annc

extern "C" void annc_set_current_threadpool_ctx(annc::runtime::AnncThreadPoolCtx* ctx);
extern "C" annc::runtime::AnncThreadPoolCtx* annc_get_current_threadpool_ctx();
extern "C" void annc_clear_current_threadpool_ctx();

#endif // ANNC_THREAD_POOL_CONTEXT_H
