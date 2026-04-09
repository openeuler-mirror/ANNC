#include "Kernel/threadpool/ThreadPoolContext.h"

namespace {

thread_local annc::runtime::AnncThreadPoolCtx* g_current_thread_pool_ctx = nullptr;

} // namespace

namespace annc {
namespace runtime {

void setCurrentThreadPoolCtx(AnncThreadPoolCtx* ctx) {
    annc_set_current_threadpool_ctx(ctx);
}

AnncThreadPoolCtx* getCurrentThreadPoolCtx() {
    return annc_get_current_threadpool_ctx();
}

void clearCurrentThreadPoolCtx() {
    annc_clear_current_threadpool_ctx();
}

} // namespace runtime
} // namespace annc

extern "C" void annc_set_current_threadpool_ctx(annc::runtime::AnncThreadPoolCtx* ctx) {
    g_current_thread_pool_ctx = ctx;
}

extern "C" annc::runtime::AnncThreadPoolCtx* annc_get_current_threadpool_ctx() {
    return g_current_thread_pool_ctx;
}

extern "C" void annc_clear_current_threadpool_ctx() {
    g_current_thread_pool_ctx = nullptr;
}
