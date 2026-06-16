#include "Kernel/threadpool/ThreadPool.h"
#include <iostream>
#include <thread>

namespace {

thread_local annc::threadpool::AnncThreadPool* g_current_thread_pool = nullptr;

} // namespace

namespace annc {
namespace threadpool {

void setCurrentThreadPool(AnncThreadPool* thread_pool) {
    g_current_thread_pool = thread_pool;
}

AnncThreadPool* getCurrentThreadPool() {
    std::cout << "get_tid=" << std::this_thread::get_id() << std::endl;
    return g_current_thread_pool;
}

void clearCurrentThreadPool() {
    g_current_thread_pool = nullptr;
}

} // namespace threadpool
} // namespace annc

extern "C" void annc_set_current_threadpool(annc::threadpool::AnncThreadPool* thread_pool) {
    annc::threadpool::setCurrentThreadPool(thread_pool);
}

extern "C" annc::threadpool::AnncThreadPool* annc_get_current_threadpool() {
    return annc::threadpool::getCurrentThreadPool();
}

extern "C" void annc_clear_current_threadpool() {
    annc::threadpool::clearCurrentThreadPool();
}
