#ifndef ANNC_THREAD_POOL_H
#define ANNC_THREAD_POOL_H

#include <cstdint>
#include <functional>
#include <optional>

namespace annc {
namespace threadpool {

struct ParallelForOptions {
    std::optional<std::int64_t> cost_per_unit;
    std::optional<std::int64_t> grain_size;
};

class AnncThreadPool {
public:
    virtual ~AnncThreadPool() = default;

    virtual int num_threads() const = 0;
    virtual bool in_parallel_region() const = 0;
    virtual void parallel_for(
        std::int64_t total,
        const ParallelForOptions& options,
        const std::function<void(std::int64_t, std::int64_t)>& fn) = 0;
};

void setCurrentThreadPool(AnncThreadPool* thread_pool);
AnncThreadPool* getCurrentThreadPool();
void clearCurrentThreadPool();

} // namespace threadpool
} // namespace annc

extern "C" void annc_set_current_threadpool(annc::threadpool::AnncThreadPool* thread_pool);
extern "C" annc::threadpool::AnncThreadPool* annc_get_current_threadpool();
extern "C" void annc_clear_current_threadpool();

#endif // ANNC_THREAD_POOL_H
