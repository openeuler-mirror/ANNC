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

    // The host's own pool object, for kernels that call a host-native library
    // entry with it.  The ORT EP hands over ORT's operator pool here: MLAS
    // takes that very object (mlas.h's MLAS_THREADPOOL is a type alias of
    // onnxruntime::concurrency::ThreadPool), so a kernel that links MLAS
    // passes this pointer to MlasGemmBatch and MLAS owns the threading.
    // Null means "nothing to hand over" -- a host with no such pool, or one
    // whose ABI the kernel must not touch -- and the kernel must fall back to
    // its own path (parallel_for or serial).  The default hands over nothing,
    // so a host that does not override it needs no change; the slot is
    // appended at the end of the vtable, which the same-tree build rule for
    // host and kernel already covers.
    virtual void* native_pool() const { return nullptr; }
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
