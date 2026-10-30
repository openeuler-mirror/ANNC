#include <algorithm>
#include <exception>
#include <mutex>
#include <stdexcept>

#include "Context/ThreadPool.h"
#include "c_api.h"

namespace {

class ThreadPoolAdapter final : public annc::threadpool::AnncThreadPool {
 public:
  explicit ThreadPoolAdapter(const AnncThreadPoolCallbacks& callbacks)
      : callbacks_(callbacks) {}

  int num_threads() const override {
    return callbacks_.num_threads(callbacks_.context);
  }
  bool in_parallel_region() const override {
    return callbacks_.in_parallel_region(callbacks_.context) != 0;
  }

  void parallel_for(int64_t total,
                    const annc::threadpool::ParallelForOptions& options,
                    const std::function<void(int64_t, int64_t)>& fn) override {
    if (total <= 0) return;
    const int64_t block_size = options.grain_size.has_value()
                                   ? std::max<int64_t>(*options.grain_size, 1)
                                   : 0;
    const int64_t cost =
        std::max<int64_t>(options.cost_per_unit.value_or(1), 1);
    struct Task {
      const std::function<void(int64_t, int64_t)>& fn;
      std::mutex mutex;
      std::exception_ptr error;
    } task{fn, {}, {}};
    // A kernel exception must not unwind through a TF callback compiled with
    // another ABI. Rethrow on this side after synchronous scheduling finishes.
    const auto invoke = [](void* context, int64_t begin, int64_t end) {
      auto& task = *static_cast<Task*>(context);
      try {
        task.fn(begin, end);
      } catch (...) {
        std::lock_guard<std::mutex> lock(task.mutex);
        if (!task.error) task.error = std::current_exception();
      }
    };
    const int status = callbacks_.parallel_for(callbacks_.context, total,
                                               block_size, cost, invoke, &task);
    if (task.error) std::rethrow_exception(task.error);
    if (status)
      throw std::runtime_error("TensorFlow thread-pool callback failed");
  }

 private:
  AnncThreadPoolCallbacks callbacks_;
};

}  // namespace

extern "C" AnncThreadPoolAdapter* anncCreateThreadPoolAdapter(
    const AnncThreadPoolCallbacks* callbacks) {
  if (!callbacks || !callbacks->num_threads || !callbacks->in_parallel_region ||
      !callbacks->parallel_for) {
    return nullptr;
  }
  try {
    auto* pool = static_cast<annc::threadpool::AnncThreadPool*>(
        new ThreadPoolAdapter(*callbacks));
    return reinterpret_cast<AnncThreadPoolAdapter*>(pool);
  } catch (...) {
    return nullptr;
  }
}

extern "C" void anncDestroyThreadPoolAdapter(AnncThreadPoolAdapter* adapter) {
  delete reinterpret_cast<annc::threadpool::AnncThreadPool*>(adapter);
}
