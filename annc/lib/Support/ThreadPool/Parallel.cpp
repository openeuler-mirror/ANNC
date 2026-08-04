#include "Support/ThreadPool/Parallel.h"

#include <algorithm>

namespace annc {
namespace threadpool {

namespace {

class ScopedThreadPoolActivation final {
 public:
  explicit ScopedThreadPoolActivation(
      annc::threadpool::AnncThreadPool* thread_pool)
      : previous_(annc::threadpool::getCurrentThreadPool()) {
    if (previous_ != thread_pool) {
      annc::threadpool::setCurrentThreadPool(thread_pool);
      restore_ = true;
    }
  }

  ~ScopedThreadPoolActivation() {
    if (restore_) annc::threadpool::setCurrentThreadPool(previous_);
  }

  ScopedThreadPoolActivation(const ScopedThreadPoolActivation&) = delete;
  ScopedThreadPoolActivation& operator=(const ScopedThreadPoolActivation&) =
      delete;

 private:
  annc::threadpool::AnncThreadPool* previous_;
  bool restore_ = false;
};

class ScopedThreadPoolDeactivation final {
 public:
  ScopedThreadPoolDeactivation()
      : previous_(annc::threadpool::getCurrentThreadPool()) {
    annc::threadpool::setCurrentThreadPool(nullptr);
  }

  ~ScopedThreadPoolDeactivation() {
    annc::threadpool::setCurrentThreadPool(previous_);
  }

  ScopedThreadPoolDeactivation(const ScopedThreadPoolDeactivation&) = delete;
  ScopedThreadPoolDeactivation& operator=(const ScopedThreadPoolDeactivation&) =
      delete;

 private:
  annc::threadpool::AnncThreadPool* previous_;
};

void runSerial(std::int64_t total,
               const std::function<void(std::int64_t, std::int64_t)>& fn) {
  if (total <= 0) {
    return;
  }
  fn(0, total);
}

}  // namespace

int num_threads(AnncThreadPool* thread_pool) {
  if (thread_pool == nullptr) {
    return 1;
  }

  const int threads = thread_pool->num_threads();
  return threads > 0 ? threads : 1;
}

bool in_parallel_region(AnncThreadPool* thread_pool) {
  if (thread_pool == nullptr) {
    return false;
  }
  return thread_pool->in_parallel_region();
}

void parallel_for(AnncThreadPool* thread_pool, std::int64_t total,
                  const ParallelForOptions& options,
                  const std::function<void(std::int64_t, std::int64_t)>& fn) {
  if (total <= 0) {
    return;
  }

  const std::int64_t grain_size =
      std::max<std::int64_t>(options.grain_size.value_or(1), 1);
  const bool should_run_parallel =
      thread_pool != nullptr && total > grain_size &&
      num_threads(thread_pool) > 1 && !in_parallel_region(thread_pool);

  if (!should_run_parallel) {
    // A synchronous callback must not use the caller's pool for a nested
    // parallel region.
    ScopedThreadPoolDeactivation deactivation;
    runSerial(total, fn);
    return;
  }

  thread_pool->parallel_for(
      total, options, [&](std::int64_t begin, std::int64_t end) {
        // Workers may have independent TLS, so expose the active pool while
        // generated task code is running and restore the previous state.
        ScopedThreadPoolActivation activation(thread_pool);
        fn(begin, end);
      });
}

void parallel_for(AnncThreadPool* thread_pool, std::int64_t total,
                  const std::function<void(std::int64_t, std::int64_t)>& fn) {
  parallel_for(thread_pool, total, ParallelForOptions{}, fn);
}

}  // namespace threadpool
}  // namespace annc

extern "C" std::int64_t annc_threadpool_num_threads() {
  return annc::threadpool::num_threads(
      annc::threadpool::getCurrentThreadPool());
}

extern "C" std::int32_t annc_threadpool_in_parallel() {
  return annc::threadpool::in_parallel_region(
             annc::threadpool::getCurrentThreadPool())
             ? 1
             : 0;
}

extern "C" void annc_threadpool_parallel_for(
    std::int64_t total, std::int64_t cost_per_unit,
    annc::threadpool::ParallelForCallback callback, void* context) {
  if (callback == nullptr) return;

  annc::threadpool::ParallelForOptions options;
  if (cost_per_unit > 0) options.cost_per_unit = cost_per_unit;
  annc::threadpool::parallel_for(
      annc::threadpool::getCurrentThreadPool(), total, options,
      [callback, context](std::int64_t begin, std::int64_t end) {
        callback(context, begin, end);
      });
}
