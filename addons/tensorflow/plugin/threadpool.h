#ifndef ANNC_TENSORFLOW_PLUGIN_THREADPOOL_H
#define ANNC_TENSORFLOW_PLUGIN_THREADPOOL_H

namespace annc {
namespace threadpool {
class AnncThreadPool;
}  // namespace threadpool
}  // namespace annc

namespace tensorflow {

// The binding scope needs only opaque pool pointers. It does not
// inspect the C++17 pool interface or own the adapter passed to it.
class ScopedAnncThreadPool final {
 public:
  ScopedAnncThreadPool(
      annc::threadpool::AnncThreadPool* thread_pool,
      void (*set_current_threadpool)(annc::threadpool::AnncThreadPool*),
      annc::threadpool::AnncThreadPool* (*get_current_threadpool)())
      : set_current_threadpool_(set_current_threadpool),
        previous_(get_current_threadpool ? get_current_threadpool() : nullptr),
        active_(set_current_threadpool != nullptr) {
    if (set_current_threadpool_) {
      set_current_threadpool_(thread_pool);
    }
  }

  ~ScopedAnncThreadPool() { Restore(); }

  void Restore() {
    if (active_) {
      set_current_threadpool_(previous_);
      active_ = false;
    }
  }

  ScopedAnncThreadPool(const ScopedAnncThreadPool&) = delete;
  ScopedAnncThreadPool& operator=(const ScopedAnncThreadPool&) = delete;

 private:
  void (*set_current_threadpool_)(annc::threadpool::AnncThreadPool*);
  annc::threadpool::AnncThreadPool* previous_;
  bool active_;
};

}  // namespace tensorflow

#endif  // ANNC_TENSORFLOW_PLUGIN_THREADPOOL_H
