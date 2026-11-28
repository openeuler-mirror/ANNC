#ifndef ANNC_TENSORFLOW_CONTEXT_THREAD_POOL_H
#define ANNC_TENSORFLOW_CONTEXT_THREAD_POOL_H

#include "Context/ThreadPool.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {
class OpKernelContext;

namespace annc_context {

// Borrows TensorFlow's intra-op pool; it never owns or creates worker threads.
class TensorFlowAnncThreadPool final : public annc::threadpool::AnncThreadPool {
 public:
  explicit TensorFlowAnncThreadPool(thread::ThreadPool* workers);

  int num_threads() const override;
  bool in_parallel_region() const override;
  void parallel_for(int64_t total,
                    const annc::threadpool::ParallelForOptions& options,
                    const std::function<void(int64_t, int64_t)>& fn) override;

 private:
  thread::ThreadPool* workers_;
};

thread::ThreadPool* GetTensorFlowCpuThreadPool(OpKernelContext* context);

}  // namespace annc_context
}  // namespace tensorflow

#endif
