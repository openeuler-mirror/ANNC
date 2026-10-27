#ifndef ANNC_KERNEL_BUILTIN_KDNN_ADAPTOR_THREADPOOL_H
#define ANNC_KERNEL_BUILTIN_KDNN_ADAPTOR_THREADPOOL_H

#include <cstdint>
#include <functional>

#include "Context/KernelContextUtils.h"
#include "kdnn_config.h"
#include "service/kdnn_threading.hpp"

namespace annc::kernels::kdnn_adaptor {

class KDNNThreadPoolAdaptor final : public KDNN::Threading::ThreadpoolIface {
 public:
  explicit KDNNThreadPoolAdaptor(KernelContext& context)
      : workers_(context.getThreadPool()) {}

  int GetNumThreads() const override { return workers_.numThreads(); }

  bool IsInParallel() const override {
    return false;  // The synchronous context service handles nested execution.
  }

  void ParallelFor(int total, int64_t cost_per_unit,
                   const std::function<void(int, int)>& fn) override {
    Shard(workers_.numThreads(), workers_, total, cost_per_unit,
          [&](int64_t begin, int64_t end) {
            fn(static_cast<int>(begin), static_cast<int>(end));
          });
  }

 private:
  KernelThreadPool workers_;
};

class ScopedKDNNThreadPoolActivation final {
 public:
  explicit ScopedKDNNThreadPoolActivation(KernelContext& context)
      : adaptor_(context) {
#if defined(KDNN_CPU_THREADING_RUNTIME) && \
    (KDNN_CPU_THREADING_RUNTIME == KDNN_RUNTIME_THREADPOOL)
    KDNN::Threading::ActivateThreadpool(&adaptor_);
#endif
  }

  ~ScopedKDNNThreadPoolActivation() {
#if defined(KDNN_CPU_THREADING_RUNTIME) && \
    (KDNN_CPU_THREADING_RUNTIME == KDNN_RUNTIME_THREADPOOL)
    KDNN::Threading::DeactivateThreadpool();
#endif
  }

  ScopedKDNNThreadPoolActivation(const ScopedKDNNThreadPoolActivation&) =
      delete;
  ScopedKDNNThreadPoolActivation& operator=(
      const ScopedKDNNThreadPoolActivation&) = delete;

 private:
  KDNNThreadPoolAdaptor adaptor_;
};

}  // namespace annc::kernels::kdnn_adaptor

#endif  // ANNC_KERNEL_BUILTIN_KDNN_ADAPTOR_THREADPOOL_H
