#ifndef ANNC_KERNEL_BUILTIN_KDNN_ADAPTOR_THREADPOOL_H
#define ANNC_KERNEL_BUILTIN_KDNN_ADAPTOR_THREADPOOL_H

#include <cstdint>
#include <functional>

#include "Kernel/threadpool/ThreadPool.h"
#include "kdnn_config.h"
#include "service/kdnn_threading.hpp"

namespace annc::kernels::kdnn_adaptor {

class KDNNThreadPoolAdaptor final : public KDNN::Threading::ThreadpoolIface {
public:
    explicit KDNNThreadPoolAdaptor(annc::threadpool::AnncThreadPool* thread_pool)
        : thread_pool_(thread_pool) {}

    int GetNumThreads() const override {
        if (thread_pool_ == nullptr) {
            return 1;
        }

        const int threads = thread_pool_->num_threads();
        return threads > 0 ? threads : 1;
    }

    bool IsInParallel() const override {
        return thread_pool_ != nullptr && thread_pool_->in_parallel_region();
    }

    void ParallelFor(int total,
                     int64_t cost_per_unit,
                     const std::function<void(int, int)>& fn) override {
        if (thread_pool_ == nullptr || total <= 0) {
            if (total > 0) {
                fn(0, total);
            }
            return;
        }

        annc::threadpool::ParallelForOptions options;
        if (cost_per_unit > 0) {
            options.cost_per_unit = cost_per_unit;
        }

        thread_pool_->parallel_for(
            total,
            options,
            [&](std::int64_t begin, std::int64_t end) {
                fn(static_cast<int>(begin), static_cast<int>(end));
            });
    }

private:
    annc::threadpool::AnncThreadPool* thread_pool_;
};

class ScopedKDNNThreadPoolActivation final {
public:
    explicit ScopedKDNNThreadPoolActivation(annc::threadpool::AnncThreadPool* thread_pool)
        : adaptor_(thread_pool) {
#if defined(KDNN_CPU_THREADING_RUNTIME) && (KDNN_CPU_THREADING_RUNTIME == KDNN_RUNTIME_THREADPOOL)
        if (thread_pool != nullptr) {
            KDNN::Threading::ActivateThreadpool(&adaptor_);
            active_ = true;
        }
#else
        (void)thread_pool;
#endif
    }

    ~ScopedKDNNThreadPoolActivation() {
#if defined(KDNN_CPU_THREADING_RUNTIME) && (KDNN_CPU_THREADING_RUNTIME == KDNN_RUNTIME_THREADPOOL)
        if (active_) {
            KDNN::Threading::DeactivateThreadpool();
        }
#endif
    }

    ScopedKDNNThreadPoolActivation(const ScopedKDNNThreadPoolActivation&) = delete;
    ScopedKDNNThreadPoolActivation& operator=(const ScopedKDNNThreadPoolActivation&) = delete;

private:
    KDNNThreadPoolAdaptor adaptor_;
    bool active_ = false;
};

} // namespace annc::kernels::kdnn_adaptor

#endif // ANNC_KERNEL_BUILTIN_KDNN_ADAPTOR_THREADPOOL_H
