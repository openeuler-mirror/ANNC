#include "ThreadPool.h"

#include <algorithm>
#include <optional>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow::annc_context {

TensorFlowAnncThreadPool::TensorFlowAnncThreadPool(thread::ThreadPool* workers)
    : workers_(workers) {}

int TensorFlowAnncThreadPool::num_threads() const {
  return workers_ ? workers_->NumThreads() : 1;
}

bool TensorFlowAnncThreadPool::in_parallel_region() const {
  return workers_ && workers_->CurrentThreadId() >= 0;
}

void TensorFlowAnncThreadPool::parallel_for(
    int64_t total, const annc::threadpool::ParallelForOptions& options,
    const std::function<void(int64_t, int64_t)>& fn) {
  if (!workers_ || total <= 0) {
    if (total > 0) fn(0, total);
    return;
  }

  if (options.max_parallelism.has_value()) {
    const int max_parallelism =
        std::max(1, std::min(*options.max_parallelism, num_threads()));
    if (max_parallelism == 1 || in_parallel_region()) {
      fn(0, total);
    } else {
      Shard(max_parallelism, workers_, total,
            std::max<int64_t>(options.cost_per_unit.value_or(1), 1), fn);
    }
    return;
  }

  if (options.grain_size.has_value()) {
    workers_->ParallelFor(
        total,
        thread::ThreadPool::SchedulingParams(
            thread::ThreadPool::SchedulingStrategy::kFixedBlockSize,
            std::nullopt, std::max<int64_t>(*options.grain_size, 1)),
        fn);
    return;
  }

  const int64_t cost_per_unit =
      std::max<int64_t>(options.cost_per_unit.value_or(1), 1);
  workers_->ParallelFor(total,
                        thread::ThreadPool::SchedulingParams(
                            thread::ThreadPool::SchedulingStrategy::kAdaptive,
                            cost_per_unit, std::nullopt),
                        fn);
}

thread::ThreadPool* GetTensorFlowCpuThreadPool(OpKernelContext* context) {
  if (!context || !context->device()) {
    return nullptr;
  }

  const DeviceBase::CpuWorkerThreads* worker_threads = nullptr;
  // DeviceBase::tensorflow_cpu_worker_threads() CHECKs when no CPU worker
  // threads were installed, so only call it for CPU devices where ANNCFused is
  // registered today.
  worker_threads = context->device()->tensorflow_cpu_worker_threads();
  return worker_threads ? worker_threads->workers : nullptr;
}

}  // namespace tensorflow::annc_context
