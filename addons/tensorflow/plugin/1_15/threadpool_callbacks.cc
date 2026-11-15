#include "threadpool_callbacks.h"

#include <algorithm>

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/util/work_sharder.h"

namespace annc {
namespace tensorflow {
namespace {

using TfPool = ::tensorflow::thread::ThreadPool;

int NumThreads(void* context) {
  return static_cast<TfPool*>(context)->NumThreads();
}

int InParallelRegion(void* context) {
  return static_cast<TfPool*>(context)->CurrentThreadId() >= 0;
}

int ParallelFor(void* context, std::int64_t total, std::int64_t block_size,
                std::int64_t cost_per_unit, int max_parallelism,
                AnncThreadPoolTask task, void* task_context) {
  auto& pool = *static_cast<TfPool*>(context);
  try {
    const auto fn = [task, task_context](std::int64_t begin, std::int64_t end) {
      task(task_context, begin, end);
    };
    if (max_parallelism > 0) {
      const int limit = std::max(1, std::min(max_parallelism, pool.NumThreads()));
      if (limit == 1 || InParallelRegion(context)) {
        fn(0, total);
      } else {
        ::tensorflow::Shard(limit, &pool, total, cost_per_unit, fn);
      }
    } else if (block_size > 0) {
      pool.TransformRangeConcurrently(std::min(block_size, total), total, fn);
    } else {
      pool.ParallelFor(total, cost_per_unit, fn);
    }
    return 0;
  } catch (...) {
    return 1;
  }
}

}  // namespace

AnncThreadPoolCallbacks MakeThreadPoolCallbacks(
    ::tensorflow::thread::ThreadPool& pool) {
  return {&pool, NumThreads, InParallelRegion, ParallelFor};
}

}  // namespace tensorflow
}  // namespace annc
