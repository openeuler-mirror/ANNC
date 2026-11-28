#include <stdexcept>

#include "Context/ThreadPool.h"

// The kernel side owns C++17 options, the task and its exceptions. Only plain
// values cross back to the caller built with the target SDK's C++ ABI.
extern "C" int RunPoolKernel(int total, int cost, int grain, int* visits,
                             bool throw_task, int max_parallelism) {
  struct TaskError {};
  annc::threadpool::ParallelForOptions options;
  if (cost >= 0) options.cost_per_unit = cost;
  if (grain >= 0) options.grain_size = grain;
  if (max_parallelism >= 0) options.max_parallelism = max_parallelism;
  try {
    annc_get_current_threadpool()->parallel_for(
        total, options, [&](int64_t begin, int64_t end) {
          if (throw_task) throw TaskError{};
          for (int64_t i = begin; i < end; ++i) ++visits[i];
        });
    return 0;
  } catch (const TaskError&) {
    return 1;
  } catch (const std::runtime_error&) {
    return 2;
  }
}
