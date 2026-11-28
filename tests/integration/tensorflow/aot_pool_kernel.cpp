#include <atomic>

#include "Context/ThreadPool.h"
#include "Kernel/MemRefTypes.h"

namespace {
using Pool = annc::threadpool::AnncThreadPool;
thread_local Pool* currentPool = nullptr;
thread_local bool ranWithPool = false;
std::atomic<int> restoredCalls{0};
}  // namespace

extern "C" Pool* annc_get_current_threadpool() { return currentPool; }

extern "C" void annc_set_current_threadpool(Pool* pool) {
  if (ranWithPool && !pool) {
    ++restoredCalls;
    ranWithPool = false;
  }
  currentPool = pool;
}

extern "C" int restored_pool_calls() { return restoredCalls.load(); }

extern "C" void _mlir_ciface_add(const AnncMemRef2DF32* lhs,
                                 const AnncMemRef2DF32* rhs,
                                 AnncMemRef2DF32* out) {
  const int64_t count = out->sizes[0] * out->sizes[1];
  auto add = [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i)
      out->aligned[out->offset + i] =
          lhs->aligned[lhs->offset + i] + rhs->aligned[rhs->offset + i];
  };
  if (currentPool) {
    annc::threadpool::ParallelForOptions options;
    options.max_parallelism = 2;
    options.grain_size = 1;
    currentPool->parallel_for(count, options, add);
    ranWithPool = true;
  } else {
    add(0, count);
  }
}
