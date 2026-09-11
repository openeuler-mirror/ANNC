#include "Target/aarch64/Gemm/GemmDispatch.h"

#include <cstdint>

#include "Support/ThreadPool/Parallel.h"

namespace {

struct GemmDispatchContext {
  AnncAArch64GemmTask task;
  AnncMemRef2DF32 lhs;
  AnncMemRef2DF32 rhs;
  AnncMemRef2DF32 out;
};

void runGemmTasks(const GemmDispatchContext& context, std::int64_t begin,
                  std::int64_t end) {
  for (std::int64_t taskId = begin; taskId < end; ++taskId) {
    context.task(taskId, context.lhs.allocated, context.lhs.aligned,
                 context.lhs.offset, context.lhs.sizes[0],
                 context.lhs.sizes[1], context.lhs.strides[0],
                 context.lhs.strides[1], context.rhs.allocated,
                 context.rhs.aligned, context.rhs.offset,
                 context.rhs.sizes[0], context.rhs.sizes[1],
                 context.rhs.strides[0], context.rhs.strides[1],
                 context.out.allocated, context.out.aligned,
                 context.out.offset, context.out.sizes[0],
                 context.out.sizes[1], context.out.strides[0],
                 context.out.strides[1]);
  }
}

}  // namespace

extern "C" void _mlir_ciface_annc_threadpool_parallel_for_gemm(
    std::int64_t total, AnncAArch64GemmTask task, const AnncMemRef2DF32* lhs,
    const AnncMemRef2DF32* rhs, const AnncMemRef2DF32* out) {
  if (task == nullptr || lhs == nullptr || rhs == nullptr || out == nullptr)
    return;

  const GemmDispatchContext context{task, *lhs, *rhs, *out};
  annc::threadpool::ParallelForOptions options;
  // One GEMM task per queue item: fixed-block dispatch lets fast workers
  // absorb the fatter tail tasks instead of pre-packaging them.
  options.grain_size = 1;
  annc::threadpool::parallel_for(
      annc::threadpool::getCurrentThreadPool(), total, options,
      [&context](std::int64_t begin, std::int64_t end) {
        runGemmTasks(context, begin, end);
      });
}
