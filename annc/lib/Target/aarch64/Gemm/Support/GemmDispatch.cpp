#include "Target/aarch64/Gemm/GemmDispatch.h"

#include <array>
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

struct GemmEpilogueDispatchContext {
  AnncAArch64GemmEpilogueTask task;
  AnncMemRef2DF32 lhs;
  AnncMemRef2DF32 rhs;
  AnncMemRef2DF32 out;
  std::array<AnncUnrankedMemRefF32, 8> epilogue;
};

void runGemmEpilogueTasks(const GemmEpilogueDispatchContext& context,
                          std::int64_t begin, std::int64_t end) {
  for (std::int64_t taskId = begin; taskId < end; ++taskId) {
    context.task(
        taskId, context.lhs.allocated, context.lhs.aligned,
        context.lhs.offset, context.lhs.sizes[0], context.lhs.sizes[1],
        context.lhs.strides[0], context.lhs.strides[1],
        context.rhs.allocated, context.rhs.aligned, context.rhs.offset,
        context.rhs.sizes[0], context.rhs.sizes[1], context.rhs.strides[0],
        context.rhs.strides[1], context.out.allocated, context.out.aligned,
        context.out.offset, context.out.sizes[0], context.out.sizes[1],
        context.out.strides[0], context.out.strides[1],
        context.epilogue[0].rank, context.epilogue[0].descriptor,
        context.epilogue[1].rank, context.epilogue[1].descriptor,
        context.epilogue[2].rank, context.epilogue[2].descriptor,
        context.epilogue[3].rank, context.epilogue[3].descriptor,
        context.epilogue[4].rank, context.epilogue[4].descriptor,
        context.epilogue[5].rank, context.epilogue[5].descriptor,
        context.epilogue[6].rank, context.epilogue[6].descriptor,
        context.epilogue[7].rank, context.epilogue[7].descriptor);
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

extern "C" void _mlir_ciface_annc_threadpool_parallel_for_gemm_epilogue(
    std::int64_t total, AnncAArch64GemmEpilogueTask task,
    const AnncMemRef2DF32* lhs, const AnncMemRef2DF32* rhs,
    const AnncMemRef2DF32* out, const AnncUnrankedMemRefF32* epilogue0,
    const AnncUnrankedMemRefF32* epilogue1,
    const AnncUnrankedMemRefF32* epilogue2,
    const AnncUnrankedMemRefF32* epilogue3,
    const AnncUnrankedMemRefF32* epilogue4,
    const AnncUnrankedMemRefF32* epilogue5,
    const AnncUnrankedMemRefF32* epilogue6,
    const AnncUnrankedMemRefF32* epilogue7) {
  if (task == nullptr || lhs == nullptr || rhs == nullptr || out == nullptr ||
      epilogue0 == nullptr || epilogue1 == nullptr || epilogue2 == nullptr ||
      epilogue3 == nullptr || epilogue4 == nullptr || epilogue5 == nullptr ||
      epilogue6 == nullptr || epilogue7 == nullptr)
    return;

  const GemmEpilogueDispatchContext context{
      task,
      *lhs,
      *rhs,
      *out,
      {*epilogue0, *epilogue1, *epilogue2, *epilogue3, *epilogue4,
       *epilogue5, *epilogue6, *epilogue7}};
  annc::threadpool::ParallelForOptions options;
  options.grain_size = 1;
  annc::threadpool::parallel_for(
      annc::threadpool::getCurrentThreadPool(), total, options,
      [&context](std::int64_t begin, std::int64_t end) {
        runGemmEpilogueTasks(context, begin, end);
      });
}
