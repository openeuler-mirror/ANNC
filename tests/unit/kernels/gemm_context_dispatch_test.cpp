#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <functional>
#include <stdexcept>

#include "Context/ThreadPool.h"
#include "Target/aarch64/Gemm/GemmDispatch.h"

namespace {

class RecordingPool : public annc::threadpool::AnncThreadPool {
 public:
  int num_threads() const override { return 4; }
  bool in_parallel_region() const override { return nested; }
  void parallel_for(
      std::int64_t total, const annc::threadpool::ParallelForOptions& options,
      const std::function<void(std::int64_t, std::int64_t)>& fn) override {
    ++calls;
    if (throwOnParallel) throw std::runtime_error("thread pool failed");
    EXPECT_EQ(options.grain_size, 1);
    EXPECT_FALSE(options.cost_per_unit.has_value());
    EXPECT_FALSE(options.max_parallelism.has_value());
    for (std::int64_t i = total; i > 0; --i) fn(i - 1, i);
  }
  int calls = 0;
  bool nested = false;
  bool throwOnParallel = false;
};

void gemmTask(int64_t task_id, float* lhs_allocated, float* lhs_aligned,
              int64_t lhs_offset, int64_t lhs_size0, int64_t lhs_size1,
              int64_t lhs_stride0, int64_t lhs_stride1, float* rhs_allocated,
              float* rhs_aligned, int64_t rhs_offset, int64_t rhs_size0,
              int64_t rhs_size1, int64_t rhs_stride0, int64_t rhs_stride1,
              float* out_allocated, float* out_aligned, int64_t out_offset,
              int64_t out_size0, int64_t out_size1, int64_t out_stride0,
              int64_t out_stride1) {
  EXPECT_EQ(lhs_allocated, lhs_aligned);
  EXPECT_EQ(rhs_allocated, rhs_aligned);
  EXPECT_EQ(out_allocated, out_aligned);
  EXPECT_EQ(
      (std::array<int64_t, 4>{lhs_size0, lhs_size1, lhs_stride0, lhs_stride1}),
      (std::array<int64_t, 4>{4, 3, 3, 1}));
  EXPECT_EQ(
      (std::array<int64_t, 4>{rhs_size0, rhs_size1, rhs_stride0, rhs_stride1}),
      (std::array<int64_t, 4>{3, 2, 2, 1}));
  EXPECT_EQ(
      (std::array<int64_t, 4>{out_size0, out_size1, out_stride0, out_stride1}),
      (std::array<int64_t, 4>{4, 2, 2, 1}));
  for (int64_t n = 0; n < out_size1; ++n) {
    float value = 0;
    for (int64_t k = 0; k < lhs_size1; ++k)
      value +=
          lhs_aligned[lhs_offset + task_id * lhs_stride0 + k * lhs_stride1] *
          rhs_aligned[rhs_offset + k * rhs_stride0 + n * rhs_stride1];
    out_aligned[out_offset + task_id * out_stride0 + n * out_stride1] = value;
  }
}

void epilogueTask(int64_t task_id, float* lhs_allocated, float* lhs_aligned,
                  int64_t lhs_offset, int64_t lhs_size0, int64_t lhs_size1,
                  int64_t lhs_stride0, int64_t lhs_stride1,
                  float* rhs_allocated, float* rhs_aligned, int64_t rhs_offset,
                  int64_t rhs_size0, int64_t rhs_size1, int64_t rhs_stride0,
                  int64_t rhs_stride1, float* out_allocated, float* out_aligned,
                  int64_t out_offset, int64_t out_size0, int64_t out_size1,
                  int64_t out_stride0, int64_t out_stride1,
                  int64_t epilogue0_rank, void* epilogue0_descriptor,
                  int64_t epilogue1_rank, void* epilogue1_descriptor,
                  int64_t epilogue2_rank, void* epilogue2_descriptor,
                  int64_t epilogue3_rank, void* epilogue3_descriptor,
                  int64_t epilogue4_rank, void* epilogue4_descriptor,
                  int64_t epilogue5_rank, void* epilogue5_descriptor,
                  int64_t epilogue6_rank, void* epilogue6_descriptor,
                  int64_t epilogue7_rank, void* epilogue7_descriptor) {
  gemmTask(task_id, lhs_allocated, lhs_aligned, lhs_offset, lhs_size0,
           lhs_size1, lhs_stride0, lhs_stride1, rhs_allocated, rhs_aligned,
           rhs_offset, rhs_size0, rhs_size1, rhs_stride0, rhs_stride1,
           out_allocated, out_aligned, out_offset, out_size0, out_size1,
           out_stride0, out_stride1);
  std::array<int64_t, 8> ranks = {
      epilogue0_rank, epilogue1_rank, epilogue2_rank, epilogue3_rank,
      epilogue4_rank, epilogue5_rank, epilogue6_rank, epilogue7_rank};
  std::array<void*, 8> descriptors = {
      epilogue0_descriptor, epilogue1_descriptor, epilogue2_descriptor,
      epilogue3_descriptor, epilogue4_descriptor, epilogue5_descriptor,
      epilogue6_descriptor, epilogue7_descriptor};
  float bias = 0;
  for (unsigned i = 0; i < descriptors.size(); ++i) {
    EXPECT_EQ(ranks[i], 1);
    const auto* descriptor = static_cast<AnncMemRef1DF32*>(descriptors[i]);
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(descriptor->sizes[0], 1);
    bias += (i + 1) * descriptor->aligned[descriptor->offset];
  }
  for (int64_t n = 0; n < out_size1; ++n)
    out_aligned[out_offset + task_id * out_stride0 + n * out_stride1] += bias;
}

class GemmContextDispatch : public ::testing::Test {
 protected:
  void SetUp() override {
    previous = annc::threadpool::getCurrentThreadPool();
    annc::threadpool::setCurrentThreadPool(&tlsPool);
    services.struct_size = sizeof(services);
    services.abi_version = ANNC_CONTEXT_ABI_VERSION;
    services.allocate_output = [](void*, uint32_t, const AnncTensorDesc*,
                                  AnncTensorView*) { return ANNC_STATUS_OK; };
    services.allocate_temp = [](void*, const AnncTensorDesc*, AnncTensorView*) {
      return ANNC_STATUS_OK;
    };
    services.report_error = [](void*, AnncStatusCode, const char*) {};
    services.get_status = [](void*) { return ANNC_STATUS_OK; };
    services.thread_pool = &contextPool;
    context.struct_size = sizeof(context);
    context.abi_version = ANNC_CONTEXT_ABI_VERSION;
    context.services = &services;
  }
  void TearDown() override { annc::threadpool::setCurrentThreadPool(previous); }

  void run() {
    _mlir_ciface_annc_context_parallel_for_gemm(&context, 4, gemmTask, &lhs,
                                                &rhs, &out);
  }
  void expectProduct() {
    EXPECT_EQ(output,
              (std::array<float, 8>{22, 28, 49, 64, 76, 100, 103, 136}));
  }

  RecordingPool contextPool;
  RecordingPool tlsPool;
  annc::threadpool::AnncThreadPool* previous = nullptr;
  AnncRuntimeServices services{};
  AnncKernelContext context{};
  std::array<float, 12> lhsData{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::array<float, 6> rhsData{1, 2, 3, 4, 5, 6};
  std::array<float, 8> output{};
  AnncMemRef2DF32 lhs{lhsData.data(), lhsData.data(), 0, {4, 3}, {3, 1}};
  AnncMemRef2DF32 rhs{rhsData.data(), rhsData.data(), 0, {3, 2}, {2, 1}};
  AnncMemRef2DF32 out{output.data(), output.data(), 0, {4, 2}, {2, 1}};
};

TEST_F(GemmContextDispatch, UsesContextPoolInsteadOfTls) {
  run();
  expectProduct();
  EXPECT_EQ(contextPool.calls, 1);
  EXPECT_EQ(tlsPool.calls, 0);
  EXPECT_EQ(annc::threadpool::getCurrentThreadPool(), &tlsPool);
}

TEST_F(GemmContextDispatch, NullContextPoolRunsSeriallyWithoutTlsFallback) {
  services.thread_pool = nullptr;
  run();
  expectProduct();
  EXPECT_EQ(contextPool.calls, 0);
  EXPECT_EQ(tlsPool.calls, 0);
  EXPECT_EQ(annc::threadpool::getCurrentThreadPool(), &tlsPool);
}

TEST_F(GemmContextDispatch, PropagatesThreadPoolExceptionsToHostBoundary) {
  contextPool.throwOnParallel = true;
  EXPECT_THROW(run(), std::runtime_error);
}

TEST_F(GemmContextDispatch, NestedContextPoolRunsSerially) {
  contextPool.nested = true;
  run();
  expectProduct();
  EXPECT_EQ(contextPool.calls, 0);
  EXPECT_EQ(tlsPool.calls, 0);
}

TEST_F(GemmContextDispatch, EpiloguePreservesAllDescriptorsAndUsesContextPool) {
  std::array<float, 8> bias{0, 1, 2, 3, 4, 5, 6, 7};
  std::array<AnncMemRef1DF32, 8> descriptors;
  std::array<AnncUnrankedMemRefF32, 8> args;
  for (unsigned i = 0; i < args.size(); ++i) {
    descriptors[i] = {&bias[i], &bias[i], 0, {1}, {1}};
    args[i] = {1, &descriptors[i]};
  }
  _mlir_ciface_annc_context_parallel_for_gemm_epilogue(
      &context, 4, epilogueTask, &lhs, &rhs, &out, &args[0], &args[1], &args[2],
      &args[3], &args[4], &args[5], &args[6], &args[7]);
  EXPECT_EQ(output,
            (std::array<float, 8>{190, 196, 217, 232, 244, 268, 271, 304}));
  EXPECT_EQ(contextPool.calls, 1);
  EXPECT_EQ(tlsPool.calls, 0);
}

TEST_F(GemmContextDispatch, LegacyDispatchStillUsesTls) {
  _mlir_ciface_annc_threadpool_parallel_for_gemm(4, gemmTask, &lhs, &rhs, &out);
  expectProduct();
  EXPECT_EQ(contextPool.calls, 0);
  EXPECT_EQ(tlsPool.calls, 1);
}

}  // namespace
