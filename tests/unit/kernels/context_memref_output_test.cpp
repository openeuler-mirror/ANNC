#include <array>
#include <cstdint>
#include <deque>

#include <gtest/gtest.h>

#include "Context/KernelContext.h"

namespace {

struct OutputHost {
  std::array<float, 6> data{};
  std::deque<std::array<int64_t, 7>> scratch;
  AnncStatusCode status = ANNC_STATUS_OK;
  AnncAllocationFlags outputFlags = ANNC_ALLOCATION_NONE;
  int outputCalls = 0;
  int tempCalls = 0;
  int64_t sizes[2] = {2, 3};
  int64_t strides[2] = {3, 1};
  int64_t scratchSize = 7;
  int64_t scratchStride = 1;

  static AnncStatusCode AllocateOutput(void* opaque, uint32_t,
                                       const AnncTensorDesc* request,
                                       AnncTensorView* output) {
    auto& host = *static_cast<OutputHost*>(opaque);
    ++host.outputCalls;
    host.outputFlags = request->flags;
    host.data.fill(7.0f);
    if (request->flags & ANNC_ALLOCATION_ZERO) host.data.fill(0.0f);
    *output = {host.data.data(), host.data.data(), 0, host.sizes, host.strides,
               ANNC_ELEMENT_TYPE_F32, 2, ANNC_LAYOUT_STRIDED, ANNC_TENSOR_BOUND};
    return ANNC_STATUS_OK;
  }

  static AnncStatusCode AllocateTemp(void* opaque, const AnncTensorDesc*,
                                     AnncTensorView* output) {
    auto& host = *static_cast<OutputHost*>(opaque);
    ++host.tempCalls;
    host.scratch.emplace_back();
    auto& storage = host.scratch.back();
    *output = {storage.data(), storage.data(), 0, &host.scratchSize,
               &host.scratchStride,
               ANNC_ELEMENT_TYPE_I64, 1, ANNC_LAYOUT_STRIDED,
               ANNC_TENSOR_BOUND};
    return ANNC_STATUS_OK;
  }

  static void ReportError(void* opaque, AnncStatusCode code, const char*) {
    static_cast<OutputHost*>(opaque)->status = code;
  }

  static AnncStatusCode GetStatus(void* opaque) {
    return static_cast<OutputHost*>(opaque)->status;
  }
};

TEST(ContextMemRefOutputTest, ReusesAllocatedOutputWithoutClearing) {
  OutputHost host;
  AnncRuntimeServices services{sizeof(AnncRuntimeServices),
                               ANNC_CONTEXT_ABI_VERSION,
                               &host,
                               OutputHost::AllocateOutput,
                               OutputHost::AllocateTemp,
                               OutputHost::ReportError,
                               OutputHost::GetStatus,
                               nullptr};
  AnncTensorView output{};
  output.element_type = ANNC_ELEMENT_TYPE_F32;
  AnncKernelContext context{sizeof(AnncKernelContext), ANNC_CONTEXT_ABI_VERSION,
                            nullptr, 0, &output, 1, nullptr, 0, nullptr, 0,
                            &services};
  const int64_t dims[] = {2, 3};
  ASSERT_NE(annc_context_alloc_output_memref(&context, 0, 2, dims), nullptr);
  EXPECT_EQ(host.outputFlags, ANNC_ALLOCATION_NONE);
  EXPECT_EQ(host.outputCalls, 1);
  EXPECT_EQ(host.data, (std::array<float, 6>{7, 7, 7, 7, 7, 7}));

  host.data.fill(9.0f);
  ASSERT_NE(annc_context_alloc_output_memref(&context, 0, 2, dims), nullptr);
  EXPECT_EQ(host.outputCalls, 1);
  EXPECT_EQ(host.data, (std::array<float, 6>{9, 9, 9, 9, 9, 9}));
  EXPECT_EQ(host.status, ANNC_STATUS_OK);
}

TEST(ContextMemRefInputTest, RejectsStridedInputForDenseMemRefAccessor) {
  OutputHost host;
  AnncRuntimeServices services{sizeof(AnncRuntimeServices),
                               ANNC_CONTEXT_ABI_VERSION,
                               &host,
                               OutputHost::AllocateOutput,
                               OutputHost::AllocateTemp,
                               OutputHost::ReportError,
                               OutputHost::GetStatus,
                               nullptr};
  int64_t sizes[] = {2, 3};
  int64_t strides[] = {4, 1};
  AnncTensorView input{host.data.data(), host.data.data(), 0, sizes, strides,
                       ANNC_ELEMENT_TYPE_F32, 2, ANNC_LAYOUT_STRIDED,
                       ANNC_TENSOR_BOUND};
  AnncKernelContext context{sizeof(AnncKernelContext), ANNC_CONTEXT_ABI_VERSION,
                            &input, 1, nullptr, 0, nullptr, 0, nullptr, 0,
                            &services};

  EXPECT_EQ(annc_context_get_input_memref(&context, 0), nullptr);
  EXPECT_EQ(host.status, ANNC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(host.tempCalls, 0);
}

}  // namespace
