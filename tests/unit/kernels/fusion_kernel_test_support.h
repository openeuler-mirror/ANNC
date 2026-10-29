#ifndef ANNC_TESTS_UNIT_KERNELS_FUSION_KERNEL_TEST_SUPPORT_H_
#define ANNC_TESTS_UNIT_KERNELS_FUSION_KERNEL_TEST_SUPPORT_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "Context/KernelContextUtils.h"
#include "Kernel/ExecutionContext.h"
#include "Kernel/ExecutionContextUtils.h"
#include "Kernel/MemRefTypes.h"

namespace annc::test {

// Context harness for fused kernels. Owns allocated output buffers and
// their shape/stride metadata for the duration of the synchronous call.
class OutputBuffers {
 public:
  static constexpr uint32_t kSlots = 4;

  static AnncStatusCode Allocate(void* handle, uint32_t slot,
                                 const AnncTensorDesc* desc, AnncTensorView* result) {
    auto* self = reinterpret_cast<OutputBuffers*>(handle);
    if (self == nullptr || desc == nullptr || slot >= kSlots ||
        desc->element_type != ANNC_ELEMENT_TYPE_F32) {
      return ANNC_STATUS_INVALID_ARGUMENT;
    }
    ++self->allocate_calls;
    self->last_slot = slot;
    self->last_rank = desc->rank;
    self->last_element_type = desc->element_type;
    if (desc->rank == 0) {
      self->last_dims.clear();
    } else {
      self->last_dims.assign(desc->dims, desc->dims + desc->rank);
    }
    size_t count = 1;
    for (uint32_t i = 0; i < desc->rank; ++i) {
      if (desc->dims[i] < 0) return ANNC_STATUS_INVALID_ARGUMENT;
      if (desc->dims[i] != 0 &&
          count > std::numeric_limits<size_t>::max() /
                      static_cast<size_t>(desc->dims[i])) {
        return ANNC_STATUS_INVALID_ARGUMENT;
      }
      count *= static_cast<size_t>(desc->dims[i]);
    }
    if (self->fail_next_allocation) {
      self->fail_next_allocation = false;
      return ANNC_STATUS_ALLOCATION_FAILED;
    }
    self->buffers[slot].assign(count, 0.0f);
    self->sizes[slot] = self->last_dims;
    self->strides[slot].resize(desc->rank);
    int64_t stride = 1;
    for (uint32_t i = desc->rank; i > 0; --i) {
      self->strides[slot][i - 1] = stride;
      stride *= desc->dims[i - 1];
    }
    void* data = count == 0 ? nullptr : self->buffers[slot].data();
    *result = {data, data, 0, self->sizes[slot].data(),
               self->strides[slot].data(), desc->element_type, desc->rank,
               ANNC_LAYOUT_STRIDED, ANNC_TENSOR_BOUND};
    return ANNC_STATUS_OK;
  }

  static AnncStatusCode ParallelFor(void*, int64_t total, int64_t,
                                    AnncContextTask task, void* task_data) {
    if (task == nullptr || total < 0) return ANNC_STATUS_INVALID_ARGUMENT;
    if (total != 0) task(task_data, 0, total);
    return ANNC_STATUS_OK;
  }

  static int32_t NumThreads(void*) { return 1; }

  AnncExecutionContext context() {
    AnncExecutionContext c{};
    c.struct_size = sizeof(c);
    c.abi_version = ANNC_EXECUTION_ABI_VERSION;
    c.handle = reinterpret_cast<AnncExecutionHandle*>(this);
    c.allocate_output = [](AnncExecutionHandle* handle, uint32_t slot, AnncTensorDesc* desc) {
      AnncTensorView view{};
      auto code = Allocate(handle, slot, desc, &view);
      if (code == ANNC_STATUS_OK) desc->data = view.aligned;
      return code;
    };
    c.allocate_temp = [](AnncExecutionHandle*, AnncTensorDesc*) { return ANNC_STATUS_INVALID_ARGUMENT; };
    c.report_error = [](AnncExecutionHandle* handle, AnncStatusCode code, const char* msg) {
      auto* self = reinterpret_cast<OutputBuffers*>(handle);
      self->last_error_status = code;
      self->last_error = msg ? msg : "";
    };
    return c;
  }

  AnncKernelContext kernelContext() {
    services = {};
    services.struct_size = sizeof(services);
    services.abi_version = ANNC_CONTEXT_ABI_VERSION;
    services.opaque = this;
    services.allocate_output = Allocate;
    services.allocate_temp = [](void*, const AnncTensorDesc*, AnncTensorView*) {
      return ANNC_STATUS_INVALID_ARGUMENT;
    };
    services.report_error = [](void* handle, AnncStatusCode status,
                               const char* message) {
      auto* self = static_cast<OutputBuffers*>(handle);
      self->last_error_status = status;
      self->last_error = message == nullptr ? "" : message;
    };
    services.get_status = [](void* handle) {
      return static_cast<OutputBuffers*>(handle)->last_error_status;
    };
    services.parallel_for = ParallelFor;
    services.num_threads = NumThreads;
    AnncKernelContext c{};
    c.struct_size = sizeof(c);
    c.abi_version = ANNC_CONTEXT_ABI_VERSION;
    c.outputs = views;
    c.num_outputs = kSlots;
    c.services = &services;
    return c;
  }

  const std::vector<float>& output(uint32_t slot = 0) const {
    return buffers[slot];
  }

  // Make the next allocate_output call fail, to exercise the kernels'
  // allocation-failure path.
  void FailNextAllocation() { fail_next_allocation = true; }

  AnncRuntimeServices services{};
  AnncTensorView views[kSlots]{};
  std::vector<int64_t> sizes[kSlots], strides[kSlots];
  std::vector<float> buffers[kSlots];
  uint32_t allocate_calls = 0;
  uint32_t last_slot = 0;
  uint32_t last_rank = 0;
  AnncElementType last_element_type = ANNC_ELEMENT_TYPE_F32;
  std::vector<int64_t> last_dims;
  bool fail_next_allocation = false;
  AnncStatusCode last_error_status = ANNC_STATUS_OK;
  std::string last_error;
};

template <typename MemRef>
AnncTensorView View(const MemRef* ref) {
  if (!ref) return {};
  constexpr uint32_t rank = sizeof(ref->sizes) / sizeof(int64_t);
  return {ref->allocated, ref->aligned, ref->offset, ref->sizes, ref->strides,
          ANNC_ELEMENT_TYPE_F32, rank, ANNC_LAYOUT_STRIDED, ANNC_TENSOR_BOUND};
}
inline AnncTensorView View(const AnncMemRef0DF32* ref) {
  if (!ref) return {};
  return {ref->allocated, ref->aligned, ref->offset, nullptr, nullptr,
          ANNC_ELEMENT_TYPE_F32, 0, ANNC_LAYOUT_STRIDED, ANNC_TENSOR_BOUND};
}
inline AnncTensorView View(std::nullptr_t) { return {}; }

inline AnncStatusCode Invoke(
    AnncStatusCode (*fn)(annc::kernels::KernelContext&), AnncKernelContext* raw,
    std::initializer_list<AnncTensorView> inputs) {
  if (raw) {
    raw->inputs = inputs.begin();
    raw->num_inputs = inputs.size();
    auto* owner = static_cast<OutputBuffers*>(raw->services->opaque);
    owner->last_error_status = ANNC_STATUS_OK;
    std::fill(std::begin(owner->views), std::end(owner->views), AnncTensorView{});
  }
  return annc::kernels::invokeContextKernel(raw, fn);
}

// Tolerances absorb the difference between the kernel's summation order and
// the straightforward reference loop; a wrong result is far outside them.
inline void ExpectNear(float actual, float expected, float tolerance,
                       const char* what) {
  if (std::isnan(expected)) {
    EXPECT_TRUE(std::isnan(actual)) << what;
    return;
  }
  if (std::isinf(expected)) {
    EXPECT_EQ(actual, expected) << what;
    return;
  }
  const float scale = std::max(1.0f, std::abs(expected));
  EXPECT_NEAR(actual, expected, tolerance * scale) << what;
}

}  // namespace annc::test

#endif  // ANNC_TESTS_UNIT_KERNELS_FUSION_KERNEL_TEST_SUPPORT_H_
