#ifndef ANNC_TESTS_UNIT_KERNELS_FUSION_KERNEL_TEST_SUPPORT_H_
#define ANNC_TESTS_UNIT_KERNELS_FUSION_KERNEL_TEST_SUPPORT_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "Kernel/ExecutionContext.h"
#include "Kernel/MemRefTypes.h"

namespace annc::test {

// Minimal execution context for the execution-v2 kernel entry points.  The
// fused kernels allocate their output through allocate_output (slot 0) and
// never use temporary buffers, so this harness only has to size one flat
// buffer per slot and hand its data pointer back.
class OutputBuffers {
 public:
  static constexpr uint32_t kSlots = 4;

  static AnncStatusCode Allocate(AnncExecutionHandle* handle, uint32_t slot,
                                 AnncTensorDesc* desc) {
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
    desc->data = count == 0 ? nullptr : self->buffers[slot].data();
    return ANNC_STATUS_OK;
  }

  AnncExecutionContext context() {
    AnncExecutionContext c{};
    c.struct_size = sizeof(c);
    c.abi_version = ANNC_EXECUTION_ABI_VERSION;
    c.handle = reinterpret_cast<AnncExecutionHandle*>(this);
    c.allocate_output = Allocate;
    c.allocate_temp = [](AnncExecutionHandle*, AnncTensorDesc*) {
      return ANNC_STATUS_INVALID_ARGUMENT;
    };
    c.report_error = [](AnncExecutionHandle* handle, AnncStatusCode status,
                        const char* message) {
      auto* self = reinterpret_cast<OutputBuffers*>(handle);
      if (self != nullptr) {
        self->last_error_status = status;
        self->last_error = message == nullptr ? "" : message;
      }
    };
    return c;
  }

  const std::vector<float>& output(uint32_t slot = 0) const {
    return buffers[slot];
  }

  // Make the next allocate_output call fail, to exercise the kernels'
  // allocation-failure path.
  void FailNextAllocation() { fail_next_allocation = true; }

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
