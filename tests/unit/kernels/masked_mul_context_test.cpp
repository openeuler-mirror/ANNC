#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>

#include "masked_mul_aarch64.cpp"

namespace {
struct Tensor {
  void* data = nullptr;
  uint32_t rank = 0;
  AnncElementType type = ANNC_ELEMENT_TYPE_F32;
  std::array<int64_t, 3> sizes{}, strides{};
  Tensor() = default;
  Tensor(void* ptr, uint32_t ndim, const int64_t* shape,
         AnncElementType element = ANNC_ELEMENT_TYPE_F32)
      : data(ptr), rank(ndim), type(element) {
    int64_t stride = 1;
    for (uint32_t i = rank; i > 0; --i) {
      sizes[i - 1] = shape[i - 1];
      strides[i - 1] = stride;
      stride *= shape[i - 1];
    }
  }
  AnncTensorView view() const {
    return {data,
            data,
            0,
            sizes.data(),
            strides.data(),
            type,
            rank,
            ANNC_LAYOUT_STRIDED,
            ANNC_TENSOR_BOUND};
  }
};

struct Outputs {
  std::array<std::vector<float>, 11> buffers;
  uint32_t allocate_calls = 0;
  bool fail_next_allocation = false;
  AnncStatusCode last_error_status = ANNC_STATUS_OK;
  std::string last_error;
  std::array<Tensor, 11> tensors;
  std::array<AnncTensorView, 11> views{};
  AnncStatusCode status = ANNC_STATUS_OK;
  static AnncStatusCode parallel(void*, int64_t total, int64_t,
                                 AnncContextTask task, void* task_data) {
    if (task == nullptr || total < 0) return ANNC_STATUS_INVALID_ARGUMENT;
    if (total != 0) task(task_data, 0, total);
    return ANNC_STATUS_OK;
  }
  static int32_t threads(void*) { return 1; }
  static AnncStatusCode allocate(void* opaque, uint32_t slot,
                                 const AnncTensorDesc* desc,
                                 AnncTensorView* out) {
    auto& self = *static_cast<Outputs*>(opaque);
    if (slot >= self.buffers.size()) return ANNC_STATUS_INVALID_ARGUMENT;
    ++self.allocate_calls;
    if (self.fail_next_allocation) {
      self.fail_next_allocation = false;
      return ANNC_STATUS_ALLOCATION_FAILED;
    }
    size_t count = 1;
    for (uint32_t i = 0; i < desc->rank; ++i) count *= desc->dims[i];
    self.buffers[slot].resize(count);
    self.tensors[slot] =
        Tensor(self.buffers[slot].data(), desc->rank, desc->dims);
    *out = self.tensors[slot].view();
    return ANNC_STATUS_OK;
  }
  template <size_t Arity>
  AnncStatusCode run(const std::array<Tensor, Arity>& data, const Tensor& ids,
                     std::array<int64_t, 8> values, bool flat,
                     bool nullContext = false, bool nullIds = false) {
    std::array<AnncTensorView, Arity + 1> inputs;
    for (size_t i = 0; i < Arity; ++i) inputs[i] = data[i].view();
    inputs[Arity] = nullIds ? AnncTensorView{} : ids.view();
    std::array<AnncAttributeView, 8> attrs;
    for (size_t i = 0; i < values.size(); ++i)
      attrs[i] = {ANNC_ATTRIBUTE_I64, 0, &values[i], 1};
    AnncRuntimeServices services{};
    services.struct_size = sizeof(services);
    services.abi_version = ANNC_CONTEXT_ABI_VERSION;
    services.opaque = this;
    services.allocate_output = allocate;
    services.allocate_temp = [](void*, const AnncTensorDesc*, AnncTensorView*) {
      return ANNC_STATUS_INVALID_ARGUMENT;
    };
    services.report_error = [](void* opaque, AnncStatusCode code,
                               const char* message) {
      auto& self = *static_cast<Outputs*>(opaque);
      self.status = self.last_error_status = code;
      self.last_error = message ? message : "";
    };
    services.get_status = [](void* self) {
      return static_cast<Outputs*>(self)->status;
    };
    services.parallel_for = parallel;
    services.num_threads = threads;
    AnncKernelContext context{
        sizeof(AnncKernelContext),
        ANNC_CONTEXT_ABI_VERSION,
        inputs.data(),
        static_cast<uint32_t>(inputs.size()),
        views.data(),
        static_cast<uint32_t>(values[1] + values[2] + flat),
        attrs.data(),
        static_cast<uint32_t>(attrs.size()),
        nullptr,
        0,
        &services};
    return annc::kernels::invokeContextKernel(
        nullContext ? nullptr : &context, [&](auto& c) {
          return maskedMulSiteImpl<Arity>(c, flat);
        });
  }
};

// Real arithmetic: argument aliases, selected sum inputs, external inputs,
// multiple outputs, SIMD tails and an escaping flat mask. No zero shortcuts.
TEST(MaskedMulContext, MultiOutputArithmeticAndSpecialValues) {
  for (int64_t n : {1, 3, 4, 7, 8, 12, 17}) {
    for (int64_t product : {0, 1}) {
      const int64_t shape[] = {2, 3, n}, idShape[] = {2, 3};
      std::array<std::vector<float>, 3> values;
      std::array<Tensor, 3> refs;
      const float special[] = {-0.0f,
                               0.0f,
                               -2.0f,
                               3.0f,
                               std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::quiet_NaN()};
      for (size_t j = 0; j < 3; ++j) {
        values[j].resize(6 * n);
        for (int64_t i = 0; i < 6 * n; ++i) values[j][i] = special[(i + j) % 6];
        refs[j] = Tensor(values[j].data(), 3, shape);
      }
      std::vector<int64_t> ids = {0, 1, 0, 2, 0, 3};
      auto idRef = Tensor(ids.data(), 2, idShape, ANNC_ELEMENT_TYPE_I64);
      Outputs outputs;
      // raw: arg1, arg0, arg1; sum uses raw0 + raw1 + external arg2.
      float divisor = product ? 1.0f : 3.0f;
      uint32_t bits;
      std::memcpy(&bits, &divisor, sizeof(bits));
      ASSERT_EQ(
          outputs.run(refs, idRef, {2, 3, 2, 0x101, 3, product, bits, 0}, true),
          ANNC_STATUS_OK);
      for (int64_t row = 0; row < 6; ++row) {
        float factor = ids[row] ? 1.0f : 0.0f;
        EXPECT_EQ(outputs.buffers[5][row], factor);
        for (int64_t col = 0; col < n; ++col) {
          int64_t i = row * n + col;
          float a = values[1][i] * factor, b = values[0][i] * factor;
          float merged = product ? ((1.0f * a) * b) * values[2][i]
                                 : ((0.0f + a) + b) + values[2][i];
          const float expected[] = {a, b, a, (merged / divisor) * factor,
                                    (merged / divisor) * factor};
          for (int slot = 0; slot < 5; ++slot) {
            float actual = outputs.buffers[slot][i], want = expected[slot];
            if (std::isnan(want)) {
              EXPECT_TRUE(std::isnan(actual));
            } else {
              EXPECT_EQ(actual, want) << "n=" << n << " slot=" << slot;
              if (want == 0) {
                EXPECT_EQ(std::signbit(actual), std::signbit(want));
              }
            }
          }
        }
      }
    }
  }
}

TEST(MaskedMulContext, FiniteCancellationKeepsInputOrder) {
  const int64_t shape[] = {1, 1, 5}, idShape[] = {1, 1};
  std::array<std::vector<float>, 3> values = {
      std::vector<float>(5, 16777216.0f), std::vector<float>(5, -16777216.0f),
      std::vector<float>(5, 1.0f)};
  std::array<Tensor, 3> refs;
  for (size_t i = 0; i < values.size(); ++i) {
    refs[i] = Tensor(values[i].data(), 3, shape);
  }
  int64_t id = 1;
  auto ids = Tensor(&id, 2, idShape, ANNC_ELEMENT_TYPE_I64);
  Outputs outputs;
  ASSERT_EQ(outputs.run(refs, ids, {2, 2, 1, 0x10, 3, 0, 0x3f800000, 0}, false),
            ANNC_STATUS_OK);
  for (float value : outputs.buffers[2]) EXPECT_EQ(value, 1.0f);
}

// rows == d0*d1 == 0: the kernel still has to allocate every declared output
// slot (the host checks that), then returns before touching any element.
TEST(MaskedMulContext, ZeroRowsAllocatesAnEmptyOutput) {
  auto run = [](int64_t d0, int64_t d1) {
    const int64_t shape[] = {d0, d1, 4}, idShape[] = {d0, d1};
    std::array<std::vector<float>, 2> values;
    std::array<Tensor, 2> refs;

    for (size_t i = 0; i < 2; ++i) {
      refs[i] = Tensor(values[i].data(), 3, shape);
    }
    std::vector<int64_t> ids;
    auto idRef = Tensor(ids.data(), 2, idShape, ANNC_ELEMENT_TYPE_I64);
    Outputs outputs;

    EXPECT_EQ(
        outputs.run(refs, idRef, {2, 2, 0, 0x10, 0, 0, 0x3f800000, 0}, false),
        ANNC_STATUS_OK);
    EXPECT_EQ(outputs.allocate_calls, 2u);
    EXPECT_TRUE(outputs.buffers[0].empty());
    EXPECT_TRUE(outputs.buffers[1].empty());
  };
  run(0, 3);  // empty batch
  run(2, 0);  // empty sequence
}

// Negative extents have no meaning as shapes; reject before allocating.
TEST(MaskedMulContext, RejectsNegativeDimensions) {
  auto run = [](int64_t d0, int64_t d1, int64_t n) {
    const int64_t shape[] = {d0, d1, n}, idShape[] = {d0, d1};
    std::array<std::vector<float>, 2> values;
    std::array<Tensor, 2> refs;

    for (size_t i = 0; i < 2; ++i) {
      refs[i] = Tensor(values[i].data(), 3, shape);
    }
    std::vector<int64_t> ids;
    auto idRef = Tensor(ids.data(), 2, idShape, ANNC_ELEMENT_TYPE_I64);
    Outputs outputs;

    EXPECT_EQ(
        outputs.run(refs, idRef, {2, 2, 0, 0x10, 0, 0, 0x3f800000, 0}, false),
        ANNC_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(outputs.allocate_calls, 0u);
  };
  run(-1, 3, 4);
  run(2, -1, 4);
  run(2, 3, -1);
}

// The kernel reads inputs and ids with explicit row-major strides; a
// non-contiguous operand has to be rejected instead of silently misread.
TEST(MaskedMulContext, RejectsNonContiguousInputs) {
  const int64_t shape[] = {2, 3, 4}, idShape[] = {2, 3};
  std::array<std::vector<float>, 2> values = {std::vector<float>(24, 1.0f),
                                              std::vector<float>(24, 2.0f)};
  std::vector<int64_t> ids(6, 1);

  auto run = [&](bool nonContiguousData, bool nonContiguousIds) {
    std::array<Tensor, 2> refs;

    for (size_t i = 0; i < 2; ++i) {
      refs[i] = Tensor(values[i].data(), 3, shape);
    }
    auto idRef = Tensor(ids.data(), 2, idShape, ANNC_ELEMENT_TYPE_I64);
    if (nonContiguousData) refs[1].strides[2] = 2;
    if (nonContiguousIds) idRef.strides[1] = 2;

    Outputs outputs;

    EXPECT_EQ(
        outputs.run(refs, idRef, {2, 2, 0, 0x10, 0, 0, 0x3f800000, 0}, false),
        ANNC_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(outputs.allocate_calls, 0u);
  };
  run(true, false);  // data stride
  run(false, true);  // ids stride
  run(true, true);
}

// The run loop allocates every output through the execution context; a failed
// allocation is reported to the host and must not be treated as success.
TEST(MaskedMulContext, FailsWhenOutputAllocationFails) {
  const int64_t shape[] = {1, 2, 4}, idShape[] = {1, 2};
  std::vector<float> value(8, 1.0f);
  auto dataRef = Tensor(value.data(), 3, shape);
  std::array<Tensor, 2> refs = {dataRef, dataRef};
  std::vector<int64_t> ids(2, 1);
  auto idRef = Tensor(ids.data(), 2, idShape, ANNC_ELEMENT_TYPE_I64);

  Outputs outputs;

  outputs.fail_next_allocation = true;
  EXPECT_EQ(
      outputs.run(refs, idRef, {2, 2, 0, 0x10, 0, 0, 0x3f800000, 0}, false),
      ANNC_STATUS_ALLOCATION_FAILED);
  EXPECT_EQ(outputs.allocate_calls, 1u);
  EXPECT_EQ(outputs.last_error_status, ANNC_STATUS_ALLOCATION_FAILED);
  EXPECT_NE(outputs.last_error.find("output allocation failed"),
            std::string::npos);
}

// The 11-data-argument entry point (MaskedMul11) is the largest ABI the
// kernel spec registers; model1's Sum/Div site exercises it.  Until now only
// the disabled microbenchmark reached this arity.
TEST(MaskedMulContext, ElevenInputSiteProducesEveryOutput) {
  constexpr int64_t rows = 2, n = 4;
  const int64_t shape[] = {1, rows, n}, idShape[] = {1, rows};
  std::array<std::vector<float>, 11> values;
  std::array<Tensor, 11> refs;

  for (size_t i = 0; i < values.size(); ++i) {
    values[i].resize(rows * n);
    for (int64_t j = 0; j < rows * n; ++j)
      values[i][j] = 0.25f * static_cast<float>(i + j);
    refs[i] = Tensor(values[i].data(), 3, shape);
  }
  int64_t idValues[rows] = {1, 0};
  auto idRef = Tensor(idValues, 2, idShape, ANNC_ELEMENT_TYPE_I64);

  Outputs outputs;

  ASSERT_EQ(
      outputs.run(refs, idRef,
                  {11, 11, 0, 0xA9876543210LL, 0, 0, 0x3f800000, 0}, false),
      ANNC_STATUS_OK);
  EXPECT_EQ(outputs.allocate_calls, 11u);
  for (int64_t row = 0; row < rows; ++row) {
    const float factor = idValues[row] != 0 ? 1.0f : 0.0f;
    for (size_t i = 0; i < values.size(); ++i) {
      for (int64_t col = 0; col < n; ++col) {
        const int64_t index = row * n + col;
        EXPECT_EQ(outputs.buffers[i][index], values[i][index] * factor)
            << "input=" << i << " row=" << row << " col=" << col;
      }
    }
  }
}

// n == 0: every output slot is still allocated (empty) before the early
// return, same as the zero-row path.
TEST(MaskedMulContext, ZeroColumnsAllocatesAnEmptyOutput) {
  const int64_t shape[] = {2, 3, 0}, idShape[] = {2, 3};
  std::array<std::vector<float>, 2> values;
  std::array<Tensor, 2> refs;

  for (size_t i = 0; i < 2; ++i) {
    refs[i] = Tensor(values[i].data(), 3, shape);
  }
  std::vector<int64_t> ids(6, 1);
  auto idRef = Tensor(ids.data(), 2, idShape, ANNC_ELEMENT_TYPE_I64);
  Outputs outputs;

  EXPECT_EQ(
      outputs.run(refs, idRef, {2, 2, 0, 0x10, 0, 0, 0x3f800000, 0}, false),
      ANNC_STATUS_OK);
  EXPECT_EQ(outputs.allocate_calls, 2u);
  EXPECT_TRUE(outputs.buffers[0].empty());
  EXPECT_TRUE(outputs.buffers[1].empty());
}

// Malformed site metadata: every case must be rejected by the kernel's own
// validation instead of reading out of bounds.  The divisor check runs after
// allocation (the host requires every declared slot to be allocated first),
// so that one case still reports two allocation calls.
TEST(MaskedMulContext, RejectsInvalidSiteMetadata) {
  const int64_t shape[] = {1, 2, 4}, idShape[] = {1, 2};
  std::array<std::vector<float>, 2> values = {std::vector<float>(8, 1.0f),
                                              std::vector<float>(8, 2.0f)};
  std::vector<int64_t> ids(2, 1);

  auto run = [&](bool nullExecution, bool nullIds, int64_t dataArgs,
                 int64_t rawMuls, int64_t fedMuls, int64_t argMap,
                 int64_t divisorBits, uint32_t expectedAllocations) {
    std::array<Tensor, 2> refs;

    for (size_t i = 0; i < 2; ++i) {
      refs[i] = Tensor(values[i].data(), 3, shape);
    }
    auto idRef = Tensor(ids.data(), 2, idShape, ANNC_ELEMENT_TYPE_I64);
    Outputs outputs;

    EXPECT_EQ(
        outputs.run(refs, idRef,
                    {dataArgs, rawMuls, fedMuls, argMap, 1, 0, divisorBits, 0},
                    false, nullExecution, nullIds),
        ANNC_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(outputs.allocate_calls, expectedAllocations);
  };

  run(true, false, 2, 1, 0, 0x0, 0x3f800000, 0u);    // null execution
  run(false, true, 2, 1, 0, 0x0, 0x3f800000, 0u);    // null ids
  run(false, false, 0, 1, 0, 0x0, 0x3f800000, 0u);   // dataArgs < 1
  run(false, false, 2, 0, 0, 0x0, 0x3f800000, 0u);   // rawMuls < 1
  run(false, false, 2, 1, -1, 0x0, 0x3f800000, 0u);  // fedMuls < 0
  run(false, false, 2, 1, 0, 0x2, 0x3f800000, 0u);   // argMap out of range
  run(false, false, 3, 1, 0, 0x0, 0x3f800000, 0u);   // dataArgs > Arity
  run(false, false, 2, 1, 1, 0x0, 0x0, 2u);          // divisor 0 with fed Mul
}

// Manual microbenchmark excluded from CTest: model1's 11-input Sum/Div site.
TEST(MaskedMulContext, DISABLED_Model1FanInMicrobenchmark) {
  constexpr int64_t rows = 3200, n = 8;
  const int64_t shape[] = {32, 100, n}, idShape[] = {32, 100};
  std::array<std::vector<float>, 11> values;
  std::array<Tensor, 11> refs;
  for (size_t j = 0; j < 11; ++j) {
    values[j].assign(rows * n, 0.125f * (j + 1));
    refs[j] = Tensor(values[j].data(), 3, shape);
  }
  std::vector<int64_t> ids(rows, 1);
  auto idRef = Tensor(ids.data(), 2, idShape, ANNC_ELEMENT_TYPE_I64);
  Outputs outputs;
  auto run = [&] {
    return outputs.run(refs, idRef, {4, 4, 1, 0x3210, 15, 0, 0x41200000, 0},
                       false);
  };
  for (int i = 0; i < 20; ++i) ASSERT_EQ(run(), ANNC_STATUS_OK);
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 1000; ++i) ASSERT_EQ(run(), ANNC_STATUS_OK);
  const double us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - start)
                        .count() /
                    1000;
  std::printf("model1_fanin_us=%.3f checksum=%.6f\n", us,
              outputs.buffers[4][0]);
}
}  // namespace
