#include <array>
#include <chrono>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "masked_mul_aarch64.cpp"

namespace {
struct Outputs {
  std::array<std::vector<float>, 6> buffers;
  static AnncStatusCode allocate(AnncExecutionHandle* h, uint32_t slot,
                                 AnncTensorDesc* desc) {
    auto& self = *reinterpret_cast<Outputs*>(h);
    if (slot >= self.buffers.size()) return ANNC_STATUS_INVALID_ARGUMENT;
    size_t count = 1;
    for (uint32_t i = 0; i < desc->rank; ++i) count *= desc->dims[i];
    self.buffers[slot].resize(count);
    desc->data = self.buffers[slot].data();
    return ANNC_STATUS_OK;
  }
  AnncExecutionContext context() {
    AnncExecutionContext c{};
    c.struct_size = sizeof(c);
    c.abi_version = ANNC_EXECUTION_ABI_VERSION;
    c.handle = reinterpret_cast<AnncExecutionHandle*>(this);
    c.allocate_output = allocate;
    c.allocate_temp = [](AnncExecutionHandle*, AnncTensorDesc*) {
      return ANNC_STATUS_INVALID_ARGUMENT;
    };
    c.report_error = [](AnncExecutionHandle*, AnncStatusCode, const char*) {};
    return c;
  }
};

// Real arithmetic: argument aliases, selected sum inputs, external inputs,
// multiple outputs, SIMD tails and an escaping flat mask. No zero shortcuts.
TEST(MaskedMulContext, MultiOutputArithmeticAndSpecialValues) {
  for (int64_t n : {1, 3, 4, 7, 8, 12, 17}) {
    for (int64_t product : {0, 1}) {
      const int64_t shape[] = {2, 3, n}, idShape[] = {2, 3};
      std::array<std::vector<float>, 3> values;
      std::array<AnncMemRef3DF32, 3> refs;
      AnncMemRef3DF32* inputs[3];
      const float special[] = {-0.0f, 0.0f, -2.0f, 3.0f,
                              std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()};
      for (size_t j = 0; j < 3; ++j) {
        values[j].resize(6 * n);
        for (int64_t i = 0; i < 6 * n; ++i)
          values[j][i] = special[(i + j) % 6];
        refs[j] = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
            values[j].data(), 3, shape);
        inputs[j] = &refs[j];
      }
      std::vector<int64_t> ids = {0, 1, 0, 2, 0, 3};
      auto idRef = annc::kernels::execution::makeMemRef<AnncMemRef2DI64>(
          ids.data(), 2, idShape);
      Outputs outputs;
      auto ctx = outputs.context();
      // raw: arg1, arg0, arg1; sum uses raw0 + raw1 + external arg2.
      float divisor = product ? 1.0f : 3.0f;
      uint32_t bits;
      std::memcpy(&bits, &divisor, sizeof(bits));
      ASSERT_EQ(maskedMulSiteImpl(nullptr, &ctx, inputs, &idRef, 2, 3, 2,
                                 0x101, 3, product, bits, 0, true),
                annc::kernels::KernelStatus::Success);
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
  std::array<AnncMemRef3DF32, 3> refs;
  AnncMemRef3DF32* inputs[3];
  for (size_t i = 0; i < values.size(); ++i) {
    refs[i] = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
        values[i].data(), 3, shape);
    inputs[i] = &refs[i];
  }
  int64_t id = 1;
  auto ids = annc::kernels::execution::makeMemRef<AnncMemRef2DI64>(&id, 2, idShape);
  Outputs outputs;
  auto ctx = outputs.context();
  ASSERT_EQ(maskedMulSiteImpl(nullptr, &ctx, inputs, &ids, 2, 2, 1,
                             0x10, 3, 0, 0x3f800000, 0, false),
            annc::kernels::KernelStatus::Success);
  for (float value : outputs.buffers[2]) EXPECT_EQ(value, 1.0f);
}

// Manual microbenchmark excluded from CTest: model1's 11-input Sum/Div site.
TEST(MaskedMulContext, DISABLED_Model1FanInMicrobenchmark) {
  constexpr int64_t rows = 3200, n = 8;
  const int64_t shape[] = {32, 100, n}, idShape[] = {32, 100};
  std::array<std::vector<float>, 11> values;
  std::array<AnncMemRef3DF32, 11> refs;
  AnncMemRef3DF32* inputs[11];
  for (size_t j = 0; j < 11; ++j) {
    values[j].assign(rows * n, 0.125f * (j + 1));
    refs[j] = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
        values[j].data(), 3, shape);
    inputs[j] = &refs[j];
  }
  std::vector<int64_t> ids(rows, 1);
  auto idRef = annc::kernels::execution::makeMemRef<AnncMemRef2DI64>(
      ids.data(), 2, idShape);
  Outputs outputs;
  auto ctx = outputs.context();
  auto run = [&] {
    return maskedMulSiteImpl(nullptr, &ctx, inputs, &idRef, 4, 4, 1,
                            0x3210, 15, 0, 0x41200000, 0, false);
  };
  for (int i = 0; i < 20; ++i) ASSERT_EQ(run(), annc::kernels::KernelStatus::Success);
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 1000; ++i) ASSERT_EQ(run(), annc::kernels::KernelStatus::Success);
  const double us = std::chrono::duration<double, std::micro>(
      std::chrono::steady_clock::now() - start).count() / 1000;
  std::printf("model1_fanin_us=%.3f checksum=%.6f\n", us, outputs.buffers[4][0]);
}
}  // namespace
