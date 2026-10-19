#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "fusion_kernel_test_support.h"
#include "tensordot_matmul_aarch64.cpp"

namespace {

using annc::kernels::KernelStatus;
using annc::test::ExpectNear;
using annc::test::OutputBuffers;

TEST(TensordotMatMulContext, MatchesReferenceArithmetic) {
  struct Shape {
    int64_t d0, d1, k, n;
  };
  for (const Shape shape : {Shape{2, 3, 4, 5}, Shape{1, 1, 1, 1},
                            Shape{3, 2, 7, 17}}) {
    std::vector<float> x(shape.d0 * shape.d1 * shape.k);
    std::vector<float> w(shape.k * shape.n);
    for (size_t i = 0; i < x.size(); ++i) {
      x[i] = 0.125f * static_cast<float>(static_cast<int>(i % 11) - 5);
    }
    for (size_t i = 0; i < w.size(); ++i) {
      w[i] = 0.25f * static_cast<float>(static_cast<int>((i * 3) % 13) - 6);
    }

    const int64_t inputDims[] = {shape.d0, shape.d1, shape.k};
    const int64_t weightDims[] = {shape.k, shape.n};
    auto input = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
        x.data(), 3, inputDims);
    auto weight = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
        w.data(), 2, weightDims);

    OutputBuffers outputs;
    auto context = outputs.context();
    ASSERT_EQ(tensordotMatMulImpl(nullptr, &context, &input, &weight),
              KernelStatus::Success);
    ASSERT_EQ(outputs.allocate_calls, 1u);
    ASSERT_EQ(outputs.output().size(),
              static_cast<size_t>(shape.d0 * shape.d1 * shape.n));

    for (int64_t i = 0; i < shape.d0; ++i) {
      for (int64_t j = 0; j < shape.d1; ++j) {
        for (int64_t col = 0; col < shape.n; ++col) {
          float expected = 0.0f;
          for (int64_t t = 0; t < shape.k; ++t) {
            expected += x[(i * shape.d1 + j) * shape.k + t] *
                        w[t * shape.n + col];
          }
          ExpectNear(outputs.output()[(i * shape.d1 + j) * shape.n + col],
                     expected, 1e-5f, "tensordot arithmetic");
        }
      }
    }
  }
}

TEST(TensordotMatMulContext, ZeroContractWritesZeros) {
  const int64_t d0 = 2, d1 = 2, k = 0, n = 3;
  std::vector<float> x;
  std::vector<float> w;
  const int64_t inputDims[] = {d0, d1, k};
  const int64_t weightDims[] = {k, n};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      x.data(), 3, inputDims);
  auto weight = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      w.data(), 2, weightDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(tensordotMatMulImpl(nullptr, &context, &input, &weight),
            KernelStatus::Success);
  ASSERT_EQ(outputs.allocate_calls, 1u);
  EXPECT_EQ(outputs.last_slot, 0u);
  EXPECT_EQ(outputs.last_rank, 3u);
  EXPECT_EQ(outputs.last_element_type, ANNC_ELEMENT_TYPE_F32);
  EXPECT_EQ(outputs.last_dims, (std::vector<int64_t>{d0, d1, n}));
  ASSERT_EQ(outputs.output().size(), static_cast<size_t>(d0 * d1 * n));
  for (float value : outputs.output()) EXPECT_EQ(value, 0.0f);
}

TEST(TensordotMatMulContext, ZeroRowsAllocatesAnEmptyOutput) {
  const int64_t d0 = 0, d1 = 2, k = 3, n = 4;
  std::vector<float> x;
  std::vector<float> w(static_cast<size_t>(k * n), 1.0f);
  const int64_t inputDims[] = {d0, d1, k};
  const int64_t weightDims[] = {k, n};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      x.data(), 3, inputDims);
  auto weight = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      w.data(), 2, weightDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(tensordotMatMulImpl(nullptr, &context, &input, &weight),
            KernelStatus::Success);
  EXPECT_EQ(outputs.allocate_calls, 1u);
  EXPECT_EQ(outputs.last_slot, 0u);
  EXPECT_EQ(outputs.last_rank, 3u);
  EXPECT_EQ(outputs.last_dims, (std::vector<int64_t>{d0, d1, n}));
  EXPECT_TRUE(outputs.output().empty());
}

TEST(TensordotMatMulContext, ZeroColumnsAllocatesAnEmptyOutput) {
  const int64_t d0 = 2, d1 = 2, k = 3, n = 0;
  std::vector<float> x(static_cast<size_t>(d0 * d1 * k), 1.0f);
  std::vector<float> w;
  const int64_t inputDims[] = {d0, d1, k};
  const int64_t weightDims[] = {k, n};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      x.data(), 3, inputDims);
  auto weight = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      w.data(), 2, weightDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(tensordotMatMulImpl(nullptr, &context, &input, &weight),
            KernelStatus::Success);
  EXPECT_EQ(outputs.allocate_calls, 1u);
  EXPECT_EQ(outputs.last_slot, 0u);
  EXPECT_EQ(outputs.last_rank, 3u);
  EXPECT_EQ(outputs.last_dims, (std::vector<int64_t>{d0, d1, n}));
  EXPECT_TRUE(outputs.output().empty());
}

TEST(TensordotMatMulContext, RejectsMismatchedContractDimension) {
  const int64_t d0 = 1, d1 = 1, k = 2, n = 3;
  std::vector<float> x(static_cast<size_t>(d0 * d1 * k), 1.0f);
  std::vector<float> w(9, 1.0f);  // weight first dim is 3, not k=2
  const int64_t inputDims[] = {d0, d1, k};
  const int64_t weightDims[] = {3, n};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      x.data(), 3, inputDims);
  auto weight = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      w.data(), 2, weightDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  EXPECT_EQ(tensordotMatMulImpl(nullptr, &context, &input, &weight),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

TEST(TensordotMatMulContext, RejectsNonContiguousInputs) {
  const int64_t d0 = 1, d1 = 1, k = 2, n = 2;
  std::vector<float> x(static_cast<size_t>(d0 * d1 * k), 1.0f);
  std::vector<float> w(static_cast<size_t>(k * n), 1.0f);
  const int64_t inputDims[] = {d0, d1, k};
  const int64_t weightDims[] = {k, n};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      x.data(), 3, inputDims);
  auto weight = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      w.data(), 2, weightDims);
  input.strides[2] = 2;

  OutputBuffers outputs;
  auto context = outputs.context();
  EXPECT_EQ(tensordotMatMulImpl(nullptr, &context, &input, &weight),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);

  input = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      x.data(), 3, inputDims);
  weight.strides[0] = n + 1;
  EXPECT_EQ(tensordotMatMulImpl(nullptr, &context, &input, &weight),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);

  weight = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      w.data(), 2, weightDims);
  weight.strides[1] = 2;
  EXPECT_EQ(tensordotMatMulImpl(nullptr, &context, &input, &weight),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);

  input = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      x.data(), 3, inputDims);
  weight = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      w.data(), 2, weightDims);
  input.strides[1] = k + 1;
  EXPECT_EQ(tensordotMatMulImpl(nullptr, &context, &input, &weight),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);

  input = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      x.data(), 3, inputDims);
  input.strides[0] = d1 * k + 1;
  EXPECT_EQ(tensordotMatMulImpl(nullptr, &context, &input, &weight),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

TEST(TensordotMatMulContext, RejectsNegativeDimensions) {
  const int64_t d0 = 1, d1 = 1, k = -1, n = 2;
  std::vector<float> x;
  std::vector<float> w(static_cast<size_t>(2), 1.0f);
  const int64_t inputDims[] = {d0, d1, k};
  const int64_t weightDims[] = {2, n};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      x.data(), 3, inputDims);
  auto weight = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      w.data(), 2, weightDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  EXPECT_EQ(tensordotMatMulImpl(nullptr, &context, &input, &weight),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

TEST(TensordotMatMulContext, FailsWhenOutputAllocationFails) {
  const int64_t d0 = 1, d1 = 1, k = 1, n = 1;
  std::vector<float> x(1, 1.0f);
  std::vector<float> w(1, 1.0f);
  const int64_t inputDims[] = {d0, d1, k};
  const int64_t weightDims[] = {k, n};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      x.data(), 3, inputDims);
  auto weight = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      w.data(), 2, weightDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  outputs.FailNextAllocation();
  EXPECT_EQ(tensordotMatMulImpl(nullptr, &context, &input, &weight),
            KernelStatus::RuntimeError);
  EXPECT_EQ(outputs.allocate_calls, 1u);
  EXPECT_NE(outputs.last_error.find("output allocation failed"),
            std::string::npos);
}

TEST(TensordotMatMulContext, RejectsNullArguments) {
  const int64_t d0 = 1, d1 = 1, k = 1, n = 1;
  std::vector<float> x(1, 1.0f);
  std::vector<float> w(1, 1.0f);
  const int64_t inputDims[] = {d0, d1, k};
  const int64_t weightDims[] = {k, n};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      x.data(), 3, inputDims);
  auto weight = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      w.data(), 2, weightDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  EXPECT_EQ(tensordotMatMulImpl(nullptr, nullptr, &input, &weight),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(tensordotMatMulImpl(nullptr, &context, nullptr, &weight),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(tensordotMatMulImpl(nullptr, &context, &input, nullptr),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

TEST(TensordotMatMulContext, PropagatesNaN) {
  const int64_t d0 = 1, d1 = 1, k = 1, n = 1;
  std::vector<float> x = {std::numeric_limits<float>::quiet_NaN()};
  std::vector<float> w = {2.0f};
  const int64_t inputDims[] = {d0, d1, k};
  const int64_t weightDims[] = {k, n};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      x.data(), 3, inputDims);
  auto weight = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      w.data(), 2, weightDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(tensordotMatMulImpl(nullptr, &context, &input, &weight),
            KernelStatus::Success);
  ASSERT_EQ(outputs.output().size(), 1u);
  EXPECT_TRUE(std::isnan(outputs.output()[0]));
}

}  // namespace
