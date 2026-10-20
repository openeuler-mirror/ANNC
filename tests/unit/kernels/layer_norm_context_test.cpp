#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "fusion_kernel_test_support.h"
#include "layer_norm_aarch64.cpp"

namespace {

using annc::kernels::KernelStatus;
using annc::test::ExpectNear;
using annc::test::OutputBuffers;

std::vector<float> ReferenceLayerNorm(const std::vector<float>& x,
                                      int64_t rows, int64_t cols,
                                      const std::vector<float>& gamma,
                                      const std::vector<float>& beta,
                                      float epsilon) {
  std::vector<float> out(static_cast<size_t>(rows * cols));
  const float colsF = static_cast<float>(cols);
  for (int64_t row = 0; row < rows; ++row) {
    const float* xRow = x.data() + row * cols;
    float sum = 0.0f;
    for (int64_t col = 0; col < cols; ++col) sum += xRow[col];
    const float mean = sum / colsF;

    float sq = 0.0f;
    for (int64_t col = 0; col < cols; ++col) {
      const float d = xRow[col] - mean;
      sq += d * d;
    }
    const float rsqrt = 1.0f / std::sqrt(sq / colsF + epsilon);
    for (int64_t col = 0; col < cols; ++col) {
      const float factor = rsqrt * gamma[col];
      out[row * cols + col] =
          xRow[col] * factor + (beta[col] - mean * factor);
    }
  }
  return out;
}

void Fill(std::vector<float>* values, float scale, int offset) {
  for (size_t i = 0; i < values->size(); ++i) {
    (*values)[i] =
        scale * static_cast<float>(static_cast<int>(i % 11) - offset);
  }
}

void ExpectOutput(const std::vector<float>& got,
                  const std::vector<float>& expected, float tolerance) {
  ASSERT_EQ(got.size(), expected.size());
  for (size_t i = 0; i < got.size(); ++i) {
    ExpectNear(got[i], expected[i], tolerance, "layer norm");
  }
}

AnncMemRef0DF32 MakeScalar(float* value) {
  return annc::kernels::execution::makeMemRef<AnncMemRef0DF32>(value, 0,
                                                               nullptr);
}

TEST(LayerNormContext, ScalarPathMatchesReference) {
  const int64_t rows = 3, cols = 16;
  std::vector<float> x(static_cast<size_t>(rows * cols));
  std::vector<float> gamma(static_cast<size_t>(cols));
  std::vector<float> beta(static_cast<size_t>(cols));
  Fill(&x, 0.25f, 5);
  Fill(&gamma, 0.125f, 3);
  Fill(&beta, 0.0625f, 2);
  const float epsilon = 1e-5f;

  const int64_t inputDims[] = {rows, cols};
  const int64_t vectorDims[] = {cols};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      x.data(), 2, inputDims);
  auto gammaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      gamma.data(), 1, vectorDims);
  auto betaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      beta.data(), 1, vectorDims);
  auto epsRef = MakeScalar(const_cast<float*>(&epsilon));

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(layerNormImpl(nullptr, &context, &input, &gammaRef, &betaRef,
                          &epsRef),
            KernelStatus::Success);
  EXPECT_EQ(outputs.allocate_calls, 1u);
  ExpectOutput(outputs.output(),
               ReferenceLayerNorm(x, rows, cols, gamma, beta, epsilon),
               1e-6f);
}

TEST(LayerNormContext, NeonPathMatchesReference) {
#if !defined(__aarch64__)
  GTEST_SKIP() << "NEON path requires aarch64";
#endif
  const int64_t rows = 2, cols = 333;
  std::vector<float> x(static_cast<size_t>(rows * cols));
  std::vector<float> gamma(static_cast<size_t>(cols));
  std::vector<float> beta(static_cast<size_t>(cols));
  Fill(&x, 0.5f, 6);
  Fill(&gamma, 0.25f, 4);
  Fill(&beta, 0.125f, 3);
  const float epsilon = 1e-5f;

  const int64_t inputDims[] = {rows, cols};
  const int64_t vectorDims[] = {cols};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      x.data(), 2, inputDims);
  auto gammaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      gamma.data(), 1, vectorDims);
  auto betaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      beta.data(), 1, vectorDims);
  auto epsRef = MakeScalar(const_cast<float*>(&epsilon));

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(layerNormImpl(nullptr, &context, &input, &gammaRef, &betaRef,
                          &epsRef),
            KernelStatus::Success);
  ExpectOutput(outputs.output(),
               ReferenceLayerNorm(x, rows, cols, gamma, beta, epsilon),
               1e-5f);
}

TEST(LayerNormContext, SingleColumnMatchesReference) {
  const int64_t rows = 2, cols = 1;
  std::vector<float> x = {1.0f, -3.0f};
  std::vector<float> gamma = {2.0f};
  std::vector<float> beta = {-0.5f};
  const float epsilon = 1e-3f;

  const int64_t inputDims[] = {rows, cols};
  const int64_t vectorDims[] = {cols};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      x.data(), 2, inputDims);
  auto gammaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      gamma.data(), 1, vectorDims);
  auto betaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      beta.data(), 1, vectorDims);
  auto epsRef = MakeScalar(const_cast<float*>(&epsilon));

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(layerNormImpl(nullptr, &context, &input, &gammaRef, &betaRef,
                          &epsRef),
            KernelStatus::Success);
  ExpectOutput(outputs.output(),
               ReferenceLayerNorm(x, rows, cols, gamma, beta, epsilon),
               1e-6f);
}

TEST(LayerNormContext, ZeroRowsAllocatesAnEmptyOutput) {
  const int64_t rows = 0, cols = 16;
  std::vector<float> x;
  std::vector<float> gamma(static_cast<size_t>(cols), 1.0f);
  std::vector<float> beta(static_cast<size_t>(cols), 0.0f);
  const float epsilon = 1e-5f;

  const int64_t inputDims[] = {rows, cols};
  const int64_t vectorDims[] = {cols};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      x.data(), 2, inputDims);
  auto gammaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      gamma.data(), 1, vectorDims);
  auto betaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      beta.data(), 1, vectorDims);
  auto epsRef = MakeScalar(const_cast<float*>(&epsilon));

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(layerNormImpl(nullptr, &context, &input, &gammaRef, &betaRef,
                          &epsRef),
            KernelStatus::Success);
  EXPECT_EQ(outputs.allocate_calls, 1u);
  EXPECT_EQ(outputs.last_slot, 0u);
  EXPECT_EQ(outputs.last_rank, 2u);
  EXPECT_EQ(outputs.last_dims, (std::vector<int64_t>{rows, cols}));
  EXPECT_TRUE(outputs.output().empty());
}

TEST(LayerNormContext, RejectsMismatchedGamma) {
  const int64_t rows = 1, cols = 8;
  std::vector<float> x(static_cast<size_t>(rows * cols), 1.0f);
  std::vector<float> gamma(static_cast<size_t>(cols - 1), 1.0f);
  std::vector<float> beta(static_cast<size_t>(cols), 0.0f);
  const float epsilon = 1e-5f;

  const int64_t inputDims[] = {rows, cols};
  const int64_t gammaDims[] = {cols - 1};
  const int64_t betaDims[] = {cols};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      x.data(), 2, inputDims);
  auto gammaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      gamma.data(), 1, gammaDims);
  auto betaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      beta.data(), 1, betaDims);
  auto epsRef = MakeScalar(const_cast<float*>(&epsilon));

  OutputBuffers outputs;
  auto context = outputs.context();
  EXPECT_EQ(layerNormImpl(nullptr, &context, &input, &gammaRef, &betaRef,
                          &epsRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

TEST(LayerNormContext, RejectsMismatchedBeta) {
  const int64_t rows = 1, cols = 8;
  std::vector<float> x(static_cast<size_t>(rows * cols), 1.0f);
  std::vector<float> gamma(static_cast<size_t>(cols), 1.0f);
  std::vector<float> beta(static_cast<size_t>(cols - 1), 0.0f);
  const float epsilon = 1e-5f;

  const int64_t inputDims[] = {rows, cols};
  const int64_t gammaDims[] = {cols};
  const int64_t betaDims[] = {cols - 1};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      x.data(), 2, inputDims);
  auto gammaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      gamma.data(), 1, gammaDims);
  auto betaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      beta.data(), 1, betaDims);
  auto epsRef = MakeScalar(const_cast<float*>(&epsilon));

  OutputBuffers outputs;
  auto context = outputs.context();
  EXPECT_EQ(layerNormImpl(nullptr, &context, &input, &gammaRef, &betaRef,
                          &epsRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

TEST(LayerNormContext, RejectsZeroColumns) {
  const int64_t rows = 2, cols = 0;
  std::vector<float> x;
  std::vector<float> gamma;
  std::vector<float> beta;
  const float epsilon = 1e-5f;

  const int64_t inputDims[] = {rows, cols};
  const int64_t vectorDims[] = {cols};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      x.data(), 2, inputDims);
  auto gammaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      gamma.data(), 1, vectorDims);
  auto betaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      beta.data(), 1, vectorDims);
  auto epsRef = MakeScalar(const_cast<float*>(&epsilon));

  OutputBuffers outputs;
  auto context = outputs.context();
  EXPECT_EQ(layerNormImpl(nullptr, &context, &input, &gammaRef, &betaRef,
                          &epsRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

TEST(LayerNormContext, RejectsNonContiguousInput) {
  const int64_t rows = 2, cols = 4;
  std::vector<float> x(static_cast<size_t>(rows * cols), 1.0f);
  std::vector<float> gamma(static_cast<size_t>(cols), 1.0f);
  std::vector<float> beta(static_cast<size_t>(cols), 0.0f);
  const float epsilon = 1e-5f;

  const int64_t inputDims[] = {rows, cols};
  const int64_t vectorDims[] = {cols};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      x.data(), 2, inputDims);
  auto gammaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      gamma.data(), 1, vectorDims);
  auto betaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      beta.data(), 1, vectorDims);
  auto epsRef = MakeScalar(const_cast<float*>(&epsilon));
  input.strides[1] = 2;

  OutputBuffers outputs;
  auto context = outputs.context();
  EXPECT_EQ(layerNormImpl(nullptr, &context, &input, &gammaRef, &betaRef,
                          &epsRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);

  input = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      x.data(), 2, inputDims);
  input.strides[0] = cols + 1;
  EXPECT_EQ(layerNormImpl(nullptr, &context, &input, &gammaRef, &betaRef,
                          &epsRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

TEST(LayerNormContext, PropagatesNaN) {
  const int64_t rows = 1, cols = 4;
  std::vector<float> x = {std::numeric_limits<float>::quiet_NaN(), 1.0f, 2.0f,
                          3.0f};
  std::vector<float> gamma(static_cast<size_t>(cols), 1.0f);
  std::vector<float> beta(static_cast<size_t>(cols), 0.0f);
  const float epsilon = 1e-5f;

  const int64_t inputDims[] = {rows, cols};
  const int64_t vectorDims[] = {cols};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      x.data(), 2, inputDims);
  auto gammaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      gamma.data(), 1, vectorDims);
  auto betaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      beta.data(), 1, vectorDims);
  auto epsRef = MakeScalar(const_cast<float*>(&epsilon));

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(layerNormImpl(nullptr, &context, &input, &gammaRef, &betaRef,
                          &epsRef),
            KernelStatus::Success);
  ASSERT_EQ(outputs.output().size(), static_cast<size_t>(cols));
  for (float value : outputs.output()) EXPECT_TRUE(std::isnan(value));
}

TEST(LayerNormContext, RejectsNonContiguousGammaAndBeta) {
  const int64_t rows = 1, cols = 4;
  std::vector<float> x(static_cast<size_t>(rows * cols), 1.0f);
  std::vector<float> gamma(static_cast<size_t>(cols), 1.0f);
  std::vector<float> beta(static_cast<size_t>(cols), 0.0f);
  const float epsilon = 1e-5f;

  const int64_t inputDims[] = {rows, cols};
  const int64_t vectorDims[] = {cols};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      x.data(), 2, inputDims);
  auto gammaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      gamma.data(), 1, vectorDims);
  auto betaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      beta.data(), 1, vectorDims);
  auto epsRef = MakeScalar(const_cast<float*>(&epsilon));

  OutputBuffers outputs;
  auto context = outputs.context();
  gammaRef.strides[0] = 2;
  EXPECT_EQ(layerNormImpl(nullptr, &context, &input, &gammaRef, &betaRef,
                          &epsRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);

  gammaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      gamma.data(), 1, vectorDims);
  betaRef.strides[0] = 2;
  EXPECT_EQ(layerNormImpl(nullptr, &context, &input, &gammaRef, &betaRef,
                          &epsRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

TEST(LayerNormContext, RejectsNegativeRows) {
  const int64_t rows = -1, cols = 4;
  std::vector<float> x;
  std::vector<float> gamma(static_cast<size_t>(cols), 1.0f);
  std::vector<float> beta(static_cast<size_t>(cols), 0.0f);
  const float epsilon = 1e-5f;

  const int64_t inputDims[] = {rows, cols};
  const int64_t vectorDims[] = {cols};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      x.data(), 2, inputDims);
  auto gammaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      gamma.data(), 1, vectorDims);
  auto betaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      beta.data(), 1, vectorDims);
  auto epsRef = MakeScalar(const_cast<float*>(&epsilon));

  OutputBuffers outputs;
  auto context = outputs.context();
  EXPECT_EQ(layerNormImpl(nullptr, &context, &input, &gammaRef, &betaRef,
                          &epsRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

TEST(LayerNormContext, NeonPathPropagatesNaN) {
#if !defined(__aarch64__)
  GTEST_SKIP() << "NEON path requires aarch64";
#endif
  const int64_t rows = 1, cols = 333;
  std::vector<float> x(static_cast<size_t>(cols), 1.0f);
  x[17] = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> gamma(static_cast<size_t>(cols), 1.0f);
  std::vector<float> beta(static_cast<size_t>(cols), 0.0f);
  const float epsilon = 1e-5f;

  const int64_t inputDims[] = {rows, cols};
  const int64_t vectorDims[] = {cols};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      x.data(), 2, inputDims);
  auto gammaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      gamma.data(), 1, vectorDims);
  auto betaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      beta.data(), 1, vectorDims);
  auto epsRef = MakeScalar(const_cast<float*>(&epsilon));

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(layerNormImpl(nullptr, &context, &input, &gammaRef, &betaRef,
                          &epsRef),
            KernelStatus::Success);
  EXPECT_EQ(outputs.allocate_calls, 1u);
  EXPECT_EQ(outputs.last_slot, 0u);
  EXPECT_EQ(outputs.last_rank, 2u);
  EXPECT_EQ(outputs.last_dims, (std::vector<int64_t>{rows, cols}));
  ASSERT_EQ(outputs.output().size(), static_cast<size_t>(cols));
  for (float value : outputs.output()) EXPECT_TRUE(std::isnan(value));
}

TEST(LayerNormContext, FailsWhenOutputAllocationFails) {
  const int64_t rows = 1, cols = 4;
  std::vector<float> x(static_cast<size_t>(rows * cols), 1.0f);
  std::vector<float> gamma(static_cast<size_t>(cols), 1.0f);
  std::vector<float> beta(static_cast<size_t>(cols), 0.0f);
  const float epsilon = 1e-5f;

  const int64_t inputDims[] = {rows, cols};
  const int64_t vectorDims[] = {cols};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      x.data(), 2, inputDims);
  auto gammaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      gamma.data(), 1, vectorDims);
  auto betaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      beta.data(), 1, vectorDims);
  auto epsRef = MakeScalar(const_cast<float*>(&epsilon));

  OutputBuffers outputs;
  auto context = outputs.context();
  outputs.FailNextAllocation();
  EXPECT_EQ(layerNormImpl(nullptr, &context, &input, &gammaRef, &betaRef,
                          &epsRef),
            KernelStatus::RuntimeError);
  EXPECT_EQ(outputs.allocate_calls, 1u);
  EXPECT_NE(outputs.last_error.find("output allocation failed"),
            std::string::npos);
}

TEST(LayerNormContext, RejectsNullArguments) {
  const int64_t rows = 1, cols = 4;
  std::vector<float> x(static_cast<size_t>(rows * cols), 1.0f);
  std::vector<float> gamma(static_cast<size_t>(cols), 1.0f);
  std::vector<float> beta(static_cast<size_t>(cols), 0.0f);
  const float epsilon = 1e-5f;

  const int64_t inputDims[] = {rows, cols};
  const int64_t vectorDims[] = {cols};
  auto input = annc::kernels::execution::makeMemRef<AnncMemRef2DF32>(
      x.data(), 2, inputDims);
  auto gammaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      gamma.data(), 1, vectorDims);
  auto betaRef = annc::kernels::execution::makeMemRef<AnncMemRef1DF32>(
      beta.data(), 1, vectorDims);
  auto epsRef = MakeScalar(const_cast<float*>(&epsilon));

  OutputBuffers outputs;
  auto context = outputs.context();
  EXPECT_EQ(layerNormImpl(nullptr, nullptr, &input, &gammaRef, &betaRef,
                          &epsRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(layerNormImpl(nullptr, &context, nullptr, &gammaRef, &betaRef,
                          &epsRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(layerNormImpl(nullptr, &context, &input, nullptr, &betaRef,
                          &epsRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(layerNormImpl(nullptr, &context, &input, &gammaRef, nullptr,
                          &epsRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(layerNormImpl(nullptr, &context, &input, &gammaRef, &betaRef,
                          nullptr),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

}  // namespace
