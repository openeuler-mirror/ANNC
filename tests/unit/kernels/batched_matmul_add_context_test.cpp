#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "fusion_kernel_test_support.h"
#include "batched_matmul_add_aarch64.cpp"

namespace {

using annc::kernels::KernelStatus;
using annc::test::ExpectNear;
using annc::test::OutputBuffers;

std::vector<float> ReferenceRank3(const std::vector<float>& data,
                                  int64_t dataBatch, int64_t d1, int64_t k,
                                  const std::vector<float>& weight,
                                  int64_t weightBatch, int64_t n,
                                  const std::vector<float>& bias) {
  const int64_t outBatch = dataBatch == 1 ? weightBatch : dataBatch;
  std::vector<float> out(static_cast<size_t>(outBatch * d1 * n));
  for (int64_t i = 0; i < outBatch; ++i) {
    const int64_t di = dataBatch == 1 ? 0 : i;
    const int64_t wi = weightBatch == 1 ? 0 : i;
    for (int64_t j = 0; j < d1; ++j) {
      for (int64_t col = 0; col < n; ++col) {
        float sum = 0.0f;
        for (int64_t t = 0; t < k; ++t) {
          sum += data[(di * d1 + j) * k + t] *
                 weight[(wi * k + t) * n + col];
        }
        out[(i * d1 + j) * n + col] = sum + bias[i * n + col];
      }
    }
  }
  return out;
}

std::vector<float> ReferenceRank4(const std::vector<float>& data, int64_t d0,
                                  int64_t d1, int64_t m, int64_t k,
                                  const std::vector<float>& weight, int64_t w0,
                                  int64_t w1, int64_t n,
                                  const std::vector<float>& bias,
                                  int64_t out0, int64_t out1) {
  std::vector<float> out(static_cast<size_t>(out0 * out1 * m * n));
  for (int64_t i = 0; i < out0; ++i) {
    for (int64_t j = 0; j < out1; ++j) {
      const int64_t di = d0 == 1 ? 0 : i;
      const int64_t dj = d1 == 1 ? 0 : j;
      const int64_t wi = w0 == 1 ? 0 : i;
      const int64_t wj = w1 == 1 ? 0 : j;
      for (int64_t t = 0; t < m; ++t) {
        for (int64_t col = 0; col < n; ++col) {
          float sum = 0.0f;
          for (int64_t kk = 0; kk < k; ++kk) {
            sum += data[((di * d1 + dj) * m + t) * k + kk] *
                   weight[((wi * w1 + wj) * k + kk) * n + col];
          }
          out[((i * out1 + j) * m + t) * n + col] =
              sum + bias[(i * out1 + j) * n + col];
        }
      }
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

void ExpectRank3(const std::vector<float>& got,
                 const std::vector<float>& expected) {
  ASSERT_EQ(got.size(), expected.size());
  for (size_t i = 0; i < got.size(); ++i) {
    ExpectNear(got[i], expected[i], 1e-5f, "rank-3 batched matmul+add");
  }
}

void ExpectRank4(const std::vector<float>& got,
                 const std::vector<float>& expected) {
  ASSERT_EQ(got.size(), expected.size());
  for (size_t i = 0; i < got.size(); ++i) {
    ExpectNear(got[i], expected[i], 1e-5f, "rank-4 batched matmul+add");
  }
}

TEST(BatchedMatMulAddContext, Rank3MatchesReference) {
  const int64_t dataBatch = 2, d1 = 3, k = 4, n = 5;
  std::vector<float> data(static_cast<size_t>(dataBatch * d1 * k));
  std::vector<float> weight(static_cast<size_t>(dataBatch * k * n));
  std::vector<float> bias(static_cast<size_t>(dataBatch * n));
  Fill(&data, 0.125f, 5);
  Fill(&weight, 0.25f, 6);
  Fill(&bias, 0.5f, 3);

  const int64_t dataDims[] = {dataBatch, d1, k};
  const int64_t weightDims[] = {dataBatch, k, n};
  const int64_t biasDims[] = {dataBatch, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      data.data(), 3, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      weight.data(), 3, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      bias.data(), 3, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(batchedMatMulAddImpl(nullptr, &context, &dataRef, &weightRef,
                                 &biasRef),
            KernelStatus::Success);
  EXPECT_EQ(outputs.allocate_calls, 1u);
  ExpectRank3(outputs.output(),
              ReferenceRank3(data, dataBatch, d1, k, weight, dataBatch, n,
                             bias));
}

TEST(BatchedMatMulAddContext, Rank3BroadcastsDataBatch) {
  const int64_t dataBatch = 1, weightBatch = 2, d1 = 3, k = 4, n = 5;
  std::vector<float> data(static_cast<size_t>(dataBatch * d1 * k));
  std::vector<float> weight(static_cast<size_t>(weightBatch * k * n));
  std::vector<float> bias(static_cast<size_t>(weightBatch * n));
  Fill(&data, 0.125f, 5);
  Fill(&weight, 0.25f, 6);
  Fill(&bias, 0.5f, 3);

  const int64_t dataDims[] = {dataBatch, d1, k};
  const int64_t weightDims[] = {weightBatch, k, n};
  const int64_t biasDims[] = {weightBatch, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      data.data(), 3, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      weight.data(), 3, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      bias.data(), 3, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(batchedMatMulAddImpl(nullptr, &context, &dataRef, &weightRef,
                                 &biasRef),
            KernelStatus::Success);
  ExpectRank3(outputs.output(),
              ReferenceRank3(data, dataBatch, d1, k, weight, weightBatch, n,
                             bias));
}

TEST(BatchedMatMulAddContext, Rank3BroadcastsWeightBatch) {
  const int64_t dataBatch = 2, weightBatch = 1, d1 = 3, k = 4, n = 5;
  std::vector<float> data(static_cast<size_t>(dataBatch * d1 * k));
  std::vector<float> weight(static_cast<size_t>(weightBatch * k * n));
  std::vector<float> bias(static_cast<size_t>(dataBatch * n));
  Fill(&data, 0.125f, 5);
  Fill(&weight, 0.25f, 6);
  Fill(&bias, 0.5f, 3);

  const int64_t dataDims[] = {dataBatch, d1, k};
  const int64_t weightDims[] = {weightBatch, k, n};
  const int64_t biasDims[] = {dataBatch, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      data.data(), 3, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      weight.data(), 3, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      bias.data(), 3, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(batchedMatMulAddImpl(nullptr, &context, &dataRef, &weightRef,
                                 &biasRef),
            KernelStatus::Success);
  ExpectRank3(outputs.output(),
              ReferenceRank3(data, dataBatch, d1, k, weight, weightBatch, n,
                             bias));
}

TEST(BatchedMatMulAddContext, Rank3ZeroContractAddsBiasOnly) {
  const int64_t dataBatch = 2, d1 = 3, k = 0, n = 4;
  std::vector<float> data;
  std::vector<float> weight;
  std::vector<float> bias(static_cast<size_t>(dataBatch * n));
  Fill(&bias, 0.5f, 3);

  const int64_t dataDims[] = {dataBatch, d1, k};
  const int64_t weightDims[] = {dataBatch, k, n};
  const int64_t biasDims[] = {dataBatch, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      data.data(), 3, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      weight.data(), 3, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      bias.data(), 3, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(batchedMatMulAddImpl(nullptr, &context, &dataRef, &weightRef,
                                 &biasRef),
            KernelStatus::Success);
  ASSERT_EQ(outputs.output().size(),
            static_cast<size_t>(dataBatch * d1 * n));
  for (int64_t i = 0; i < dataBatch; ++i) {
    for (int64_t j = 0; j < d1; ++j) {
      for (int64_t col = 0; col < n; ++col) {
        EXPECT_EQ(outputs.output()[(i * d1 + j) * n + col],
                  bias[i * n + col]);
      }
    }
  }
}

TEST(BatchedMatMulAddContext, Rank3ZeroRowsAllocatesAnEmptyOutput) {
  const int64_t dataBatch = 2, d1 = 0, k = 3, n = 4;
  std::vector<float> data;
  std::vector<float> weight(static_cast<size_t>(dataBatch * k * n), 1.0f);
  std::vector<float> bias(static_cast<size_t>(dataBatch * n), 1.0f);

  const int64_t dataDims[] = {dataBatch, d1, k};
  const int64_t weightDims[] = {dataBatch, k, n};
  const int64_t biasDims[] = {dataBatch, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      data.data(), 3, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      weight.data(), 3, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      bias.data(), 3, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(batchedMatMulAddImpl(nullptr, &context, &dataRef, &weightRef,
                                 &biasRef),
            KernelStatus::Success);
  EXPECT_EQ(outputs.allocate_calls, 1u);
  EXPECT_EQ(outputs.last_slot, 0u);
  EXPECT_EQ(outputs.last_rank, 3u);
  EXPECT_EQ(outputs.last_dims, (std::vector<int64_t>{dataBatch, d1, n}));
  EXPECT_TRUE(outputs.output().empty());
}

TEST(BatchedMatMulAddContext, Rank3ZeroColumnsAllocatesAnEmptyOutput) {
  const int64_t dataBatch = 2, d1 = 3, k = 4, n = 0;
  std::vector<float> data(static_cast<size_t>(dataBatch * d1 * k), 1.0f);
  std::vector<float> weight;
  std::vector<float> bias;

  const int64_t dataDims[] = {dataBatch, d1, k};
  const int64_t weightDims[] = {dataBatch, k, n};
  const int64_t biasDims[] = {dataBatch, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      data.data(), 3, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      weight.data(), 3, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      bias.data(), 3, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(batchedMatMulAddImpl(nullptr, &context, &dataRef, &weightRef,
                                 &biasRef),
            KernelStatus::Success);
  EXPECT_EQ(outputs.allocate_calls, 1u);
  EXPECT_EQ(outputs.last_slot, 0u);
  EXPECT_EQ(outputs.last_rank, 3u);
  EXPECT_EQ(outputs.last_dims, (std::vector<int64_t>{dataBatch, d1, n}));
  EXPECT_TRUE(outputs.output().empty());
}

TEST(BatchedMatMulAddContext, Rank3RejectsBiasBatchMismatch) {
  const int64_t dataBatch = 2, d1 = 2, k = 2, n = 3;
  std::vector<float> data(static_cast<size_t>(dataBatch * d1 * k), 1.0f);
  std::vector<float> weight(static_cast<size_t>(dataBatch * k * n), 1.0f);
  std::vector<float> bias(static_cast<size_t>(n), 1.0f);  // batch 1

  const int64_t dataDims[] = {dataBatch, d1, k};
  const int64_t weightDims[] = {dataBatch, k, n};
  const int64_t biasDims[] = {1, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      data.data(), 3, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      weight.data(), 3, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      bias.data(), 3, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  EXPECT_EQ(batchedMatMulAddImpl(nullptr, &context, &dataRef, &weightRef,
                                 &biasRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

TEST(BatchedMatMulAddContext, Rank3RejectsNonContiguousData) {
  const int64_t dataBatch = 1, d1 = 2, k = 2, n = 2;
  std::vector<float> data(static_cast<size_t>(dataBatch * d1 * k), 1.0f);
  std::vector<float> weight(static_cast<size_t>(dataBatch * k * n), 1.0f);
  std::vector<float> bias(static_cast<size_t>(dataBatch * n), 1.0f);

  const int64_t dataDims[] = {dataBatch, d1, k};
  const int64_t weightDims[] = {dataBatch, k, n};
  const int64_t biasDims[] = {dataBatch, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      data.data(), 3, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      weight.data(), 3, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      bias.data(), 3, biasDims);
  dataRef.strides[2] = 2;

  OutputBuffers outputs;
  auto context = outputs.context();
  EXPECT_EQ(batchedMatMulAddImpl(nullptr, &context, &dataRef, &weightRef,
                                 &biasRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

TEST(BatchedMatMulAddContext, Rank4MatchesReference) {
  const int64_t d0 = 2, d1 = 1, m = 2, k = 3;
  const int64_t w0 = 2, w1 = 2, n = 4;
  const int64_t out0 = 2, out1 = 2;
  std::vector<float> data(static_cast<size_t>(d0 * d1 * m * k));
  std::vector<float> weight(static_cast<size_t>(w0 * w1 * k * n));
  std::vector<float> bias(static_cast<size_t>(out0 * out1 * n));
  Fill(&data, 0.125f, 5);
  Fill(&weight, 0.25f, 6);
  Fill(&bias, 0.5f, 3);

  const int64_t dataDims[] = {d0, d1, m, k};
  const int64_t weightDims[] = {w0, w1, k, n};
  const int64_t biasDims[] = {out0, out1, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      data.data(), 4, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      weight.data(), 4, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      bias.data(), 4, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(batchedMatMulAddRank4Impl(nullptr, &context, &dataRef, &weightRef,
                                      &biasRef),
            KernelStatus::Success);
  EXPECT_EQ(outputs.allocate_calls, 1u);
  ExpectRank4(outputs.output(),
              ReferenceRank4(data, d0, d1, m, k, weight, w0, w1, n, bias,
                             out0, out1));
}

TEST(BatchedMatMulAddContext, Rank4BroadcastsLeadingDims) {
  const int64_t d0 = 1, d1 = 1, m = 2, k = 3;
  const int64_t w0 = 1, w1 = 1, n = 4;
  const int64_t out0 = 2, out1 = 2;
  std::vector<float> data(static_cast<size_t>(d0 * d1 * m * k));
  std::vector<float> weight(static_cast<size_t>(w0 * w1 * k * n));
  std::vector<float> bias(static_cast<size_t>(out0 * out1 * n));
  Fill(&data, 0.125f, 5);
  Fill(&weight, 0.25f, 6);
  Fill(&bias, 0.5f, 3);

  const int64_t dataDims[] = {d0, d1, m, k};
  const int64_t weightDims[] = {w0, w1, k, n};
  const int64_t biasDims[] = {out0, out1, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      data.data(), 4, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      weight.data(), 4, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      bias.data(), 4, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(batchedMatMulAddRank4Impl(nullptr, &context, &dataRef, &weightRef,
                                      &biasRef),
            KernelStatus::Success);
  ExpectRank4(outputs.output(),
              ReferenceRank4(data, d0, d1, m, k, weight, w0, w1, n, bias,
                             out0, out1));
}

TEST(BatchedMatMulAddContext, Rank4RejectsBatchMismatch) {
  const int64_t d0 = 2, d1 = 1, m = 2, k = 2;
  const int64_t w0 = 3, w1 = 2, n = 2;
  const int64_t out0 = 2, out1 = 2;
  std::vector<float> data(static_cast<size_t>(d0 * d1 * m * k), 1.0f);
  std::vector<float> weight(static_cast<size_t>(w0 * w1 * k * n), 1.0f);
  std::vector<float> bias(static_cast<size_t>(out0 * out1 * n), 1.0f);

  const int64_t dataDims[] = {d0, d1, m, k};
  const int64_t weightDims[] = {w0, w1, k, n};
  const int64_t biasDims[] = {out0, out1, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      data.data(), 4, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      weight.data(), 4, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      bias.data(), 4, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  EXPECT_EQ(batchedMatMulAddRank4Impl(nullptr, &context, &dataRef, &weightRef,
                                      &biasRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

TEST(BatchedMatMulAddContext, Rank4ZeroRowsAllocatesAnEmptyOutput) {
  const int64_t d0 = 2, d1 = 2, m = 0, k = 3;
  const int64_t w0 = 2, w1 = 2, n = 4;
  const int64_t out0 = 2, out1 = 2;
  std::vector<float> data;
  std::vector<float> weight(static_cast<size_t>(w0 * w1 * k * n), 1.0f);
  std::vector<float> bias(static_cast<size_t>(out0 * out1 * n), 1.0f);

  const int64_t dataDims[] = {d0, d1, m, k};
  const int64_t weightDims[] = {w0, w1, k, n};
  const int64_t biasDims[] = {out0, out1, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      data.data(), 4, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      weight.data(), 4, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      bias.data(), 4, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(batchedMatMulAddRank4Impl(nullptr, &context, &dataRef, &weightRef,
                                      &biasRef),
            KernelStatus::Success);
  EXPECT_EQ(outputs.allocate_calls, 1u);
  EXPECT_EQ(outputs.last_slot, 0u);
  EXPECT_EQ(outputs.last_rank, 4u);
  EXPECT_EQ(outputs.last_dims,
            (std::vector<int64_t>{out0, out1, m, n}));
  EXPECT_TRUE(outputs.output().empty());
}

TEST(BatchedMatMulAddContext, Rank3CoversSixteenWideBlock) {
  const int64_t dataBatch = 1, d1 = 2, k = 3, n = 17;
  std::vector<float> data(static_cast<size_t>(dataBatch * d1 * k));
  std::vector<float> weight(static_cast<size_t>(dataBatch * k * n));
  std::vector<float> bias(static_cast<size_t>(dataBatch * n));
  Fill(&data, 0.125f, 5);
  Fill(&weight, 0.25f, 6);
  Fill(&bias, 0.5f, 3);

  const int64_t dataDims[] = {dataBatch, d1, k};
  const int64_t weightDims[] = {dataBatch, k, n};
  const int64_t biasDims[] = {dataBatch, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      data.data(), 3, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      weight.data(), 3, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      bias.data(), 3, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(batchedMatMulAddImpl(nullptr, &context, &dataRef, &weightRef,
                                 &biasRef),
            KernelStatus::Success);
  ExpectRank3(outputs.output(),
              ReferenceRank3(data, dataBatch, d1, k, weight, dataBatch, n,
                             bias));
}

TEST(BatchedMatMulAddContext, Rank4CoversSixteenWideBlock) {
  const int64_t d0 = 1, d1 = 1, m = 1, k = 2;
  const int64_t w0 = 1, w1 = 1, n = 17;
  const int64_t out0 = 1, out1 = 1;
  std::vector<float> data(static_cast<size_t>(d0 * d1 * m * k));
  std::vector<float> weight(static_cast<size_t>(w0 * w1 * k * n));
  std::vector<float> bias(static_cast<size_t>(out0 * out1 * n));
  Fill(&data, 0.125f, 5);
  Fill(&weight, 0.25f, 6);
  Fill(&bias, 0.5f, 3);

  const int64_t dataDims[] = {d0, d1, m, k};
  const int64_t weightDims[] = {w0, w1, k, n};
  const int64_t biasDims[] = {out0, out1, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      data.data(), 4, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      weight.data(), 4, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      bias.data(), 4, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(batchedMatMulAddRank4Impl(nullptr, &context, &dataRef, &weightRef,
                                      &biasRef),
            KernelStatus::Success);
  ExpectRank4(outputs.output(),
              ReferenceRank4(data, d0, d1, m, k, weight, w0, w1, n, bias,
                             out0, out1));
}

TEST(BatchedMatMulAddContext, Rank3RejectsNonContiguousWeightAndBias) {
  const int64_t dataBatch = 1, d1 = 2, k = 2, n = 2;
  std::vector<float> data(static_cast<size_t>(dataBatch * d1 * k), 1.0f);
  std::vector<float> weight(static_cast<size_t>(dataBatch * k * n), 1.0f);
  std::vector<float> bias(static_cast<size_t>(dataBatch * n), 1.0f);

  const int64_t dataDims[] = {dataBatch, d1, k};
  const int64_t weightDims[] = {dataBatch, k, n};
  const int64_t biasDims[] = {dataBatch, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      data.data(), 3, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      weight.data(), 3, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      bias.data(), 3, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  weightRef.strides[2] = 2;
  EXPECT_EQ(batchedMatMulAddImpl(nullptr, &context, &dataRef, &weightRef,
                                 &biasRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);

  weightRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      weight.data(), 3, weightDims);
  biasRef.strides[1] = n + 1;
  EXPECT_EQ(batchedMatMulAddImpl(nullptr, &context, &dataRef, &weightRef,
                                 &biasRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

TEST(BatchedMatMulAddContext, Rank3RejectsNegativeDimensions) {
  const int64_t dataBatch = 1, d1 = 2, k = -1, n = 2;
  std::vector<float> data;
  std::vector<float> weight(static_cast<size_t>(2 * n), 1.0f);
  std::vector<float> bias(static_cast<size_t>(n), 1.0f);

  const int64_t dataDims[] = {dataBatch, d1, k};
  const int64_t weightDims[] = {dataBatch, 2, n};
  const int64_t biasDims[] = {dataBatch, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      data.data(), 3, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      weight.data(), 3, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      bias.data(), 3, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  EXPECT_EQ(batchedMatMulAddImpl(nullptr, &context, &dataRef, &weightRef,
                                 &biasRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

TEST(BatchedMatMulAddContext, Rank4RejectsNegativeDimensions) {
  const int64_t d0 = 1, d1 = 1, m = -1, k = 2;
  const int64_t w0 = 1, w1 = 1, n = 2;
  const int64_t out0 = 1, out1 = 1;
  std::vector<float> data;
  std::vector<float> weight(static_cast<size_t>(w0 * w1 * k * n), 1.0f);
  std::vector<float> bias(static_cast<size_t>(out0 * out1 * n), 1.0f);

  const int64_t dataDims[] = {d0, d1, m, k};
  const int64_t weightDims[] = {w0, w1, k, n};
  const int64_t biasDims[] = {out0, out1, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      data.data(), 4, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      weight.data(), 4, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      bias.data(), 4, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  EXPECT_EQ(batchedMatMulAddRank4Impl(nullptr, &context, &dataRef, &weightRef,
                                      &biasRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

TEST(BatchedMatMulAddContext, Rank4ZeroContractAddsBiasOnly) {
  const int64_t d0 = 1, d1 = 1, m = 2, k = 0;
  const int64_t w0 = 1, w1 = 1, n = 3;
  const int64_t out0 = 1, out1 = 1;
  std::vector<float> data;
  std::vector<float> weight;
  std::vector<float> bias(static_cast<size_t>(out0 * out1 * n));
  Fill(&bias, 0.5f, 3);

  const int64_t dataDims[] = {d0, d1, m, k};
  const int64_t weightDims[] = {w0, w1, k, n};
  const int64_t biasDims[] = {out0, out1, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      data.data(), 4, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      weight.data(), 4, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      bias.data(), 4, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(batchedMatMulAddRank4Impl(nullptr, &context, &dataRef, &weightRef,
                                      &biasRef),
            KernelStatus::Success);
  ASSERT_EQ(outputs.output().size(), static_cast<size_t>(out0 * out1 * m * n));
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      EXPECT_EQ(outputs.output()[row * n + col], bias[col]);
    }
  }
}

TEST(BatchedMatMulAddContext, Rank4ZeroColumnsAllocatesAnEmptyOutput) {
  const int64_t d0 = 1, d1 = 1, m = 2, k = 3;
  const int64_t w0 = 1, w1 = 1, n = 0;
  const int64_t out0 = 1, out1 = 1;
  std::vector<float> data(static_cast<size_t>(d0 * d1 * m * k), 1.0f);
  std::vector<float> weight;
  std::vector<float> bias;

  const int64_t dataDims[] = {d0, d1, m, k};
  const int64_t weightDims[] = {w0, w1, k, n};
  const int64_t biasDims[] = {out0, out1, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      data.data(), 4, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      weight.data(), 4, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      bias.data(), 4, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  ASSERT_EQ(batchedMatMulAddRank4Impl(nullptr, &context, &dataRef, &weightRef,
                                      &biasRef),
            KernelStatus::Success);
  EXPECT_EQ(outputs.allocate_calls, 1u);
  EXPECT_EQ(outputs.last_slot, 0u);
  EXPECT_EQ(outputs.last_rank, 4u);
  EXPECT_EQ(outputs.last_dims,
            (std::vector<int64_t>{out0, out1, m, n}));
  EXPECT_TRUE(outputs.output().empty());
}

TEST(BatchedMatMulAddContext, Rank4RejectsNonContiguousData) {
  const int64_t d0 = 1, d1 = 1, m = 2, k = 2;
  const int64_t w0 = 1, w1 = 1, n = 2;
  const int64_t out0 = 1, out1 = 1;
  std::vector<float> data(static_cast<size_t>(d0 * d1 * m * k), 1.0f);
  std::vector<float> weight(static_cast<size_t>(w0 * w1 * k * n), 1.0f);
  std::vector<float> bias(static_cast<size_t>(out0 * out1 * n), 1.0f);

  const int64_t dataDims[] = {d0, d1, m, k};
  const int64_t weightDims[] = {w0, w1, k, n};
  const int64_t biasDims[] = {out0, out1, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      data.data(), 4, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      weight.data(), 4, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      bias.data(), 4, biasDims);
  dataRef.strides[2] = k + 1;

  OutputBuffers outputs;
  auto context = outputs.context();
  EXPECT_EQ(batchedMatMulAddRank4Impl(nullptr, &context, &dataRef, &weightRef,
                                      &biasRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

TEST(BatchedMatMulAddContext, Rank4RejectsNullArguments) {
  const int64_t d0 = 1, d1 = 1, m = 1, k = 1;
  const int64_t w0 = 1, w1 = 1, n = 1;
  const int64_t out0 = 1, out1 = 1;
  std::vector<float> data(1, 1.0f);
  std::vector<float> weight(1, 1.0f);
  std::vector<float> bias(1, 1.0f);

  const int64_t dataDims[] = {d0, d1, m, k};
  const int64_t weightDims[] = {w0, w1, k, n};
  const int64_t biasDims[] = {out0, out1, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      data.data(), 4, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      weight.data(), 4, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      bias.data(), 4, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  EXPECT_EQ(batchedMatMulAddRank4Impl(nullptr, nullptr, &dataRef, &weightRef,
                                      &biasRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(batchedMatMulAddRank4Impl(nullptr, &context, nullptr, &weightRef,
                                      &biasRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(batchedMatMulAddRank4Impl(nullptr, &context, &dataRef, nullptr,
                                      &biasRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(batchedMatMulAddRank4Impl(nullptr, &context, &dataRef, &weightRef,
                                      nullptr),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

TEST(BatchedMatMulAddContext, Rank3FailsWhenOutputAllocationFails) {
  const int64_t dataBatch = 1, d1 = 1, k = 1, n = 1;
  std::vector<float> data(1, 1.0f);
  std::vector<float> weight(1, 1.0f);
  std::vector<float> bias(1, 1.0f);
  const int64_t dataDims[] = {dataBatch, d1, k};
  const int64_t weightDims[] = {dataBatch, k, n};
  const int64_t biasDims[] = {dataBatch, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      data.data(), 3, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      weight.data(), 3, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      bias.data(), 3, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  outputs.FailNextAllocation();
  EXPECT_EQ(batchedMatMulAddImpl(nullptr, &context, &dataRef, &weightRef,
                                 &biasRef),
            KernelStatus::RuntimeError);
  EXPECT_EQ(outputs.allocate_calls, 1u);
  EXPECT_NE(outputs.last_error.find("output allocation failed"),
            std::string::npos);
}

TEST(BatchedMatMulAddContext, Rank4FailsWhenOutputAllocationFails) {
  const int64_t d0 = 1, d1 = 1, m = 1, k = 1;
  const int64_t w0 = 1, w1 = 1, n = 1;
  const int64_t out0 = 1, out1 = 1;
  std::vector<float> data(1, 1.0f);
  std::vector<float> weight(1, 1.0f);
  std::vector<float> bias(1, 1.0f);
  const int64_t dataDims[] = {d0, d1, m, k};
  const int64_t weightDims[] = {w0, w1, k, n};
  const int64_t biasDims[] = {out0, out1, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      data.data(), 4, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      weight.data(), 4, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef4DF32>(
      bias.data(), 4, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  outputs.FailNextAllocation();
  EXPECT_EQ(batchedMatMulAddRank4Impl(nullptr, &context, &dataRef, &weightRef,
                                      &biasRef),
            KernelStatus::RuntimeError);
  EXPECT_EQ(outputs.allocate_calls, 1u);
  EXPECT_NE(outputs.last_error.find("output allocation failed"),
            std::string::npos);
}

TEST(BatchedMatMulAddContext, RejectsNullArguments) {
  const int64_t dataBatch = 1, d1 = 1, k = 1, n = 1;
  std::vector<float> data(1, 1.0f);
  std::vector<float> weight(1, 1.0f);
  std::vector<float> bias(1, 1.0f);
  const int64_t dataDims[] = {dataBatch, d1, k};
  const int64_t weightDims[] = {dataBatch, k, n};
  const int64_t biasDims[] = {dataBatch, 1, n};
  auto dataRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      data.data(), 3, dataDims);
  auto weightRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      weight.data(), 3, weightDims);
  auto biasRef = annc::kernels::execution::makeMemRef<AnncMemRef3DF32>(
      bias.data(), 3, biasDims);

  OutputBuffers outputs;
  auto context = outputs.context();
  EXPECT_EQ(batchedMatMulAddImpl(nullptr, nullptr, &dataRef, &weightRef,
                                 &biasRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(batchedMatMulAddImpl(nullptr, &context, nullptr, &weightRef,
                                 &biasRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(batchedMatMulAddImpl(nullptr, &context, &dataRef, nullptr,
                                 &biasRef),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(batchedMatMulAddImpl(nullptr, &context, &dataRef, &weightRef,
                                 nullptr),
            KernelStatus::InvalidArgument);
  EXPECT_EQ(outputs.allocate_calls, 0u);
}

}  // namespace
