/* Copyright 2026 The Huawei Technologies Co. Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *     ==============================================================================*/

#include <cmath>
#include <vector>

#include "absl/strings/match.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

struct BiasLayout {
  int64 dims[3] = {1, 1, 1};
  int64 strides[3] = {0, 0, 0};
};

std::vector<float> MakeRange(int64 size, float start, float step) {
  std::vector<float> values(size);
  for (int64 i = 0; i < size; ++i) {
    values[i] = start + step * static_cast<float>(i);
  }
  return values;
}

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void BuildBiasLayout(const Tensor& bias, BiasLayout* layout) {
  for (int i = 0; i < bias.dims(); ++i) {
    layout->dims[3 - bias.dims() + i] = bias.dim_size(i);
  }
  layout->strides[2] = (layout->dims[2] == 1) ? 0 : 1;
  layout->strides[1] = (layout->dims[1] == 1) ? 0 : layout->dims[2];
  layout->strides[0] =
      (layout->dims[0] == 1) ? 0 : layout->dims[1] * layout->dims[2];
}

void ComputeReferenceBatchMatMulAddSigmoid(const Tensor& lhs, const Tensor& rhs,
                                           const Tensor& bias, Tensor* output) {
  const int64 batch = lhs.dim_size(0);
  const int64 m = lhs.dim_size(1);
  const int64 k = lhs.dim_size(2);
  const bool rhs_is_batched = rhs.dims() == 3;
  const int64 n = rhs.dim_size(rhs_is_batched ? 2 : 1);

  const float* lhs_ptr = lhs.flat<float>().data();
  const float* rhs_ptr = rhs.flat<float>().data();
  const float* bias_ptr = bias.flat<float>().data();
  float* out_ptr = output->flat<float>().data();

  const int64 lhs_batch_stride = m * k;
  const int64 rhs_batch_stride = rhs_is_batched ? k * n : 0;
  const int64 out_batch_stride = m * n;

  BiasLayout layout;
  BuildBiasLayout(bias, &layout);

  for (int64 b = 0; b < batch; ++b) {
    const float* lhs_batch = lhs_ptr + b * lhs_batch_stride;
    const float* rhs_batch = rhs_ptr + b * rhs_batch_stride;
    float* out_batch = out_ptr + b * out_batch_stride;

    for (int64 i = 0; i < m; ++i) {
      for (int64 j = 0; j < n; ++j) {
        const int64 bias_offset =
            b * layout.strides[0] + i * layout.strides[1] + j * layout.strides[2];
        float acc = bias_ptr[bias_offset];
        for (int64 kk = 0; kk < k; ++kk) {
          acc += lhs_batch[i * k + kk] * rhs_batch[kk * n + j];
        }
        out_batch[i * n + j] = Sigmoid(acc);
      }
    }
  }
}

class KPFusedBatchMatMulAddSigmoidTest : public OpsTestBase {
 protected:
  void MakeOp() {
    TF_ASSERT_OK(NodeDefBuilder("kp_fused_batchmatmul_add_sigmoid",
                                "KPFusedBatchMatMulAddSigmoid")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  Status RunWithInputs(const TensorShape& lhs_shape,
                       const std::vector<float>& lhs_values,
                       const TensorShape& rhs_shape,
                       const std::vector<float>& rhs_values,
                       const TensorShape& bias_shape,
                       const std::vector<float>& bias_values) {
    MakeOp();
    AddInputFromArray<float>(lhs_shape, lhs_values);
    AddInputFromArray<float>(rhs_shape, rhs_values);
    AddInputFromArray<float>(bias_shape, bias_values);
    return RunOpKernel();
  }

  void ExpectMatchesReference(const TensorShape& lhs_shape,
                              const std::vector<float>& lhs_values,
                              const TensorShape& rhs_shape,
                              const std::vector<float>& rhs_values,
                              const TensorShape& bias_shape,
                              const std::vector<float>& bias_values,
                              const TensorShape& output_shape) {
    TF_ASSERT_OK(RunWithInputs(lhs_shape, lhs_values, rhs_shape, rhs_values,
                               bias_shape, bias_values));

    Tensor lhs(allocator(), DT_FLOAT, lhs_shape);
    Tensor rhs(allocator(), DT_FLOAT, rhs_shape);
    Tensor bias(allocator(), DT_FLOAT, bias_shape);
    Tensor expected(allocator(), DT_FLOAT, output_shape);
    test::FillValues<float>(&lhs, lhs_values);
    test::FillValues<float>(&rhs, rhs_values);
    test::FillValues<float>(&bias, bias_values);
    ComputeReferenceBatchMatMulAddSigmoid(lhs, rhs, bias, &expected);
    test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
  }
};

TEST_F(KPFusedBatchMatMulAddSigmoidTest, FastPathN16BiasVector) {
  const TensorShape lhs_shape({2, 32, 64});
  const TensorShape rhs_shape({64, 16});
  const TensorShape bias_shape({16});
  ExpectMatchesReference(lhs_shape, MakeRange(2 * 32 * 64, -0.2f, 0.001f),
                         rhs_shape, MakeRange(64 * 16, 0.15f, 0.002f),
                         bias_shape, MakeRange(16, -0.3f, 0.05f),
                         TensorShape({2, 32, 16}));
}

TEST_F(KPFusedBatchMatMulAddSigmoidTest, FastPathN12BiasVector) {
  const TensorShape lhs_shape({2, 32, 12});
  const TensorShape rhs_shape({12, 12});
  const TensorShape bias_shape({12});
  ExpectMatchesReference(lhs_shape, MakeRange(2 * 32 * 12, -0.1f, 0.003f),
                         rhs_shape, MakeRange(12 * 12, 0.05f, 0.004f),
                         bias_shape, MakeRange(12, 0.2f, -0.03f),
                         TensorShape({2, 32, 12}));
}

TEST_F(KPFusedBatchMatMulAddSigmoidTest, FastPathN8BiasVector) {
  const TensorShape lhs_shape({2, 32, 16});
  const TensorShape rhs_shape({16, 8});
  const TensorShape bias_shape({8});
  ExpectMatchesReference(lhs_shape, MakeRange(2 * 32 * 16, -0.25f, 0.01f),
                         rhs_shape, MakeRange(16 * 8, 0.1f, 0.02f), bias_shape,
                         MakeRange(8, -0.4f, 0.1f), TensorShape({2, 32, 8}));
}

TEST_F(KPFusedBatchMatMulAddSigmoidTest, FastPathN4BiasVector) {
  const TensorShape lhs_shape({1, 32, 8});
  const TensorShape rhs_shape({8, 4});
  const TensorShape bias_shape({4});
  ExpectMatchesReference(lhs_shape, MakeRange(1 * 32 * 8, -0.2f, 0.015f),
                         rhs_shape, MakeRange(8 * 4, -0.1f, 0.03f), bias_shape,
                         MakeRange(4, 0.05f, -0.02f), TensorShape({1, 32, 4}));
}

TEST_F(KPFusedBatchMatMulAddSigmoidTest, GenericFallbackWithOddOutputWidth) {
  const TensorShape lhs_shape({2, 5, 7});
  const TensorShape rhs_shape({7, 3});
  const TensorShape bias_shape({3});
  ExpectMatchesReference(lhs_shape, MakeRange(2 * 5 * 7, -0.3f, 0.02f),
                         rhs_shape, MakeRange(7 * 3, 0.2f, -0.015f), bias_shape,
                         MakeRange(3, -0.25f, 0.2f), TensorShape({2, 5, 3}));
}

TEST_F(KPFusedBatchMatMulAddSigmoidTest, GenericFallbackWithBatchedRhs) {
  const TensorShape lhs_shape({2, 3, 4});
  const TensorShape rhs_shape({2, 4, 2});
  const TensorShape bias_shape({2, 1, 2});
  ExpectMatchesReference(lhs_shape, MakeRange(2 * 3 * 4, -0.2f, 0.05f),
                         rhs_shape, MakeRange(2 * 4 * 2, 0.15f, -0.04f),
                         bias_shape, MakeRange(2 * 1 * 2, 0.1f, 0.2f),
                         TensorShape({2, 3, 2}));
}

TEST_F(KPFusedBatchMatMulAddSigmoidTest, GenericFallbackWithLargeM) {
  const TensorShape lhs_shape({1, 33, 4});
  const TensorShape rhs_shape({4, 4});
  const TensorShape bias_shape({1, 4});
  ExpectMatchesReference(lhs_shape, MakeRange(1 * 33 * 4, -0.15f, 0.01f),
                         rhs_shape, MakeRange(4 * 4, 0.2f, -0.03f), bias_shape,
                         MakeRange(4, -0.2f, 0.07f), TensorShape({1, 33, 4}));
}

TEST_F(KPFusedBatchMatMulAddSigmoidTest, RejectsNonRank3Lhs) {
  Status s = RunWithInputs(TensorShape({4, 8}), MakeRange(32, 0.0f, 0.1f),
                           TensorShape({8, 4}), MakeRange(32, 0.1f, 0.05f),
                           TensorShape({4}), MakeRange(4, 0.0f, 0.1f));
  EXPECT_TRUE(absl::StrContains(s.ToString(), "lhs must be rank 3")) << s;
}

TEST_F(KPFusedBatchMatMulAddSigmoidTest, RejectsInvalidRhsRank) {
  Status s = RunWithInputs(TensorShape({1, 4, 8}), MakeRange(32, 0.0f, 0.1f),
                           TensorShape({1, 2, 4, 4}),
                           MakeRange(32, 0.1f, 0.05f), TensorShape({4}),
                           MakeRange(4, 0.0f, 0.1f));
  EXPECT_TRUE(absl::StrContains(s.ToString(), "rhs must be rank 2 or 3")) << s;
}

TEST_F(KPFusedBatchMatMulAddSigmoidTest, RejectsBiasRankGreaterThanThree) {
  Status s = RunWithInputs(TensorShape({1, 4, 8}), MakeRange(32, 0.0f, 0.1f),
                           TensorShape({8, 4}), MakeRange(32, 0.1f, 0.05f),
                           TensorShape({1, 1, 1, 4}),
                           MakeRange(4, 0.0f, 0.1f));
  EXPECT_TRUE(absl::StrContains(s.ToString(), "bias rank must be <= 3")) << s;
}

TEST_F(KPFusedBatchMatMulAddSigmoidTest, RejectsBatchedRhsWithBatchMismatch) {
  Status s = RunWithInputs(TensorShape({2, 3, 4}), MakeRange(24, 0.0f, 0.1f),
                           TensorShape({3, 4, 2}), MakeRange(24, 0.2f, 0.03f),
                           TensorShape({2}), MakeRange(2, 0.0f, 0.1f));
  EXPECT_TRUE(absl::StrContains(s.ToString(), "lhs/rhs batch mismatch")) << s;
}

TEST_F(KPFusedBatchMatMulAddSigmoidTest, RejectsContractingDimMismatch) {
  Status s = RunWithInputs(TensorShape({1, 3, 5}), MakeRange(15, 0.0f, 0.1f),
                           TensorShape({4, 2}), MakeRange(8, 0.2f, 0.03f),
                           TensorShape({2}), MakeRange(2, 0.0f, 0.1f));
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "lhs/rhs contracting dim mismatch"))
      << s;
}

TEST_F(KPFusedBatchMatMulAddSigmoidTest, RejectsBiasBroadcastMismatch) {
  Status s = RunWithInputs(TensorShape({2, 3, 4}), MakeRange(24, 0.0f, 0.1f),
                           TensorShape({4, 2}), MakeRange(8, 0.2f, 0.03f),
                           TensorShape({2, 2, 2}),
                           MakeRange(8, -0.1f, 0.05f));
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "bias shape is not broadcast-compatible"))
      << s;
}

}  // namespace
}  // namespace tensorflow
