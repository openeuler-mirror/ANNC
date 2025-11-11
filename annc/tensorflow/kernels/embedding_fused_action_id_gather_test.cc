/* Copyright 2025 The Huawei Technologies Co. Authors. All Rights Reserved.
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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

class KPFusedEmbeddingActionIdGatherTest : public OpsTestBase {
 protected:
  void MakeOp(DataType indices1_type, DataType indices2_type) {
    TF_ASSERT_OK(NodeDefBuilder("fused_embedding_action_id_gather",
                                "KPFusedEmbeddingActionIdGather")
                     .Input(FakeInput(indices1_type))  // indices1
                     .Input(FakeInput(DT_FLOAT))       // params
                     .Input(FakeInput(indices2_type))  // indices2
                     .Input(FakeInput(DT_INT32))       // pack_dim
                     .Input(FakeInput(DT_INT32))       // pack
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  template <typename Tindices1, typename Tindices2>
  Status FeedAndRun(const std::vector<Tindices1>& indices1_data,
                    const TensorShape& indices1_shape,
                    const std::vector<float>& params_data,
                    const TensorShape& params_shape,
                    const std::vector<Tindices2>& indices2_data,
                    const TensorShape& indices2_shape, int pack_dim_value,
                    int pack_value) {
    inputs_.clear();
    input_types_.clear();

    MakeOp(DataTypeToEnum<Tindices1>::v(), DataTypeToEnum<Tindices2>::v());
    AddInputFromArray<Tindices1>(indices1_shape, indices1_data);
    AddInputFromArray<float>(params_shape, params_data);
    AddInputFromArray<Tindices2>(indices2_shape, indices2_data);
    AddInputFromArray<int32>(TensorShape({}), {pack_dim_value});
    AddInputFromArray<int32>(TensorShape({}), {pack_value});
    return RunOpKernel();
  }
};

TEST_F(KPFusedEmbeddingActionIdGatherTest, NormalCase) {
  std::vector<int64> indices1_data = {0, 2};
  TensorShape indices1_shape({2, 1});

  std::vector<float> params_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  TensorShape params_shape({3, 2});

  std::vector<int32> indices2_data = {1, 0};
  TensorShape indices2_shape({2, 1});

  int pack_dim_value = 2;
  int pack_value = 1;

  TF_ASSERT_OK((FeedAndRun<int64, int32>(
      indices1_data, indices1_shape, params_data, params_shape, indices2_data,
      indices2_shape, pack_dim_value, pack_value)));

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected, {5.0f, 6.0f, 0.0f, 1.0f, 2.0f, 0.0f});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

TEST_F(KPFusedEmbeddingActionIdGatherTest, DifferentIndexTypes) {
  // int64int32
  {
    std::vector<int64> indices1 = {0, 2};
    std::vector<int32> indices2 = {1, 0};
    TF_ASSERT_OK((FeedAndRun<int64, int32>(indices1, {2, 1},
                                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                                           {3, 2}, indices2, {2, 1}, 2, 1)));
    test::ExpectTensorNear<float>(
        *GetOutput(0),
        test::AsTensor<float>({5.0f, 6.0f, 0.0f, 1.0f, 2.0f, 0.0f}, {2, 3}),
        1e-5);
  }

  // int32int32
  {
    std::vector<int32> indices1 = {0, 2};
    std::vector<int32> indices2 = {1, 0};
    TF_ASSERT_OK((FeedAndRun<int32, int32>(indices1, {2, 1},
                                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                                           {3, 2}, indices2, {2, 1}, 2, 1)));
    test::ExpectTensorNear<float>(
        *GetOutput(0),
        test::AsTensor<float>({5.0f, 6.0f, 0.0f, 1.0f, 2.0f, 0.0f}, {2, 3}),
        1e-5);
  }

  // int64int64
  {
    std::vector<int64> indices1 = {0, 2};
    std::vector<int64> indices2 = {1, 0};
    TF_ASSERT_OK((FeedAndRun<int64, int64>(indices1, {2, 1},
                                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                                           {3, 2}, indices2, {2, 1}, 2, 1)));
    test::ExpectTensorNear<float>(
        *GetOutput(0),
        test::AsTensor<float>({5.0f, 6.0f, 0.0f, 1.0f, 2.0f, 0.0f}, {2, 3}),
        1e-5);
  }

  // int32int64
  {
    std::vector<int32> indices1 = {0, 2};
    std::vector<int64> indices2 = {1, 0};
    TF_ASSERT_OK((FeedAndRun<int32, int64>(indices1, {2, 1},
                                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                                           {3, 2}, indices2, {2, 1}, 2, 1)));
    test::ExpectTensorNear<float>(
        *GetOutput(0),
        test::AsTensor<float>({5.0f, 6.0f, 0.0f, 1.0f, 2.0f, 0.0f}, {2, 3}),
        1e-5);
  }
}

TEST_F(KPFusedEmbeddingActionIdGatherTest, InvalidIndices1Dims) {
  MakeOp(DT_INT64, DT_INT32);

  std::vector<int64> indices1_data = {0, 2};
  AddInputFromArray<int64>(TensorShape({2}), indices1_data);

  std::vector<float> params_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  AddInputFromArray<float>(TensorShape({3, 2}), params_data);

  std::vector<int32> indices2_data = {1, 0};
  AddInputFromArray<int32>(TensorShape({2, 1}), indices2_data);

  AddInputFromArray<int32>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {1});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.ToString(), "indices1 dims must = 2")) << s;
}

TEST_F(KPFusedEmbeddingActionIdGatherTest, InvalidIndices2Dims) {
  MakeOp(DT_INT64, DT_INT32);

  std::vector<int64> indices1_data = {0, 2};
  AddInputFromArray<int64>(TensorShape({2, 1}), indices1_data);

  std::vector<float> params_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  AddInputFromArray<float>(TensorShape({3, 2}), params_data);

  std::vector<int32> indices2_data = {1, 0};
  AddInputFromArray<int32>(TensorShape({2}), indices2_data);

  AddInputFromArray<int32>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {1});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.ToString(), "indices2 dims must = 2")) << s;
}

TEST_F(KPFusedEmbeddingActionIdGatherTest, InvalidParamsDims) {
  MakeOp(DT_INT64, DT_INT32);

  std::vector<int64> indices1_data = {0, 2};
  AddInputFromArray<int64>(TensorShape({2, 1}), indices1_data);

  std::vector<float> params_data = {1.0f, 2.0f, 3.0f, 4.0f};
  AddInputFromArray<float>(TensorShape({4}), params_data);

  std::vector<int32> indices2_data = {1, 0};
  AddInputFromArray<int32>(TensorShape({2, 1}), indices2_data);

  AddInputFromArray<int32>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {1});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.ToString(), "params dims must = 2")) << s;
}

TEST_F(KPFusedEmbeddingActionIdGatherTest, InvalidPackDimDims) {
  MakeOp(DT_INT64, DT_INT32);

  std::vector<int64> indices1_data = {0, 2};
  AddInputFromArray<int64>(TensorShape({2, 1}), indices1_data);

  std::vector<float> params_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  AddInputFromArray<float>(TensorShape({3, 2}), params_data);

  std::vector<int32> indices2_data = {1, 0};
  AddInputFromArray<int32>(TensorShape({2, 1}), indices2_data);

  AddInputFromArray<int32>(TensorShape({1}), {2});
  AddInputFromArray<int32>(TensorShape({}), {1});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.ToString(), "pack_dim is scalar")) << s;
}

TEST_F(KPFusedEmbeddingActionIdGatherTest, InvalidPackDims) {
  MakeOp(DT_INT64, DT_INT32);

  std::vector<int64> indices1_data = {0, 2};
  AddInputFromArray<int64>(TensorShape({2, 1}), indices1_data);

  std::vector<float> params_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  AddInputFromArray<float>(TensorShape({3, 2}), params_data);

  std::vector<int32> indices2_data = {1, 0};
  AddInputFromArray<int32>(TensorShape({2, 1}), indices2_data);

  AddInputFromArray<int32>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({1}), {1});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.ToString(), "pack const is scalar")) << s;
}

TEST_F(KPFusedEmbeddingActionIdGatherTest, InvalidPackSize) {
  MakeOp(DT_INT64, DT_INT32);

  std::vector<int64> indices1_data = {0, 2};
  AddInputFromArray<int64>(TensorShape({2, 1}), indices1_data);

  std::vector<float> params_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  AddInputFromArray<float>(TensorShape({3, 2}), params_data);

  std::vector<int32> indices2_data = {1, 0};
  AddInputFromArray<int32>(TensorShape({2, 1}), indices2_data);

  AddInputFromArray<int32>(TensorShape({}), {0});
  AddInputFromArray<int32>(TensorShape({}), {1});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.ToString(), "pack_size must > 0")) << s;
}

TEST_F(KPFusedEmbeddingActionIdGatherTest, IndexOutOfRange) {
  MakeOp(DT_INT64, DT_INT32);

  std::vector<int64> indices1_data = {0, 5};
  AddInputFromArray<int64>(TensorShape({2, 1}), indices1_data);

  std::vector<float> params_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  AddInputFromArray<float>(TensorShape({3, 2}), params_data);

  std::vector<int32> indices2_data = {1, 0};
  AddInputFromArray<int32>(TensorShape({2, 1}), indices2_data);

  AddInputFromArray<int32>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {1});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "GatherV2 axis=0: index out of range"))
      << s;
}

}  // namespace tensorflow
