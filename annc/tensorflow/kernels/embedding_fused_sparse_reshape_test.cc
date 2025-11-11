/* Copyright 2025 The Huawei Technologies Co. Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace {
using tensorflow::AllocatorAttributes;
using tensorflow::DT_FLOAT;
using tensorflow::DT_INT32;
using tensorflow::DT_INT64;
using tensorflow::int64;
using tensorflow::int32;
using tensorflow::NodeDefBuilder;
using tensorflow::OpsTestBase;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::test::FillValues;
using tensorflow::test::ExpectTensorEqual;

class KPFusedSparseReshapeTest : public OpsTestBase {
 protected:
  void RunValidCase(const TensorShape& slice_shape,
                    const std::vector<int64>& slice_data,
                    const std::vector<int32>& begin_val,
                    const std::vector<int64>& new_shape_val,
                    const std::vector<int64>& pack_const_val,
                    const TensorShape& expected_indices_shape,
                    const std::vector<int64>& expected_shape_val) {
    TF_EXPECT_OK(NodeDefBuilder("kp_fused_sparse_reshape", "KPFusedSparseReshape")
                     .Input(FakeInput(DT_INT64))   // slice_input
                     .Input(FakeInput(DT_INT32))   // begin
                     .Input(FakeInput(DT_INT64))   // new_shape
                     .Input(FakeInput(DT_INT64))   // pack_const
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());

    AddInputFromArray<int64>(slice_shape, slice_data);
    AddInputFromArray<int32>(TensorShape({2}), begin_val);
    AddInputFromArray<int64>(TensorShape({2}), new_shape_val);
    AddInputFromArray<int64>(TensorShape({}), pack_const_val);

    TF_ASSERT_OK(RunOpKernel());

    // 输出0: result_indices
    const Tensor& out_indices = *GetOutput(0);
    EXPECT_EQ(out_indices.shape(), expected_indices_shape);

    // 输出1: result_shape
    const Tensor& out_shape = *GetOutput(1);
    Tensor expected_shape_tensor(DT_INT64,
                                 TensorShape({static_cast<int64>(expected_shape_val.size())}));
    FillValues<int64>(&expected_shape_tensor, expected_shape_val);
    ExpectTensorEqual<int64>(expected_shape_tensor, out_shape);
  }

  Status RunOpExpectFailure(const TensorShape& slice_shape,
                            const std::vector<int64>& slice_data,
                            const std::vector<int32>& begin_val,
                            const std::vector<int64>& new_shape_val,
                            const std::vector<int64>& pack_const_val) {
    TF_CHECK_OK(NodeDefBuilder("kp_fused_sparse_reshape", "KPFusedSparseReshape")
                    .Input(FakeInput(DT_INT64))   // slice_input
                    .Input(FakeInput(DT_INT32))   // begin
                    .Input(FakeInput(DT_INT64))   // new_shape
                    .Input(FakeInput(DT_INT64))   // pack_const
                    .Finalize(node_def()));
    TF_CHECK_OK(InitOp());

    AddInputFromArray<int64>(slice_shape, slice_data);
    AddInputFromArray<int32>(TensorShape({static_cast<int64>(begin_val.size())}), begin_val);
    AddInputFromArray<int64>(TensorShape({static_cast<int64>(new_shape_val.size())}), new_shape_val);
    AddInputFromArray<int64>(TensorShape({}), pack_const_val);

    return RunOpKernel();
  }
};

// ==================== 正向测试 ====================

// 正常 reshape 案例
// pack_const=2
TEST_F(KPFusedSparseReshapeTest, Valid_NormalInput) {
  RunValidCase(
      TensorShape({4, 2}),   // slice_input shape
      {0, 1,
       1, 2,
       2, 3,
       3, 0},                // slice_input 数据
      {0, 1},                // begin = (0,1)，选第1列
      {2, 4},                // new_shape = [2,4]
      {2},                   // pack_const = [2]
      TensorShape({4, 2}),   // 预期 indices 形状
      {2, 4});               // 预期 shape
}

// pack_const = 1
TEST_F(KPFusedSparseReshapeTest, Valid_PackConst1) {
  RunValidCase(
      TensorShape({1, 2}),   // slice_input shape
      {0, 1},                // slice_input 数据
      {0, 1},                // begin = (0,1)，选第1列
      {-1, 1},               // new_shape = [-1,1]
      {1},                   // pack_const = [1]
      TensorShape({1, 2}),   // 预期 indices 形状
      {1, 1});               // 预期 shape
}

// ==================== 反向测试 ====================

// 反例1：slice_input 不是二维
TEST_F(KPFusedSparseReshapeTest, Invalid_SliceInputNot2D) {
  Status s = RunOpExpectFailure(
      TensorShape({4}), {0, 1, 2, 3},
      {0, 0},
      {2, 2},
      {4});
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "slice_input dims must == 2"));
}

// 反例2：new_shape dim size 不是 2
TEST_F(KPFusedSparseReshapeTest, Invalid_NewShapeNotLen2) {
  Status s = RunOpExpectFailure(
      TensorShape({2, 2}), {0, 1, 1, 0},
      {0, 0},
      {4, 2, 1},   // new_shape 多了1个元素
      {2});
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "new_shape dim size must == 2"));
}

// 反例3：begin[1] 超出 slice_input 列数
TEST_F(KPFusedSparseReshapeTest, Invalid_BeginOutOfRange) {
  Status s = RunOpExpectFailure(
      TensorShape({2, 2}), {0, 1, 1, 0},
      {0, 2},     // 超过列数
      {2, 2},
      {2});
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "begin[1] must < slice_input.dim_size(1)"));
}

// 反例4：target shape 有多个 -1
TEST_F(KPFusedSparseReshapeTest, Invalid_MultipleUnknownDims) {
  Status s = RunOpExpectFailure(
      TensorShape({2, 2}), {0, 1, 1, 0},
      {0, 1},
      {-1, -1},   // 两个 -1
      {2});
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "only one output dimension may be -1"));
}

// 反例5：reshape 推断维度时，总元素数不能整除，导致无法匹配 --> product * missing != dense_size
TEST_F(KPFusedSparseReshapeTest, Invalid_InferredShapeDoesNotMatch) {
  TensorShape input_indices_shape({6, 2});  // 6 个非零元素，rank=2
  std::vector<int64> input_indices_data = {
    0, 0,
    0, 1,
    0, 2,
    1, 0,
    1, 1,
    1, 2
  };  // 对应 2x3 的 dense tensor

  std::vector<int32> begin_val = {0, 0};         // 假设的 begin 输入
  std::vector<int64> new_shape_val = {-1, 4};    // reshape 到 ?x4
  std::vector<int64> pack_const_val = {1};

  Status s = RunOpExpectFailure(
      input_indices_shape,
      input_indices_data,
      begin_val,
      new_shape_val,
      pack_const_val);

  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "Input to reshape is a SparseTensor with"));
}

// 反例6：reshape 后元素数量不匹配 --> output_shape.num_elements() != dense_size
TEST_F(KPFusedSparseReshapeTest, Invalid_SizeMismatch) {
  Status s = RunOpExpectFailure(
      TensorShape({2, 2}), {0, 1, 1, 0},
      {0, 1},
      {3, 3},   // 期望 9 元素，但输入 dense size = 4
      {2});
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "Input to reshape is a tensor with"));
}

// 反例7：target_shape 包含负数但不是 -1
TEST_F(KPFusedSparseReshapeTest, Invalid_NegativeDimNotMinusOne) {
  Status s = RunOpExpectFailure(
      TensorShape({2, 2}), {0, 1, 1, 0},
      {0, 0},
      {2, -2},   // -2 是非法的
      {2});
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "size 1 must be non-negative, not -2"))
      << "Actual error: " << s.message();
}

// 反例8：target_shape 有 -1，但其他维度乘积为 0
TEST_F(KPFusedSparseReshapeTest, Invalid_ProductZeroWithUnknownDim) {
  // dense_size = 0（空 SparseTensor），target_shape = [-1, 0]
  // product = 0 → 不允许 infer
  Status s = RunOpExpectFailure(
      TensorShape({0, 2}), {},           // 空的 slice_input
      {0, 0},
      {-1, 0},   // product = 0
      {2});
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "reshape cannot infer the missing input size for an empty tensor"))
      << "Actual error: " << s.message();
}

// 反例9：begin 是 1D 但长度为 1（不够 2 个元素）
TEST_F(KPFusedSparseReshapeTest, Invalid_BeginRank1ButSize1) {
  Status s = RunOpExpectFailure(
      TensorShape({2, 2}), {0, 1, 1, 0},
      {0},                // begin = [0]，长度为 1
      {2, 2},
      {2});
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "begin must be 1D with at least 2 elements"))
      << "Actual error: " << s.message();
}

// 反例10：begin 是 1D 但长度为 3（超过 2）
TEST_F(KPFusedSparseReshapeTest, Invalid_BeginRank1ButSize3) {
  Status s = RunOpExpectFailure(
      TensorShape({2, 2}), {0, 1, 1, 0},
      {0, 1, 2},          // begin = [0,1,2]，长度为 3
      {2, 2},
      {2});
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "begin must be 1D with at least 2 elements"))
      << "Actual error: " << s.message();
}

// 反例11：pack_const 是标量（0维）
TEST_F(KPFusedSparseReshapeTest, Invalid_PackConstIsScalarButExpect1D) {
  TF_CHECK_OK(NodeDefBuilder("kp_fused_sparse_reshape", "KPFusedSparseReshape")
                  .Input(FakeInput(DT_INT64))   // slice_input
                  .Input(FakeInput(DT_INT32))   // begin
                  .Input(FakeInput(DT_INT64))   // new_shape
                  .Input(FakeInput(DT_INT64))   // pack_const
                  .Finalize(node_def()));
  TF_CHECK_OK(InitOp());

  AddInputFromArray<int64>(TensorShape({2, 2}), {0, 1, 1, 0});
  AddInputFromArray<int32>(TensorShape({2}), {0, 1});
  AddInputFromArray<int64>(TensorShape({2}), {2, 2});
  AddInputFromArray<int64>(TensorShape({1}), {1});  // pack_const = 标量 1（0维）

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "pack_const must be a scalar"))
      << "Actual error: " << s.message();
}

}  // namespace
