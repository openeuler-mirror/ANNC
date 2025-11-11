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
using tensorflow::test::ExpectClose;
using tensorflow::test::FillValues;
using tensorflow::test::AsTensor;
using tensorflow::test::ExpectTensorEqual;

class KPFusedSparseSelectTest : public OpsTestBase {
 protected:
  void RunValidCase(
      const TensorShape& shape,
      const std::vector<int32>& a_data,
      const std::vector<int32>& b_data,
      const std::vector<int32>& c_data,
      int32_t greater_val,
      int32_t equal1_val,
      int32_t equal2_val,
      const std::vector<float>& expected_y,
      const std::vector<float>& expected_w_col0) {
    
    TF_EXPECT_OK(NodeDefBuilder("kp_fused_sparse_select", "KPFusedSparseSelect")
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))  // greater
                     .Input(FakeInput(DT_INT32))  // equal1
                     .Input(FakeInput(DT_INT32))  // equal2
                     .Input(FakeInput(DT_INT32))  // equal3
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());

    AddInputFromArray<int32>(shape, a_data);
    AddInputFromArray<int32>(shape, b_data);
    AddInputFromArray<int32>(shape, c_data);
    AddInputFromArray<int32>(TensorShape({}), {greater_val});  // scalar
    AddInputFromArray<int32>(TensorShape({}), {equal1_val});
    AddInputFromArray<int32>(TensorShape({}), {equal2_val});
    AddInputFromArray<int32>(TensorShape({}), {0});  // equal3_val (未使用)

    TF_ASSERT_OK(RunOpKernel());

    const Tensor& out_x = *GetOutput(0);
    const Tensor& out_y = *GetOutput(1);
    const Tensor& out_w = *GetOutput(2);

    int32 Num_elements = expected_y.size();
    // 验证 output_x: 就是 input_a
    std::vector<float> a_data_float(a_data.begin(), a_data.end());
    ExpectTensorEqual<float>(out_x, AsTensor<float>(a_data_float, {Num_elements, 1}));

    // 验证 output_y
    ExpectTensorEqual<float>(out_y, AsTensor<float>(expected_y, {Num_elements, 1}));
    // 验证 output_w 第一列
    auto w_mat = out_w.matrix<float>();
    for (int i = 0; i < w_mat.dimension(0); ++i) {
      EXPECT_FLOAT_EQ(w_mat(i, 0), expected_w_col0[i]);
      EXPECT_FLOAT_EQ(w_mat(i, 1), 1.0f);  // 第二列必须是 1.0
    }
  }

  Status RunOpExpectFailure(
      const TensorShape& shape,
      const std::vector<int32>& a_data,
      const std::vector<int32>& b_data,
      const std::vector<int32>& c_data,
      int32_t greater_val,
      int32_t equal1_val,
      int32_t equal2_val) {
    
    TF_CHECK_OK(NodeDefBuilder("kp_fused_sparse_select", "KPFusedSparseSelect")
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Finalize(node_def()));
    TF_CHECK_OK(InitOp());
    TensorShape b_shape({static_cast<int64>(b_data.size())});
    TensorShape c_shape({static_cast<int64>(c_data.size())});
    AddInputFromArray<int32>(shape, a_data);
    AddInputFromArray<int32>(b_shape, b_data);
    AddInputFromArray<int32>(c_shape, c_data);
    AddInputFromArray<int32>(TensorShape({}), {greater_val});
    AddInputFromArray<int32>(TensorShape({}), {equal1_val});
    AddInputFromArray<int32>(TensorShape({}), {equal2_val});
    AddInputFromArray<int32>(TensorShape({}), {0});

    return RunOpKernel();
  }
};

// ==================== 正向测试 ====================
// 更多正向验证参考 fused_embedding_sparse_select_test.py
TEST_F(KPFusedSparseSelectTest, Valid_NormalInput) {
  RunValidCase(
      TensorShape({3}),                   // shape
      {5, 3, 8},                          // input_a
      {1, 2, 1},                          // input_b
      {9, 8, 7},                          // input_c (未使用)
      4,                                  // greater_val
      1,                                  // equal1_val
      3,                                  // equal2_val
      {1.0f, 0.0f, 1.0f},                 // expected_y
      {1.0f, 0.0f, 1.0f}                  // expected_w_col0
  );
}

TEST_F(KPFusedSparseSelectTest, Valid_2DInput) {
  RunValidCase(
      TensorShape({2, 2}),
      {6, 3, 8, 2},
      {2, 1, 3, 4},
      {0, 0, 0, 0},
      5,
      2,
      3,
      {1.0f, 0.0f, 1.0f, 0.0f},
      {1.0f, 0.0f, 1.0f, 0.0f}
  );
}
// ==================== 反向测试 ====================
// 反例1：input_a 与 input_b 元素数不匹配
TEST_F(KPFusedSparseSelectTest, Invalid_DimMismatch_AB) {
  Status s = RunOpExpectFailure(
      TensorShape({3}),           // a 有 3 个元素
      {1, 2, 3},
      {4, 5},                     // b 有 2 个元素 → 不匹配！
      {6, 7, 8},
      0, 1, 2
  );
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "Input num elements of a and b must match"));
}

// 反例2：input_a 与 input_c 元素数不匹配
TEST_F(KPFusedSparseSelectTest, Invalid_DimMismatch_AC) {
  Status s = RunOpExpectFailure(
      TensorShape({2}),
      {1, 2},
      {3, 4},
      {5},                        // c 只有 1 个元素 → 不匹配！
      0, 1, 2
  );
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "Input num elements of a and c must match"));
}

}