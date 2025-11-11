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

class KPFusedGatherTest : public OpsTestBase {
  protected:
    void RunValidCase(const TensorShape& data_shape,
                      const TensorShape& slice_shape,
                      const std::vector<int32>& begin_val,
                      const std::vector<int64>& slice_data,
                      const std::vector<float>& data_data,
                      const std::vector<int64>& expected_unique,
                      const std::vector<int32>& expected_indices,
                      const std::vector<float>& expected_output_data) {
      TF_EXPECT_OK(NodeDefBuilder("kp_fused_gather", "KPFusedGather")
                       .Input(FakeInput(DT_FLOAT))
                       .Input(FakeInput(DT_INT64))
                       .Input(FakeInput(DT_INT32))
                       .Finalize(node_def()));
      TF_EXPECT_OK(InitOp());

      AddInputFromArray<float>(data_shape, data_data);
      AddInputFromArray<int64>(slice_shape, slice_data);
      AddInputFromArray<int32>(TensorShape({2}), begin_val);

      TF_ASSERT_OK(RunOpKernel());
      
      const Tensor& out_unique = *GetOutput(0);
      const Tensor& out_indices = *GetOutput(1);
      const Tensor& out_data = *GetOutput(2);

      // 验证输出0: unique_values
      Tensor expected_unique_tensor(
        allocator(), DT_INT64,
        TensorShape({static_cast<int64>(expected_unique.size())})
      );
      FillValues<int64>(&expected_unique_tensor, expected_unique);
      ExpectTensorEqual<int64>(expected_unique_tensor, out_unique);

      // 验证输出1: indices
      Tensor expected_indices_tensor(
        allocator(), DT_INT32,
        TensorShape({static_cast<int64_t>(expected_indices.size())})
      );
      FillValues<int32>(&expected_indices_tensor, expected_indices);
      ExpectTensorEqual<int32>(expected_indices_tensor, out_indices); 

      // 验证输出2: out_data
      Tensor expected_data_tensor(allocator(), DT_FLOAT,
                                TensorShape({static_cast<int64>(expected_unique.size()), 12}));
      FillValues<float>(&expected_data_tensor, expected_output_data);
      ExpectClose(expected_data_tensor, out_data);  // float 用 ExpectClose
    }

    Status RunOpExpectFailure(const TensorShape& data_shape,
                              const TensorShape& slice_shape,
                              const std::vector<int32>& begin_val,
                              const std::vector<int64>& slice_data,
                              const std::vector<float>& data_data) {
      TF_CHECK_OK(NodeDefBuilder("kp_fused_gather", "KPFusedGather")
                      .Input(FakeInput(DT_FLOAT))
                      .Input(FakeInput(DT_INT64))
                      .Input(FakeInput(DT_INT32))
                      .Finalize(node_def()));
      TF_CHECK_OK(InitOp());

      AddInputFromArray<float>(data_shape, data_data);
      AddInputFromArray<int64>(slice_shape, slice_data);
      AddInputFromArray<int32>(TensorShape({2}), begin_val);
      
      return RunOpKernel();
    }
};

// 正向测试：正常输入
TEST_F(KPFusedGatherTest, Valid_NormalInput) {
  RunValidCase(
      TensorShape({2, 12}),               // data shape
      TensorShape({4, 3}),                // slice_input shape
      {0, 1},                             // begin[1] = 1 → 取第1列
      {1, 1, 3,
       0, 1, 5,
       1, 0, 7,
       0, 1, 9},                          // slice_input 数据
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
       13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f},
      {1, 0},                             // unique values from col=1
      {0, 0, 1, 0},                       // indices mapping
      {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,   // data[1]
       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}            // data[0]
  );
}

// data不是2维
TEST_F(KPFusedGatherTest, Invalid_DataDimsNot2) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  Status s = RunOpExpectFailure(
      TensorShape({4}),         // data 不是二维
      TensorShape({2, 2}),
      {0, 0},
      {0, 1, 2, 3},
      data
  );
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "Embedding table shape must be 2D"));
}

// key 不是2维
TEST_F(KPFusedGatherTest, Invalid_SliceInputDimsNot2) {
  std::vector<float> data(2 * 12, 1.0f);
  Status s = RunOpExpectFailure(
      TensorShape({2, 12}),
      TensorShape({4}),         // 1D slice_input
      {0, 0},
      {0, 1, 2, 3},
      data
  );
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "Input key must be 2D"));
}

// begin[1] 超出列范围
TEST_F(KPFusedGatherTest, Invalid_BeginColOutOfRange) {
  std::vector<float> data(2 * 12, 1.0f);
  Status s = RunOpExpectFailure(
      TensorShape({2, 12}),
      TensorShape({2, 2}),
      {0, 2},                      // begin[1] = 2，但只有 2 列 → 索引 0,1
      {0, 1, 2, 3},
      data
  );
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(),"slice cols out of keys range"));
}

// gather 索引超出 data 行数
TEST_F(KPFusedGatherTest, Invalid_IndexOutOfRangeInData) {
  std::vector<float> data(2 * 12, 1.0f);
  Status s = RunOpExpectFailure(
      TensorShape({2, 12}),
      TensorShape({2, 2}),
      {0, 0},
      {0, 1,
       2, 3},  // 索引 2 超出 data 行数（只有 0,1）
      data
  );
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(),"idx out of table range"));
}

}