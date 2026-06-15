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

#include <functional>
#include <memory>
#include <vector>

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
namespace {

class KPFusedSparseSegmentReduceNonzeroOpTest : public OpsTestBase {
 protected:
  void MakeOp(int combiner_mode) {
    TF_ASSERT_OK(NodeDefBuilder("kp_fused_sparse_segment_reduce_nonzero",
                                "KPFusedSparseSegmentReduceNonzero")
                     .Input(FakeInput(DT_FLOAT))  // data
                     .Input(FakeInput(DT_INT32))  // indices
                     .Input(FakeInput(DT_INT64))  // slice_input
                     .Input(FakeInput(DT_INT32))  // begin
                     .Attr("combiner", combiner_mode)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(KPFusedSparseSegmentReduceNonzeroOpTest, TestReduceMean) {
  MakeOp(1);

  AddInputFromArray<float>(TensorShape({8}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  AddInputFromArray<int32>(TensorShape({3}), {0, 2, 1});
  AddInputFromArray<int64>(TensorShape({3, 4}),
                           {1, 2, 2, 2, 1, 1, 2, 3, 2, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int32>(&expected, {4});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));  // output_shape

  Tensor expected_1(allocator(), DT_INT32, TensorShape({2, 1}));
  test::FillValues<int32>(&expected_1, {2, 3});
  test::ExpectTensorEqual<int32>(expected_1, *GetOutput(1));  // output_indices

  Tensor expected_2(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_2, {2, 2});
  test::ExpectTensorEqual<float>(expected_2, *GetOutput(2));  // output_nonzero
}

TEST_F(KPFusedSparseSegmentReduceNonzeroOpTest, TestReduceSum) {
  MakeOp(0);

  AddInputFromArray<float>(TensorShape({8}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  AddInputFromArray<int32>(TensorShape({3}), {0, 2, 1});
  AddInputFromArray<int64>(TensorShape({3, 4}),
                           {1, 2, 2, 2, 1, 1, 2, 3, 2, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int32>(&expected, {4});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));  // output_shape

  Tensor expected_1(allocator(), DT_INT32, TensorShape({2, 1}));
  test::FillValues<int32>(&expected_1, {2, 3});
  test::ExpectTensorEqual<int32>(expected_1, *GetOutput(1));  // output_indices

  Tensor expected_2(allocator(), DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&expected_2, {4, 2});
  test::ExpectTensorEqual<float>(expected_2, *GetOutput(2));  // output_nonzero
}

TEST_F(KPFusedSparseSegmentReduceNonzeroOpTest, Test2DReduceSum) {
  MakeOp(0);

  // data: [5, 3] 二维矩阵，每行是一个 embedding 向量
  // data[0]=[1,2,3], data[1]=[4,0,6], data[2]=[7,8,9],
  // data[3]=[0,0,0](全零), data[4]=[1,1,1]
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {1.0f, 2.0f, 3.0f,   // row 0
                            4.0f, 0.0f, 6.0f,   // row 1: dim1=0 测试元素级过滤
                            7.0f, 8.0f, 9.0f,   // row 2
                            0.0f, 0.0f, 0.0f,   // row 3: 全零
                            1.0f, 1.0f, 1.0f}); // row 4
  // indices: 每条记录从data中取哪行
  AddInputFromArray<int32>(TensorShape({5}), {0, 2, 1, 4, 3});
  // slice_input: [5, 4], 取col=2作为segment_ids
  // col=2: segment_ids = [2, 2, 3, 3, 4]
  AddInputFromArray<int64>(TensorShape({5, 4}),
                           {1, 2, 2, 2,   // row 0: seg_id=2
                            1, 1, 2, 3,   // row 1: seg_id=2
                            2, 2, 3, 4,   // row 2: seg_id=3
                            1, 2, 3, 3,   // row 3: seg_id=3
                            2, 2, 4, 4}); // row 4: seg_id=4 (全零segment)
  AddInputFromArray<int32>(TensorShape({2}), {0, 2});
  TF_ASSERT_OK(RunOpKernel());

  // output_shape: [batch_size, embed_dim] = [5, 3]
  // batch_size = max_seg_id + 1 = 4 + 1 = 5
  Tensor expected(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int32>(&expected, {5, 3});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));

  // seg_id=2: data[0]+data[2] = [1,2,3]+[7,8,9] = [8, 10, 12] 全非零 -> 3个坐标
  // seg_id=3: data[1]+data[4] = [4,0,6]+[1,1,1] = [5,  1,  7]
  //   dim=0->5(非零), dim=1->1(非零), dim=2->7(非零) -> 3个坐标
  // seg_id=4: data[3]         = [0,0,0]           全零  -> 0个坐标
  // 共6个非零元素，output_indices: [6, 2]
  Tensor expected_1(allocator(), DT_INT32, TensorShape({6, 2}));
  test::FillValues<int32>(&expected_1, {2, 0,
                                        2, 1,
                                        2, 2,
                                        3, 0,
                                        3, 1,
                                        3, 2});
  test::ExpectTensorEqual<int32>(expected_1, *GetOutput(1));

  Tensor expected_2(allocator(), DT_FLOAT, TensorShape({6}));
  test::FillValues<float>(&expected_2, {8.0f, 10.0f, 12.0f, 5.0f, 1.0f, 7.0f});
  test::ExpectTensorEqual<float>(expected_2, *GetOutput(2));
}

TEST_F(KPFusedSparseSegmentReduceNonzeroOpTest, Test2DReduceMean) {
  MakeOp(1);

  // 与 Test2DReduceSum 使用完全相同的输入
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {1.0f, 2.0f, 3.0f,
                            4.0f, 0.0f, 6.0f,
                            7.0f, 8.0f, 9.0f,
                            0.0f, 0.0f, 0.0f,
                            1.0f, 1.0f, 1.0f});
  AddInputFromArray<int32>(TensorShape({5}), {0, 2, 1, 4, 3});
  AddInputFromArray<int64>(TensorShape({5, 4}),
                           {1, 2, 2, 2,
                            1, 1, 2, 3,
                            2, 2, 3, 4,
                            1, 2, 3, 3,
                            2, 2, 4, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 2});
  TF_ASSERT_OK(RunOpKernel());

  // output_shape: [5, 3]
  Tensor expected(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int32>(&expected, {5, 3});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));

  // seg_id=2: mean([1,2,3],[7,8,9]) = [4, 5, 6]  全非零 -> 3个坐标
  // seg_id=3: mean([4,0,6],[1,1,1]) = [2.5, 0.5, 3.5]
  //   dim=0->2.5(非零), dim=1->0.5(非零), dim=2->3.5(非零) -> 3个坐标
  // seg_id=4: mean([0,0,0])         = [0, 0, 0]  全零 -> 0个坐标
  // 共6个非零元素
  Tensor expected_1(allocator(), DT_INT32, TensorShape({6, 2}));
  test::FillValues<int32>(&expected_1, {2, 0,
                                        2, 1,
                                        2, 2,
                                        3, 0,
                                        3, 1,
                                        3, 2});
  test::ExpectTensorEqual<int32>(expected_1, *GetOutput(1));

  Tensor expected_2(allocator(), DT_FLOAT, TensorShape({6}));
  test::FillValues<float>(&expected_2, {4.0f, 5.0f, 6.0f, 2.5f, 0.5f, 3.5f});
  test::ExpectTensorEqual<float>(expected_2, *GetOutput(2));
}

TEST_F(KPFusedSparseSegmentReduceNonzeroOpTest, Test2DElementWiseZeroFilter) {
  MakeOp(0);

  // 专门测试二维情况下的元素级非零过滤：
  // 某个 segment 的归约向量中只有部分维度为零，应只过滤零的维度
  // data[0]=[1,0,3], data[1]=[0,2,0] -> 两行相加=[1,2,3]
  // data[2]=[3,0,0], data[3]=[0,0,4] -> 两行相加=[3,0,4] 中间维度为0被过滤
  AddInputFromArray<float>(TensorShape({4, 3}),
                           {1.0f, 0.0f, 3.0f,   // row 0
                            0.0f, 2.0f, 0.0f,   // row 1
                            3.0f, 0.0f, 0.0f,   // row 2
                            0.0f, 0.0f, 4.0f}); // row 3
  AddInputFromArray<int32>(TensorShape({4}), {0, 1, 2, 3});
  // segment_ids (col=1): [0, 0, 1, 1]
  AddInputFromArray<int64>(TensorShape({4, 2}),
                           {9, 0,   // seg_id=0
                            9, 0,   // seg_id=0
                            9, 1,   // seg_id=1
                            9, 1}); // seg_id=1
  AddInputFromArray<int32>(TensorShape({2}), {0, 1});
  TF_ASSERT_OK(RunOpKernel());

  // output_shape: [2, 3]
  Tensor expected(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int32>(&expected, {2, 3});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));

  // seg_id=0: [1,0,3]+[0,2,0]=[1,2,3] -> [0,0]→1, [0,2]→3  (dim=1为0被过滤)
  // seg_id=1: [3,0,0]+[0,0,4]=[3,0,4] -> [1,0]→3, [1,2]→4  (dim=1为0被过滤)
  // 共4个非零元素
  Tensor expected_1(allocator(), DT_INT32, TensorShape({5, 2}));
  test::FillValues<int32>(&expected_1, {0, 0,
                                        0, 1,
                                        0, 2,
                                        1, 0,
                                        1, 2});
  test::ExpectTensorEqual<int32>(expected_1, *GetOutput(1));

  Tensor expected_2(allocator(), DT_FLOAT, TensorShape({5}));
  test::FillValues<float>(&expected_2, {1.0f, 2.0f, 3.0f, 3.0f, 4.0f});
  test::ExpectTensorEqual<float>(expected_2, *GetOutput(2));
}

TEST_F(KPFusedSparseSegmentReduceNonzeroOpTest, TestInvalidData) {
  MakeOp(0);

  // 使用3维data，应触发维度检查错误（只支持1-D 和 2-D）
  AddInputFromArray<float>(TensorShape({2, 2, 2}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  AddInputFromArray<int32>(TensorShape({3}), {0, 2, 1});
  AddInputFromArray<int64>(TensorShape({3, 4}),
                           {1, 2, 2, 2, 1, 1, 2, 3, 2, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 2});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "Input data must be a 1-D vector or 2-D matrix") !=
              std::string::npos);
}

TEST_F(KPFusedSparseSegmentReduceNonzeroOpTest, TestInvalidSliceinput) {
  MakeOp(0);

  AddInputFromArray<float>(TensorShape({8}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  AddInputFromArray<int32>(TensorShape({3}), {0, 2, 1});
  AddInputFromArray<int64>(TensorShape({3, 4, 1}),
                           {1, 2, 2, 2, 1, 1, 2, 3, 2, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 2});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "slice input must be 2-D") !=
              std::string::npos);
}

TEST_F(KPFusedSparseSegmentReduceNonzeroOpTest, TestInvalidbegin) {
  MakeOp(0);

  AddInputFromArray<float>(TensorShape({8}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  AddInputFromArray<int32>(TensorShape({3}), {0, 2, 1});
  AddInputFromArray<int64>(TensorShape({3, 4}),
                           {1, 2, 2, 2, 1, 1, 2, 3, 2, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({3}), {0, 2, 1});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "begin must have 2 elements"));
}

TEST_F(KPFusedSparseSegmentReduceNonzeroOpTest, TestColsOutOfBounds) {
  MakeOp(0);

  AddInputFromArray<float>(TensorShape({8}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  AddInputFromArray<int32>(TensorShape({3}), {0, 2, 1});
  AddInputFromArray<int64>(TensorShape({3, 4}),
                           {1, 2, 2, 2, 1, 1, 2, 3, 2, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 4});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "Column index out of range"));
}

TEST_F(KPFusedSparseSegmentReduceNonzeroOpTest, TestIndicesOutOfBounds) {
  MakeOp(0);

  AddInputFromArray<float>(TensorShape({8}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  AddInputFromArray<int32>(TensorShape({2}), {0, 2});
  AddInputFromArray<int64>(TensorShape({3, 4}),
                           {1, 2, 2, 2, 1, 1, 2, 3, 2, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 1});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), 
                  "indices and slice_input.dim_size(0) should have same size"));
}

}  // namespace
}  // namespace tensorflow
