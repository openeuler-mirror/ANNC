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

class KPFusedSparseSegmentReduceOpTest : public OpsTestBase {
 protected:
  void MakeOp(int combiner_mode) {
    TF_ASSERT_OK(NodeDefBuilder("kp_fused_sparse_segment_reduce",
                                "KPFusedSparseSegmentReduce")
                     .Input(FakeInput(DT_FLOAT))  // data
                     .Input(FakeInput(DT_INT32))  // indices
                     .Input(FakeInput(DT_INT64))  // slice_input
                     .Input(FakeInput(DT_INT32))  // begin
                     .Input(FakeInput(DT_INT32))  // begin_1
                     .Attr("combiner", combiner_mode)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(KPFusedSparseSegmentReduceOpTest, TestReduceMean) {
  MakeOp(1);

  AddInputFromArray<float>(TensorShape({4, 2}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  AddInputFromArray<int32>(TensorShape({3}), {0, 2, 1});
  AddInputFromArray<int64>(TensorShape({3, 4}),
                           {1, 2, 2, 2, 1, 1, 2, 3, 2, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 2});
  AddInputFromArray<int32>(TensorShape({1}), {1});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 2}));
  test::FillValues<float>(&expected,
                          {0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 4.0f, 3.0f, 4.0f});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));

  Tensor expected_1(allocator(), DT_INT32, TensorShape({}));
  test::FillValues<int32>(&expected_1, {2});
  test::ExpectTensorEqual<int32>(expected_1, *GetOutput(1));
}

TEST_F(KPFusedSparseSegmentReduceOpTest, TestReduceSum) {
  MakeOp(0);

  AddInputFromArray<float>(TensorShape({4, 2}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  AddInputFromArray<int32>(TensorShape({3}), {0, 2, 1});
  AddInputFromArray<int64>(TensorShape({3, 4}),
                           {1, 2, 2, 2, 1, 1, 2, 3, 2, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 2});
  AddInputFromArray<int32>(TensorShape({1}), {0});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 2}));
  test::FillValues<float>(&expected,
                          {0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 8.0f, 3.0f, 4.0f});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));

  Tensor expected_1(allocator(), DT_INT32, TensorShape({}));
  test::FillValues<int32>(&expected_1, {4});
  test::ExpectTensorEqual<int32>(expected_1, *GetOutput(1));
}

TEST_F(KPFusedSparseSegmentReduceOpTest, TestColsOutOfBounds) {
  MakeOp(0);

  AddInputFromArray<float>(TensorShape({4, 2}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  AddInputFromArray<int32>(TensorShape({3}), {0, 2, 1});
  AddInputFromArray<int64>(TensorShape({3, 4}),
                           {1, 2, 2, 2, 1, 1, 2, 3, 2, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 5});
  AddInputFromArray<int32>(TensorShape({1}), {0});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "Column index out of range"));
}

TEST_F(KPFusedSparseSegmentReduceOpTest, Test) {
  MakeOp(0);

  AddInputFromArray<float>(TensorShape({4, 2}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  AddInputFromArray<int32>(TensorShape({2}),
                           {0, 2});  //  num_indices != slice_input.dim_size(0)
  AddInputFromArray<int64>(TensorShape({3, 4}),
                           {1, 2, 2, 2, 1, 1, 2, 3, 2, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 2});
  AddInputFromArray<int32>(TensorShape({1}), {0});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), 
                  "indices and slice_input.dim_zie(0) should have same size"));
}

TEST_F(KPFusedSparseSegmentReduceOpTest, TestInvalidData) {
  MakeOp(0);

  AddInputFromArray<float>(
      TensorShape({4, 2, 1}),
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});  // data.dims() > 2
  AddInputFromArray<int32>(TensorShape({3}), {0, 2, 1});
  AddInputFromArray<int64>(TensorShape({3, 4}),
                           {1, 2, 2, 2, 1, 1, 2, 3, 2, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 2});
  AddInputFromArray<int32>(TensorShape({1}), {0});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "input must be 2-D"));
}

TEST_F(KPFusedSparseSegmentReduceOpTest, TestInvalidSliceinput) {
  MakeOp(0);

  AddInputFromArray<float>(TensorShape({4, 2}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  AddInputFromArray<int32>(TensorShape({3}), {0, 2, 1});
  AddInputFromArray<int64>(
      TensorShape({3, 4, 1}),
      {1, 2, 2, 2, 1, 1, 2, 3, 2, 2, 3, 4});  // slice_input.dims() > 2
  AddInputFromArray<int32>(TensorShape({2}), {0, 2});
  AddInputFromArray<int32>(TensorShape({1}), {0});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "slice input must be 2-D"));
}

TEST_F(KPFusedSparseSegmentReduceOpTest, TestInvalidBegin) {
  MakeOp(0);

  AddInputFromArray<float>(TensorShape({4, 2}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  AddInputFromArray<int32>(TensorShape({3}), {0, 2, 1});
  AddInputFromArray<int64>(TensorShape({3, 4}),
                           {1, 2, 2, 2, 1, 1, 2, 3, 2, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({3}),
                           {0, 2, 1});  // begin has 3 elements
  AddInputFromArray<int32>(TensorShape({1}), {0});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "begin must have 2 elements"));
}

TEST_F(KPFusedSparseSegmentReduceOpTest, TestInvalidBegin1) {
  MakeOp(0);

  AddInputFromArray<float>(TensorShape({4, 2}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  AddInputFromArray<int32>(TensorShape({3}), {0, 2, 1});
  AddInputFromArray<int64>(TensorShape({3, 4}),
                           {1, 2, 2, 2, 1, 1, 2, 3, 2, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 2});
  AddInputFromArray<int32>(TensorShape({2}), {0, 1});  // begin_1 has 2 elements

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "begin_1 must have 1 element"));
}

}  // namespace
}  // namespace tensorflow
