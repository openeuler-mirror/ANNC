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

TEST_F(KPFusedSparseSegmentReduceNonzeroOpTest, TestInvalidData) {
  MakeOp(0);

  AddInputFromArray<float>(TensorShape({4, 2}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  AddInputFromArray<int32>(TensorShape({3}), {0, 2, 1});
  AddInputFromArray<int64>(TensorShape({3, 4}),
                           {1, 2, 2, 2, 1, 1, 2, 3, 2, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {0, 2});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(), "Input data must be a vector") !=
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
                  "indices and slice_input.dim_zie(0) should have same size"));
}

}  // namespace
}  // namespace tensorflow
