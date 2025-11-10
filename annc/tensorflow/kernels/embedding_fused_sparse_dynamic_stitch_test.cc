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

class KPFusedSparseDynamicStitchOpTest : public OpsTestBase {
 protected:
  void MakeOp(int N) {
    TF_ASSERT_OK(NodeDefBuilder("kp_fused_sparse_dynamic_stitch",
                                "KPFusedSparseDynamicStitch")
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(N, DT_FLOAT))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(KPFusedSparseDynamicStitchOpTest, TestTwoTables) {
  MakeOp(2);  // num_partitions = 2

  AddInputFromArray<int64>(TensorShape({4}), {0, 3, 2, 1});
  AddInputFromArray<float>(TensorShape({3, 2}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  AddInputFromArray<float>(TensorShape({2, 2}), {7.0f, 8.0f, 9.0f, 10.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 2}));
  test::FillValues<float>(&expected,
                          {1.0f, 2.0f, 9.0f, 10.0f, 3.0f, 4.0f, 7.0f, 8.0f});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(KPFusedSparseDynamicStitchOpTest, TestDifferentStride) {
  MakeOp(2);

  AddInputFromArray<int64>(TensorShape({4}), {0, 3, 2, 1});
  AddInputFromArray<float>(TensorShape({3, 2}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  AddInputFromArray<float>(TensorShape({1, 4}), {7.0f, 8.0f, 9.0f, 10.0f});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(),"All inputs must have same second dimension"));
}

TEST_F(KPFusedSparseDynamicStitchOpTest, TestIndicesOutOfBounds) {
  MakeOp(2);

  AddInputFromArray<int64>(TensorShape({4}), {0, 6, 2, 1});
  AddInputFromArray<float>(TensorShape({3, 2}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  AddInputFromArray<float>(TensorShape({2, 2}), {7.0f, 8.0f, 9.0f, 10.0f});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(),"row_id out of range"));
}

TEST_F(KPFusedSparseDynamicStitchOpTest, TestInputDims) {
  MakeOp(2);

  AddInputFromArray<int64>(TensorShape({4}), {0, 6, 2, 1});
  AddInputFromArray<float>(TensorShape({3, 2, 1}),
                           {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  AddInputFromArray<float>(TensorShape({2, 2, 1}), {7.0f, 8.0f, 9.0f, 10.0f});

  Status s = RunOpKernel();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.message(),"input dims must == 2"));
}

}  // namespace
}  // namespace tensorflow
