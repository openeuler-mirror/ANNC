/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

class KPFusedEmbeddingPaddingTest : public OpsTestBase {
 protected:
  void MakeOp(DataType input_shape_type, DataType pooling_type, DataType reshape_type, DataType const_type) {
    TF_ASSERT_OK(NodeDefBuilder("fused_padding", "KPFusedEmbeddingPadding")
                     .Input(FakeInput(input_shape_type))
                     .Input(FakeInput(pooling_type))
                     .Input(FakeInput(const_type))
                     .Input(FakeInput(reshape_type))
                     .Input(FakeInput(const_type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  Status FeedAndRun(const int embedding_dims, const int table_size,
                    const int pooling_size, const int reshape_size) {
    MakeOp(DT_INT64, DT_FLOAT, DT_INT32, DT_INT32);
    AddInputFromArray<int64>(TensorShape({2}), {table_size, embedding_dims});
    AddInput<float>(TensorShape({pooling_size, embedding_dims}), [](int i) -> float { 
      return static_cast<float>(i + 1); 
    });
    AddInputFromArray<int32>(TensorShape({}), {pooling_size});
    AddInputFromArray<int32>(TensorShape({2}), {-1, reshape_size});
    AddInputFromArray<int32>(TensorShape({}), {embedding_dims});
    return RunOpKernel();
  }

  void MakeFastOp(DataType input_shape_type, DataType pooling_type, DataType reshape_type, DataType const_type) {
    TF_ASSERT_OK(NodeDefBuilder("fused_padding_fast", "KPFusedEmbeddingPaddingFast")
                     .Input(FakeInput(input_shape_type))
                     .Input(FakeInput(pooling_type))
                     .Input(FakeInput(const_type))
                     .Input(FakeInput(reshape_type))
                     .Input(FakeInput(const_type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  Status FeedAndRunFast(const int embedding_dims, const int table_size,
                        const int pooling_size, const int reshape_size) {
    MakeFastOp(DT_INT64, DT_FLOAT, DT_INT32, DT_INT32);
    AddInputFromArray<int64>(TensorShape({2}), {table_size, embedding_dims});
    AddInput<float>(TensorShape({pooling_size, embedding_dims}), [](int i) -> float { 
      return static_cast<float>(i + 1); 
    });
    AddInputFromArray<int32>(TensorShape({}), {pooling_size});
    AddInputFromArray<int32>(TensorShape({2}), {-1, reshape_size});
    AddInputFromArray<int32>(TensorShape({}), {embedding_dims});
    return RunOpKernel();
  }
};

TEST_F(KPFusedEmbeddingPaddingTest, FusedPaddingWithEmbeddingDims10_0) {
  // Feed and run
  const int embedding_dims = 10;
  const int table_size = 151;
  const int pooling_size = 151;
  const int reshape_size = 1510;
  TF_ASSERT_OK(FeedAndRun(embedding_dims, table_size, pooling_size, reshape_size));

  // Check the output.
  Tensor expected1(allocator(), DT_INT32, TensorShape({}));
  Tensor expected2(allocator(), DT_FLOAT, TensorShape({table_size * embedding_dims / reshape_size, reshape_size}));
  test::FillValues<int32>(&expected1, {table_size - pooling_size});
  test::FillFn<float>(&expected2, [=](int i) -> float { 
    if (i < pooling_size * embedding_dims) {
      return static_cast<float>(i + 1); 
    } else {
      return 0.0f;
    }
  });
  test::ExpectTensorEqual<int32>(expected1, *GetOutput(0));
  test::ExpectTensorNear<float>(expected2, *GetOutput(1), 1e-5);
}

TEST_F(KPFusedEmbeddingPaddingTest, FusedPaddingWithEmbeddingDims10_1) {
  // Feed and run
  const int embedding_dims = 10;
  const int table_size = 1510;
  const int pooling_size = 151;
  const int reshape_size = 1510;
  TF_ASSERT_OK(FeedAndRun(embedding_dims, table_size, pooling_size, reshape_size));

  // Check the output.
  Tensor expected1(allocator(), DT_INT32, TensorShape({}));
  Tensor expected2(allocator(), DT_FLOAT, TensorShape({table_size * embedding_dims / reshape_size, reshape_size}));
  test::FillValues<int32>(&expected1, {table_size - pooling_size});
  test::FillFn<float>(&expected2, [=](int i) -> float { 
    if (i < pooling_size * embedding_dims) {
      return static_cast<float>(i + 1); 
    } else {
      return 0.0f;
    }
  });
  test::ExpectTensorEqual<int32>(expected1, *GetOutput(0));
  test::ExpectTensorNear<float>(expected2, *GetOutput(1), 1e-5);
}

TEST_F(KPFusedEmbeddingPaddingTest, FusedPaddingWithEmbeddingDims12_0) {
  // Feed and run
  const int embedding_dims = 12;
  const int table_size = 2;
  const int pooling_size = 2;
  const int reshape_size = 24;
  TF_ASSERT_OK(FeedAndRun(embedding_dims, table_size, pooling_size, reshape_size));

  // Check the output.
  Tensor expected1(allocator(), DT_INT32, TensorShape({}));
  Tensor expected2(allocator(), DT_FLOAT, TensorShape({table_size * embedding_dims / reshape_size, reshape_size}));
  test::FillValues<int32>(&expected1, {table_size - pooling_size});
  test::FillFn<float>(&expected2, [=](int i) -> float { 
    if (i < pooling_size * embedding_dims) {
      return static_cast<float>(i + 1); 
    } else {
      return 0.0f;
    }
  });
  test::ExpectTensorEqual<int32>(expected1, *GetOutput(0));
  test::ExpectTensorNear<float>(expected2, *GetOutput(1), 1e-5);
}

TEST_F(KPFusedEmbeddingPaddingTest, FusedPaddingWithEmbeddingDims12_1) {
  // Feed and run
  const int embedding_dims = 12;
  const int table_size = 200;
  const int pooling_size = 2;
  const int reshape_size = 24;
  TF_ASSERT_OK(FeedAndRun(embedding_dims, table_size, pooling_size, reshape_size));

  // Check the output.
  Tensor expected1(allocator(), DT_INT32, TensorShape({}));
  Tensor expected2(allocator(), DT_FLOAT, TensorShape({table_size * embedding_dims / reshape_size, reshape_size}));
  test::FillValues<int32>(&expected1, {table_size - pooling_size});
  test::FillFn<float>(&expected2, [=](int i) -> float { 
    if (i < pooling_size * embedding_dims) {
      return static_cast<float>(i + 1); 
    } else {
      return 0.0f;
    }
  });
  test::ExpectTensorEqual<int32>(expected1, *GetOutput(0));
  test::ExpectTensorNear<float>(expected2, *GetOutput(1), 1e-5);
}

TEST_F(KPFusedEmbeddingPaddingTest, FusedPaddingFastWithEmbeddingDims10_0) {
  // Feed and run
  const int embedding_dims = 10;
  const int table_size = 151;
  const int pooling_size = 151;
  const int reshape_size = 1510;
  TF_ASSERT_OK(FeedAndRunFast(embedding_dims, table_size, pooling_size, reshape_size));

  // Check the output.
  Tensor expected1(allocator(), DT_INT32, TensorShape({}));
  Tensor expected2(allocator(), DT_INT32, TensorShape({}));
  test::FillValues<int32>(&expected1, {table_size - pooling_size});
  test::FillValues<int32>(&expected2, {table_size * embedding_dims / reshape_size});
  test::ExpectTensorEqual<int32>(expected1, *GetOutput(0));
  test::ExpectTensorEqual<int32>(expected2, *GetOutput(1));
}

TEST_F(KPFusedEmbeddingPaddingTest, FusedPaddingFastWithEmbeddingDims10_1) {
  // Feed and run
  const int embedding_dims = 10;
  const int table_size = 1510;
  const int pooling_size = 151;
  const int reshape_size = 1510;
  TF_ASSERT_OK(FeedAndRunFast(embedding_dims, table_size, pooling_size, reshape_size));

  // Check the output.
  Tensor expected1(allocator(), DT_INT32, TensorShape({}));
  Tensor expected2(allocator(), DT_INT32, TensorShape({}));
  test::FillValues<int32>(&expected1, {table_size - pooling_size});
  test::FillValues<int32>(&expected2, {table_size * embedding_dims / reshape_size});
  test::ExpectTensorEqual<int32>(expected1, *GetOutput(0));
  test::ExpectTensorEqual<int32>(expected2, *GetOutput(1));
}

TEST_F(KPFusedEmbeddingPaddingTest, FusedPaddingFastWithEmbeddingDims12_0) {
  // Feed and run
  const int embedding_dims = 12;
  const int table_size = 2;
  const int pooling_size = 2;
  const int reshape_size = 24;
  TF_ASSERT_OK(FeedAndRunFast(embedding_dims, table_size, pooling_size, reshape_size));

  // Check the output.
  Tensor expected1(allocator(), DT_INT32, TensorShape({}));
  Tensor expected2(allocator(), DT_INT32, TensorShape({}));
  test::FillValues<int32>(&expected1, {table_size - pooling_size});
  test::FillValues<int32>(&expected2, {table_size * embedding_dims / reshape_size});
  test::ExpectTensorEqual<int32>(expected1, *GetOutput(0));
  test::ExpectTensorEqual<int32>(expected2, *GetOutput(1));
}

TEST_F(KPFusedEmbeddingPaddingTest, FusedPaddingFastWithEmbeddingDims12_1) {
  // Feed and run
  const int embedding_dims = 12;
  const int table_size = 200;
  const int pooling_size = 2;
  const int reshape_size = 24;
  TF_ASSERT_OK(FeedAndRunFast(embedding_dims, table_size, pooling_size, reshape_size));

  // Check the output.
  Tensor expected1(allocator(), DT_INT32, TensorShape({}));
  Tensor expected2(allocator(), DT_INT32, TensorShape({}));
  test::FillValues<int32>(&expected1, {table_size - pooling_size});
  test::FillValues<int32>(&expected2, {table_size * embedding_dims / reshape_size});
  test::ExpectTensorEqual<int32>(expected1, *GetOutput(0));
  test::ExpectTensorEqual<int32>(expected2, *GetOutput(1));
}

TEST_F(KPFusedEmbeddingPaddingTest, FusedPaddingWithUnexpectReshape) {
  // Feed and run
  const int embedding_dims = 12;
  const int table_size = 200;
  const int pooling_size = 2;
  const int reshape_size = 24;
  MakeOp(DT_INT64, DT_FLOAT, DT_INT32, DT_INT32);
  AddInputFromArray<int64>(TensorShape({2}), {table_size, embedding_dims});
  AddInput<float>(TensorShape({pooling_size, embedding_dims}), [](int i) -> float { 
    return static_cast<float>(i + 1); 
  });
  AddInputFromArray<int32>(TensorShape({}), {pooling_size});
  AddInputFromArray<int32>(TensorShape({2}), {10, reshape_size});
  AddInputFromArray<int32>(TensorShape({}), {embedding_dims});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "reshape[0] is not -1"))
      << s;
}

TEST_F(KPFusedEmbeddingPaddingTest, FusedPaddingWithUnexpectPack) {
  // Feed and run
  const int embedding_dims = 12;
  const int table_size = 200;
  const int pooling_size = 2;
  const int reshape_size = 24;
  MakeOp(DT_INT64, DT_FLOAT, DT_INT32, DT_INT32);
  AddInputFromArray<int64>(TensorShape({2}), {table_size, embedding_dims});
  AddInput<float>(TensorShape({pooling_size, embedding_dims}), [](int i) -> float { 
    return static_cast<float>(i + 1); 
  });
  AddInputFromArray<int32>(TensorShape({}), {pooling_size});
  AddInputFromArray<int32>(TensorShape({2}), {-1, reshape_size});
  AddInputFromArray<int32>(TensorShape({}), {10});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "pack(10) is not equal to embedding dims"))
      << s;
}

TEST_F(KPFusedEmbeddingPaddingTest, FusedPaddingWithPoolingSizeGreaterInput) {
  // Feed and run
  const int embedding_dims = 12;
  const int table_size = 200;
  const int pooling_size = 201;
  const int reshape_size = 24;
  MakeOp(DT_INT64, DT_FLOAT, DT_INT32, DT_INT32);
  AddInputFromArray<int64>(TensorShape({2}), {table_size, embedding_dims});
  AddInput<float>(TensorShape({pooling_size, embedding_dims}), [](int i) -> float { 
    return static_cast<float>(i + 1); 
  });
  AddInputFromArray<int32>(TensorShape({}), {pooling_size});
  AddInputFromArray<int32>(TensorShape({2}), {-1, reshape_size});
  AddInputFromArray<int32>(TensorShape({}), {embedding_dims});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "Pooling size(201) is greater than Input size(200)"))
      << s;
}

}  // end namespace tensorflow
