/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/kernels/reshape_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

using namespace tensorflow;

REGISTER_OP("KPFusedSparseReshape")
    .Input("shape_input: int64")
    .Input("slice_input: int64")
    .Input("begin: int32")
    .Input("new_shape: int64")
    .Output("out_indices: int64")
    .Output("out_shape: int64")
    .SetShapeFn(shape_inference::UnknownShape);

static void ReshapeKp(OpKernelContext *context, const Tensor &input_indices_in,
             const Tensor &input_shape_in, const Tensor &target_shape_in,
             int output_indices_idx, int output_shape_idx) {
  OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices_in.shape()),
              errors::InvalidArgument(
                  "Input indices should be a matrix but received shape ",
                  input_indices_in.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape_in.shape()),
              errors::InvalidArgument(
                  "Input shape should be a vector but received shape ",
                  input_shape_in.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsVector(target_shape_in.shape()),
              errors::InvalidArgument(
                  "Target shape should be a vector but received shape ",
                  target_shape_in.shape().DebugString()));

  const int64 input_rank = input_shape_in.NumElements();
  const int64 output_rank = target_shape_in.NumElements();
  const TensorShape input_shape(input_shape_in.vec<int64>());
  const int64 dense_size = input_shape.num_elements();
  const int64 nnz = input_indices_in.shape().dim_size(0);

  TensorShape output_shape;
  int64 product = 1;
  int unknown_index = -1;
  auto target_shape = target_shape_in.vec<int64>();
  for (int d = 0; d < output_rank; ++d) {
    const int64 size = target_shape(d);
    if (size == -1) {
      OP_REQUIRES(
          context, unknown_index == -1,
          errors::InvalidArgument("only one output dimension may be -1, "
                                  "not both ",
                                  unknown_index, " and ", d));
      unknown_index = d;
      output_shape.AddDim(1);
    } else {
      OP_REQUIRES(context, size >= 0,
                  errors::InvalidArgument("size ", d,
                                          " must be non-negative, not ", size));
      product *= size;
      output_shape.AddDim(size);
    }
  }
  if (unknown_index != -1) {
    OP_REQUIRES(
        context, product > 0,
        errors::InvalidArgument("reshape cannot infer the missing "
                                "input size for an empty tensor unless all "
                                "specified input sizes are non-zero"));
    const int64 missing = dense_size / product;
    OP_REQUIRES(
        context, product * missing == dense_size,
        errors::InvalidArgument(
            "Input to reshape is a SparseTensor with ", dense_size,
            " dense values, but the requested shape requires a multiple of ",
            product, ". input_shape=", input_shape.DebugString(),
            " output_shape=", output_shape.DebugString()));
    output_shape.set_dim(unknown_index, missing);
  }

  OP_REQUIRES(
      context, output_shape.num_elements() == dense_size,
      errors::InvalidArgument("Input to reshape is a tensor with ", dense_size,
                              " dense values, but the requested shape has ",
                              output_shape.num_elements(),
                              ". input_shape=", input_shape.DebugString(),
                              " output_shape=", output_shape.DebugString()));

  if (input_shape == output_shape) {
    context->set_output(output_indices_idx, input_indices_in);
    context->set_output(output_shape_idx, input_shape_in);
    return;
  }

  gtl::InlinedVector<int64, 8> input_strides(input_rank);
  if (input_rank > 0) {
    input_strides[input_rank - 1] = 1;
    for (int d = input_rank - 2; d >= 0; --d) {
      input_strides[d] = input_strides[d + 1] * input_shape.dim_size(d + 1);
    }
  }

  gtl::InlinedVector<int64, 8> output_strides(output_rank);
  if (output_rank > 0) {
    output_strides[output_rank - 1] = 1;
    for (int d = output_rank - 2; d >= 0; --d) {
      output_strides[d] = output_strides[d + 1] * output_shape.dim_size(d + 1);
    }
  }

  Tensor *result_indices = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(output_indices_idx,
                                          TensorShape({nnz, output_rank}),
                                          &result_indices));
  auto input_ind = input_indices_in.matrix<int64>();
  auto output_ind = result_indices->matrix<int64>();
  for (int i = 0; i < nnz; ++i) {
    int64 id = 0;
    for (int j = 0; j < input_rank; ++j) {
      id += input_ind(i, j) * input_strides[j];
    }
    for (int j = 0; j < output_rank; ++j) {
      output_ind(i, j) = id / output_strides[j];
      id %= output_strides[j];
    }
  }

  Tensor *result_shape = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(output_shape_idx,
                                                   TensorShape({output_rank}),
                                                   &result_shape));
  auto output_shape_vec = result_shape->vec<int64>();
  for (int j = 0; j < output_shape.dims(); ++j) {
    output_shape_vec(j) = output_shape.dim_size(j);
  }
}

class KPFusedSparseReshape : public OpKernel {
 public:
  explicit KPFusedSparseReshape(OpKernelConstruction* context) : OpKernel(context) { }

  void Compute(OpKernelContext* context) override {
    VLOG(2) << "Executing KPFusedSparseReshape operator";
    const Tensor& shape_input = context->input(0);
    const Tensor& slice_input = context->input(1);
    const Tensor& begin = context->input(2);
    const Tensor& new_shape = context->input(3);

    OP_REQUIRES(context, slice_input.dims() == 2, errors::Internal("slice_input dims must == 2"));

    VLOG(2) << "Input shape_input shape: " << shape_input.shape().DebugString();
    VLOG(2) << "Input slice_input shape: " << slice_input.shape().DebugString();
    VLOG(2) << "Input begin value: " << begin.SummarizeValue(10);
    VLOG(2) << "Input new_shape value: " << new_shape.SummarizeValue(10);

    int32 col = begin.flat<int32>().data()[1];
    int64_t stridedslice57_out = shape_input.flat<int64>().data()[0];
    auto slice_input_mat = slice_input.matrix<int64>();

    VLOG(2) << "stridedslice57_out: " << stridedslice57_out;
    VLOG(2) << "slice_input.dim_size(0): " << slice_input.dim_size(0);
    VLOG(2) << "slice_input.dim_size(1): " << slice_input.dim_size(1);
    OP_REQUIRES(context, stridedslice57_out == slice_input.dim_size(0), errors::Internal("concat shape mismatch"));
    VLOG(2) << "Column index from begin: " << col;
    VLOG(2) << "indices size: " << stridedslice57_out;

    Tensor shape_in(DT_INT64, TensorShape({2}));
    auto tensor_flat = shape_in.flat<int64>();
    tensor_flat(0) = stridedslice57_out;
    tensor_flat(1) = 2;

    Tensor indices_in(DT_INT64, TensorShape({stridedslice57_out, 2}));
    auto indices_in_mat = indices_in.matrix<int64>();
    for (int i = 0; i < stridedslice57_out; ++i) {
        indices_in_mat(i, 0) = i;
        indices_in_mat(i, 1) = slice_input_mat(i, col);
    }
    ReshapeKp(context, indices_in, shape_in, new_shape, 0, 1);
  }
};

REGISTER_KERNEL_BUILDER(Name("KPFusedSparseReshape").Device(DEVICE_CPU),
                        KPFusedSparseReshape);
