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

#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

class KPFusedEmbeddingPaddingOp : public OpKernel {
public:
  explicit KPFusedEmbeddingPaddingOp(OpKernelConstruction* context) : OpKernel(context) {
    fast_ = (type_string() == "KPFusedEmbeddingPaddingFast");
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& origin_shape = context->input(0);
    const Tensor& input = context->input(1);
    const Tensor& input_rows = context->input(2);
    const Tensor& reshape_sizes = context->input(3);

    const Tensor& pack = context->input(4);

    VLOG(1) << "Input shape: " << input.shape().DebugString();
    OP_REQUIRES(context,
                TensorShapeUtils::IsVector(origin_shape.shape()), 
                errors::InvalidArgument("origin_shape dims must 1D, not ", origin_shape.shape().DebugString())
    );
    OP_REQUIRES(context,
                origin_shape.NumElements() == 2,
                errors::InvalidArgument("origin_shape NumElements must == 2, not ", origin_shape.NumElements())
    );
    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(input.shape()),
                errors::InvalidArgument("input dims must 2D, not ", input.shape().DebugString()));
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(input_rows.shape()),
                errors::InvalidArgument("input_rows must be a scalar")
    );
    OP_REQUIRES(context,
                TensorShapeUtils::IsVector(reshape_sizes.shape()),
                errors::InvalidArgument("sizes input must be 1-D, not ", reshape_sizes.shape().DebugString())
    );
    OP_REQUIRES(context, 
                reshape_sizes.NumElements() == 2,
                errors::InvalidArgument("reshape_sizes NumElements must == 2"));

    int input_rows_value = input_rows.scalar<int32>()();
    int padding_rows = static_cast<int32>(origin_shape.flat<int64>()(0)) - input_rows_value;
    auto reshape_cols = reshape_sizes.flat<int32>()(1);
    int output_rows = padding_rows + input.dim_size(0);
    int output_cols = input.dim_size(1);
    OP_REQUIRES(context,
                padding_rows >= 0,
                errors::InvalidArgument("Pooling size(", input_rows_value,
                ") is greater than Input size(", static_cast<int32>(origin_shape.flat<int64>()(0)), ")"));
    OP_REQUIRES(context,
                reshape_cols > 0,
                errors::InvalidArgument("reshape_cols must > 0"));
    OP_REQUIRES(context,
                reshape_sizes.flat<int32>()(0) == -1,
                errors::InvalidArgument("reshape[0] is not -1"));
    OP_REQUIRES(context,
                pack.scalar<int32>()() == output_cols,
                errors::InvalidArgument("pack(", pack.scalar<int32>()(), ") is not equal to embedding dims"));

    Tensor* output0 = nullptr;
    Tensor* output1 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}),
                                            &output0));
    output0->scalar<int32>()() = padding_rows;
    OP_REQUIRES(context,
                output_rows * output_cols % reshape_cols == 0,
                errors::InvalidArgument("padding cannot reshape to [-1, ", reshape_cols, "]")
    );
    int reshape_rows = output_rows * output_cols / reshape_cols;
    if (fast_) {
      OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}), &output1));
      output1->scalar<int32>()() = reshape_rows;
      return;
    }

    TensorShape reshaped_shape({reshape_rows, reshape_cols});
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, reshaped_shape, &output1));
    float* output_data = output1->flat<float>().data();
    const float* input_data = input.flat<float>().data();
    std::memcpy(output_data, input_data, input_rows_value * output_cols * sizeof(float));
    std::memset(output_data + input_rows_value * output_cols,
                0.0f,
                padding_rows * output_cols * sizeof(float));
  }

private:
    bool fast_;
};


REGISTER_KERNEL_BUILDER(Name("KPFusedEmbeddingPadding").Device(DEVICE_CPU),
                        KPFusedEmbeddingPaddingOp);

REGISTER_KERNEL_BUILDER(Name("KPFusedEmbeddingPaddingFast").Device(DEVICE_CPU),
                        KPFusedEmbeddingPaddingOp);

}