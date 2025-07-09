#include <iostream>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/op_kernel.h"
namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("KPFusedEmbeddingPaddingFast")
    .Input("input0: int64")
    .Input("input1: float")
    .Input("input2: int32")
    .Input("input3: int32")
    .Output("output0: int32")
    .Output("output1: int32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle scalar_shape = c->Scalar();
      c->set_output(0, scalar_shape);
      c->set_output(1, scalar_shape);
      return Status::OK();
    });


REGISTER_OP("KPFusedEmbeddingPadding")
    .Input("input0: int64")
    .Input("input1: float")
    .Input("input2: int32")
    .Input("input3: int32")
    .Output("output0: int32")
    .Output("output1: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      ShapeHandle scalar_shape = c->Scalar();
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(3, &out));
      c->set_output(0, scalar_shape);
      c->set_output(1, out);
      return Status::OK();
    });


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

    OP_REQUIRES(context, origin_shape.dims() == 1, errors::InvalidArgument("origin_shape dims must == 1"));
    OP_REQUIRES(context, origin_shape.NumElements() >= 1, errors::InvalidArgument("origin_shape NumElements must >= 1"));
    OP_REQUIRES(context, input.dims() == 2, errors::InvalidArgument("input dims must == 2"));
    OP_REQUIRES(context, input_rows.dims() == 0, errors::InvalidArgument("input_rows must be a scalar"));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsVector(reshape_sizes.shape()),
        errors::InvalidArgument("sizes input must be 1-D, not ", reshape_sizes.shape().DebugString())
    );
    OP_REQUIRES(context, reshape_sizes.NumElements() == 2, errors::InvalidArgument("reshape_sizes NumElements must == 2"));
    int input_rows_value = input_rows.scalar<int32>()();
    int padding_rows = static_cast<int32>(origin_shape.flat<int64>()(0)) - input_rows_value;
    OP_REQUIRES(context, padding_rows >= 0, errors::InvalidArgument("padding_rows must >= 0"));
    auto reshape_cols = reshape_sizes.flat<int32>()(1);
    OP_REQUIRES(context, reshape_sizes.flat<int32>()(0) == -1, errors::InvalidArgument("reshape first dim must be -1"));

    Tensor* output0 = nullptr;
    Tensor* output1 = nullptr;
    Tensor padding;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}),
                                            &output0));
    output0->scalar<int32>()() = padding_rows;

    int output_rows = padding_rows + input.dim_size(0);
    int output_cols = input.dim_size(1);
    OP_REQUIRES(
      context,
      output_rows * output_cols % reshape_cols == 0,
      errors::InvalidArgument("padding cannot reshape to [-1, ", reshape_cols, "]")
    );
    int reshape_rows = output_rows * output_cols / reshape_cols;
    if (fast_) {
      OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({}),
                                            &output1));
      output1->scalar<int32>()() = reshape_rows;
      return;
    }

    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_FLOAT, TensorShape({padding_rows + input_rows_value, output_cols}),
                                            &padding));
    auto input_matrix = input.matrix<float>();
    auto padding_matrix = padding.matrix<float>();

    padding_matrix.slice(
      Eigen::array<Eigen::Index, 2>{0, 0},
      Eigen::array<Eigen::Index, 2>{input_rows_value, output_cols}) = input_matrix;
    
    padding_matrix.slice(
      Eigen::array<Eigen::Index, 2>{input_rows_value, 0},
      Eigen::array<Eigen::Index, 2>{padding_rows, output_cols}).setZero();

    TensorShape reshaped_shape({reshape_rows, reshape_cols});
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, reshaped_shape, &output1));
    output1->flat<float>() = padding.flat<float>();
  }

  private:
    bool fast_;
};


REGISTER_KERNEL_BUILDER(Name("KPFusedEmbeddingPadding").Device(DEVICE_CPU),
                        KPFusedEmbeddingPaddingOp);

REGISTER_KERNEL_BUILDER(Name("KPFusedEmbeddingPaddingFast").Device(DEVICE_CPU),
                        KPFusedEmbeddingPaddingOp);

}