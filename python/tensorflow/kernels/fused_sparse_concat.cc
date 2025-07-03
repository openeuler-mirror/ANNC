#include <iostream>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

REGISTER_OP("KPFusedSparseConcat")
    .Input("shape: int64")
    .Input("pooling: float")
    .Input("pooling_rows: int32")
    .Output("output0: int32")
    .Output("output1: int32")
    .SetShapeFn(shape_inference::UnknownShape);

class KPFusedSparseConcatOP : public OpKernel {
 public:
  explicit KPFusedSparseConcatOP(OpKernelConstruction* context) : OpKernel(context) { }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& shape = context->input(0);
    const Tensor& pooling = context->input(1);
    const Tensor& pooling_rows = context->input(2);

    OP_REQUIRES(context, shape.dims() == 1, errors::Internal("shape dims must == 1"));
    OP_REQUIRES(context, shape.NumElements() >= 1, errors::Internal("shape NumElements must >= 1"));
    OP_REQUIRES(context, pooling.dims() == 2, errors::Internal("pooling dims must == 2"));
    OP_REQUIRES(context, pooling.dim_size(1) == 10, errors::Internal("pooling dim_size(1) == 10"));
    OP_REQUIRES(context, pooling_rows.dims() == 0, errors::Internal("pooling_rows must be a scalar"));

    int padding_rows = static_cast<int32>(shape.flat<int64>()(0)) - pooling_rows.scalar<int32>()();
    OP_REQUIRES(context, padding_rows >= 0, errors::Internal("padding_rows must >= 0"));
    Tensor* output0 = nullptr;
    Tensor* output1 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}),
                                            &output0));

    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({}),
                                            &output1));
    output0->scalar<int32>()() = padding_rows;

    int first_dims = padding_rows + pooling.dim_size(0);
    int second_dims = pooling.dim_size(1);
    OP_REQUIRES(context, first_dims * second_dims % 1510 == 0, errors::Internal("padding cannot reshape to [-1, 1510]"));
    output1->scalar<int32>()() = first_dims * second_dims / 1510;
  }
};

REGISTER_KERNEL_BUILDER(Name("KPFusedSparseConcat").Device(DEVICE_CPU),
                        KPFusedSparseConcatOP);