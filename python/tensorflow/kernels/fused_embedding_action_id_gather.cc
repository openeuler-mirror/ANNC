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

REGISTER_OP("KPFusedEmbeddingActionIdGather")
    .Input("input0: int32")
    .Input("input1: float")
    .Input("input2: int32")
    .Input("input3: int32")
    .Output("output0: float")
    .SetShapeFn(shape_inference::UnknownShape);

static void GatherV2Impl(OpKernelContext* context,
                    const float* params_data, 
                    const TensorShape& params_shape,
                    const int* indices_data,
                    const TensorShape& indices_shape,
                    int axis, Tensor* temp) {
  TensorShape temp_shape;
  const int P0 = params_shape.dim_size(0);
  int P1 = 1;
  for (int d = 0; d < indices_shape.dims(); ++d) {
    temp_shape.AddDim(indices_shape.dim_size(d));
  }

  for (int d = 1; d < params_shape.dims(); ++d) {
    temp_shape.AddDim(params_shape.dim_size(d));
    P1 *= params_shape.dim_size(d);
  }
  OP_REQUIRES_OK(context,
                  context->allocate_temp(DT_FLOAT, temp_shape, temp));
  VLOG(2) << "temp shape: " << temp->shape().DebugString();

  const int num_indices = indices_shape.num_elements();
  float* temp_data = temp->flat<float>().data();
  VLOG(2) << "num_indices : " << num_indices;
  if (axis == 0) {
    const int slice_size = P1;
    for (int i = 0; i < num_indices; ++i) {
      int idx = indices_data[i];
      if (idx < 0 || idx >= P0) {
        LOG(FATAL) << "GatherV2 axis=0: index out of range: " << idx;
      }
      std::memcpy(temp_data + i * slice_size,
                  params_data + idx * slice_size,
                  sizeof(float) * slice_size);
    }
  } else {
    LOG(FATAL) << "Only axis=0 is supported";
  }
  VLOG(2) << "temp value : " << temp->DebugString(100);
}

class KPFusedEmbeddingActionIdGatherOp : public OpKernel {
 public:
  explicit KPFusedEmbeddingActionIdGatherOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& indices1 = context->input(0);
    const Tensor& params = context->input(1);
    const Tensor& indices2 = context->input(2);
    const Tensor& pack_dim = context->input(3);

    VLOG(2) << "indices1 shape: " << indices1.shape().DebugString();
    VLOG(2) << "params shape: " << params.shape().DebugString();
    VLOG(2) << "indices2 shape: " << indices2.shape().DebugString();
    OP_REQUIRES(context, indices1.dims() <= 2, errors::InvalidArgument("indices1 dims must <= 2"));
    OP_REQUIRES(context, indices2.dims() <= 2, errors::InvalidArgument("indices2 dims must <= 2"));
    OP_REQUIRES(context, params.dims() == 2, errors::InvalidArgument("params dims must = 2"));
    OP_REQUIRES(context, pack_dim.NumElements() == 1, errors::InvalidArgument("pack_dim NumElements must = 1"));

    Tensor temp;
    GatherV2Impl(context, params.flat<float>().data(), params.shape(),
                 indices1.flat<int>().data(), indices1.shape(),
                 0, &temp);
    Tensor temp1;
    GatherV2Impl(context, temp.flat<float>().data(), temp.shape(),
                 indices2.flat<int>().data(), indices2.shape(),
                 0, &temp1);
    int pack_size = pack_dim.scalar<int32>()();
    VLOG(2) << "pack_size value: " << pack_size;
    int a_reshaped_cols = temp1.NumElements() / pack_size;
    auto a_reshaped = temp1.shaped<float, 2>({pack_size, a_reshaped_cols});
    VLOG(2) << "a_reshaped_cols : " << a_reshaped_cols;
    Tensor* output;
    int output_cols = a_reshaped_cols + 1680;
    OP_REQUIRES_OK(context,
                  context->allocate_output(0, TensorShape({pack_size, output_cols}), &output));
    VLOG(2) << "output shape: " << output->shape().DebugString();
    auto output_matrix = output->matrix<float>();
    output_matrix.slice(
      Eigen::array<Eigen::Index, 2>{0, 0},
      Eigen::array<Eigen::Index, 2>{pack_size, a_reshaped_cols}) = a_reshaped;
    
    output_matrix.slice(
      Eigen::array<Eigen::Index, 2>{0, a_reshaped_cols},
      Eigen::array<Eigen::Index, 2>{pack_size, 1680}).setZero();
  }
};


REGISTER_KERNEL_BUILDER(Name("KPFusedEmbeddingActionIdGather").Device(DEVICE_CPU),
                        KPFusedEmbeddingActionIdGatherOp);

}