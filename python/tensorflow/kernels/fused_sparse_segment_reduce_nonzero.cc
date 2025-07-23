#include <arm_neon.h>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;

REGISTER_OP("KPFusedSparseSegmentReduceNonzero")
    .Input("data: float")
    .Input("indices: Tidx")
    .Input("slice_input: int64")
    .Input("begin: int32")
    .Input("end: int32")
    .Input("strides: int32")
    .Attr("combiner: int = 1")  // 0 for SUM, 1 for MEAN
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Output("output_shape: int64")
    .Output("output_indices: int64")
    .Output("output_nonzero: float")
    .SetShapeFn(shape_inference::UnknownShape);

template <typename Tidx>
class KPFusedSparseSegmentReduceNonzeroOp : public OpKernel {
 public:
  explicit KPFusedSparseSegmentReduceNonzeroOp(OpKernelConstruction* context)
      : OpKernel(context) {
    int combiner_mode;
    OP_REQUIRES_OK(context, context->GetAttr("combiner", &combiner_mode));
    OP_REQUIRES(context, combiner_mode == 0 || combiner_mode == 1,
                errors::InvalidArgument("combiner must be 0 or 1"));
    is_mean_ = (combiner_mode == 1);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& slice_input = context->input(2);
    const Tensor& begin = context->input(3);

    int64 num_indices = indices.dim_size(0);
    int32 col = begin.flat<int32>().data()[1];
    OP_REQUIRES(context, input_tensor.dims() == 1,
                errors::InvalidArgument("Input data must be a vector"));

    auto input_data = input_tensor.flat<float>();
    auto indices_vec = indices.vec<Tidx>();
    auto slice_input_mat = slice_input.matrix<int64>();

 	// Calculate max segment_id
    int64 max_seg_id = 0;
    for (int64 i = 0; i < num_indices; ++i) {
      int64 seg_id = slice_input_mat(i, col);
      if (seg_id > max_seg_id) {
        max_seg_id = seg_id;
      }
    }

    const int64 batch_size = max_seg_id + 1;

    Tensor* output_shape = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({1}), &output_shape));
    output_shape->flat<int64>()(0) = batch_size;

    std::vector<std::pair<int64, float>> results(batch_size);
    int64 num_nonzero = 0;
    Tensor temp(DT_FLOAT, TensorShape({batch_size}));
    temp.flat<float>().setZero();
    auto temp_vec = temp.flat<float>();

    if (is_mean_) {
      Tensor counts(DT_INT32, TensorShape({batch_size}));
      counts.flat<int32>().setZero();
      auto counts_vec = counts.flat<int32>();

      for (int64 i = 0; i < num_indices; ++i) {
        const int32 seg_id = slice_input_mat(i, col);
        const int32 data_row = indices_vec(i);
        counts_vec(seg_id) += 1;
        temp_vec(seg_id) += input_data(data_row);
      }
  
      for (int64 seg = 0; seg < batch_size; ++seg) {
        const int32_t count = counts_vec(seg);
        if (count > 0) {
          const float inv_count = 1.0f / static_cast<float>(count);
          float value = temp_vec(seg);
          if (value != 0) {
            results[num_nonzero++] = {seg, value * inv_count};
          }
        }
      }
    } else {
      for (int64 i = 0; i < num_indices; ++i) {
        const int32 seg_id = slice_input_mat(i, col);
        const int32 data_row = indices_vec(i);
        temp_vec(seg_id) += input_data(data_row);
      }
  
      for (int64 seg = 0; seg < batch_size; ++seg) {
        float value = temp_vec(seg);
        if (value != 0) {
          results[num_nonzero++] = {seg, value};
        }
      }
    }
    Tensor* output_indices = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({num_nonzero, 1}),
                                            &output_indices));
    auto output_indices_data = output_indices->flat<int64>();

    Tensor* output_nonzero = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, TensorShape({num_nonzero}),
                                            &output_nonzero));
    auto output_nonzero_data = output_nonzero->flat<float>();
    for (int64 i = 0; i < num_nonzero; ++i) {
      output_indices_data(i) = results[i].first;
      output_nonzero_data(i) = results[i].second;
    }

  }

 private:
  bool is_mean_;
};

#define REGISTER_KERNEL(Tidx)                                       \
  REGISTER_KERNEL_BUILDER(Name("KPFusedSparseSegmentReduceNonzero") \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<Tidx>("Tidx"),        \
                          KPFusedSparseSegmentReduceNonzeroOp<Tidx>);
REGISTER_KERNEL(int64)
REGISTER_KERNEL(int32)
#undef REGISTER_KERNEL
