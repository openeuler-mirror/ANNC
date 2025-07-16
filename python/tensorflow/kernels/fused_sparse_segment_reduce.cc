#include <arm_neon.h>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;

REGISTER_OP("KPFusedSparseSegmentReduce")
    .Input("data: float")
    .Input("indices: Tidx")
    .Input("slice_input: int64")
    .Input("begin: int32")
    .Input("end: int32")
    .Input("strides: int32")
    .Attr("combiner: int = 1")  // 0 for SUM, 1 for MEAN
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Output("output: float")
    .Output("slice_output: int32")
    .SetShapeFn(shape_inference::UnknownShape);
template <typename Tidx>
class KPFusedSparseSegmentReduceOp : public OpKernel {
 public:
  explicit KPFusedSparseSegmentReduceOp(OpKernelConstruction* context)
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

    int64_t num_indices = indices.dim_size(0);
    int64_t embedding_size = input_tensor.dim_size(1);
    int32 col = begin.flat<int32>().data()[1];

    auto input_data = input_tensor.matrix<float>().data();
    auto indices_vec = indices.vec<Tidx>();
    auto slice_input_mat = slice_input.matrix<int64>();

    // Calculate max segment_id
    int64 max_seg_id = 0;
    for (int32 i = 0; i < num_indices; ++i) {
      int64 seg_id = slice_input_mat(i, col);
      if (seg_id > max_seg_id) {
        max_seg_id = seg_id;
      }
    }
    const int64 batch_size = max_seg_id + 1;

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({batch_size, embedding_size}), &output));
    output->flat<float>().setZero();
    Tensor* slice_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({}), &slice_out));
    slice_out->scalar<int32>()() = batch_size;

    auto output_data = output->matrix<float>().data();

    if (is_mean_) {
      Tensor counts(DT_INT32, TensorShape({batch_size}));
      counts.flat<int32>().setZero();
      auto counts_vec = counts.flat<int32>();

      for (int64 i = 0; i < num_indices; ++i) {
        const int32 seg_id = slice_input_mat(i, col);
        const int32 data_row = indices_vec(i);
        counts_vec(seg_id) += 1;

        float* output_row = output_data + seg_id * embedding_size;
        const float* input_data_row = input_data + data_row * embedding_size;
        int64 j = 0;
        for (; j + 3 < embedding_size; j += 4) {
          float32x4_t out = vld1q_f32(output_row + j);
          float32x4_t data = vld1q_f32(input_data_row + j);
          out = vaddq_f32(out, data);
          vst1q_f32(output_row + j, out);
        }

        for (; j < embedding_size; ++j) {
          output_row[j] += input_data_row[j];
        }
      }

      for (int64_t seg = 0; seg < batch_size; ++seg) {
        const int32_t count = counts_vec(seg);
        if (count > 0) {
          const float inv_count = 1.0f / static_cast<float>(count);
          const float32x4_t inv_count_vec = vdupq_n_f32(inv_count);

          float* row_start = output_data + seg * embedding_size;
          int64_t j = 0;

          for (; j + 3 < embedding_size; j += 4) {
            float32x4_t val = vld1q_f32(row_start + j);
            val = vmulq_f32(val, inv_count_vec);
            vst1q_f32(row_start + j, val);
          }

          for (; j < embedding_size; ++j) {
            row_start[j] *= inv_count;
          }
        }
      }
    } else {
      for (int64 i = 0; i < num_indices; ++i) {
        const int32 seg_id = slice_input_mat(i, col);
        const int32 data_row = indices_vec(i);

        float* output_row = output_data + seg_id * embedding_size;
        const float* input_data_row = input_data + data_row * embedding_size;
        int64 j = 0;
        for (; j + 3 < embedding_size; j += 4) {
          float32x4_t out = vld1q_f32(output_row + j);
          float32x4_t data = vld1q_f32(input_data_row + j);
          out = vaddq_f32(out, data);
          vst1q_f32(output_row + j, out);
        }

        for (; j < embedding_size; ++j) {
          output_row[j] += input_data_row[j];
        }
      }
    }
  }

 private:
  bool is_mean_;
};

#define REGISTER_KERNEL(Tidx)                                \
  REGISTER_KERNEL_BUILDER(Name("KPFusedSparseSegmentReduce") \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<Tidx>("Tidx"), \
                          KPFusedSparseSegmentReduceOp<Tidx>);
REGISTER_KERNEL(int64)
REGISTER_KERNEL(int32)
#undef REGISTER_KERNEL
