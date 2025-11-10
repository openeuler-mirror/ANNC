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

#include <arm_neon.h>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;

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
    const Tensor& begin_1 = context->input(4);

    OP_REQUIRES(context, input_tensor.dims() == 2, errors::InvalidArgument("input must be 2-D"));
    OP_REQUIRES(context, slice_input.dims() == 2, errors::InvalidArgument("slice input must be 2-D"));
    OP_REQUIRES(context, begin.NumElements() == 2,  errors::InvalidArgument("begin must have 2 elements"));
    OP_REQUIRES(context, begin_1.NumElements() == 1, errors::InvalidArgument("begin_1 must have 1 element"));
    int64_t num_indices = indices.dim_size(0);
    int64_t embedding_size = input_tensor.dim_size(1);
    int32 col = begin.flat<int32>().data()[1];
    int32 out_dim = static_cast<int32>(begin_1.flat<int32>()(0));

    OP_REQUIRES(context, col >= 0 && col < slice_input.dim_size(1), 
                 errors::InvalidArgument("Column index out of range"));
    OP_REQUIRES(context, num_indices == slice_input.dim_size(0),
                errors::InvalidArgument("indices and slice_input.dim_zie(0) should have same size"));

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
    if (out_dim == 0)
      slice_out->scalar<int32>()() = batch_size;
    else slice_out->scalar<int32>()() = embedding_size;

    auto output_data = output->matrix<float>().data();

    if (is_mean_) {
      Tensor counts(DT_INT32, TensorShape({batch_size}));
      counts.flat<int32>().setZero();
      auto counts_vec = counts.flat<int32>();

      for (int64 i = 0; i < num_indices; ++i) {
        const int64 seg_id = slice_input_mat(i, col);
        const Tidx data_row = indices_vec(i);
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
        const int64 seg_id = slice_input_mat(i, col);
        const Tidx data_row = indices_vec(i);

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