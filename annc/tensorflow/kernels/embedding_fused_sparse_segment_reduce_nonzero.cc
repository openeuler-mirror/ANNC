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
#include "absl/container/flat_hash_map.h"

using namespace tensorflow;

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

    OP_REQUIRES(context, input_tensor.dims() == 1,
                errors::InvalidArgument("Input data must be a vector"));
    OP_REQUIRES(context, slice_input.dims() == 2, errors::InvalidArgument("slice input must be 2-D"));
    OP_REQUIRES(context, begin.NumElements() == 2,  errors::InvalidArgument("begin must have 2 elements"));

    int64 num_indices = indices.dim_size(0);
    int32 col = begin.flat<int32>().data()[1];
    
    OP_REQUIRES(context, col >= 0 && col < slice_input.dim_size(1), 
                 errors::InvalidArgument("Column index out of range"));
    OP_REQUIRES(context, num_indices == slice_input.dim_size(0),
                errors::InvalidArgument("indices and slice_input.dim_zie(0) should have same size"));

    auto input_data = input_tensor.flat<float>();
    auto indices_vec = indices.vec<Tidx>();
    auto slice_input_mat = slice_input.matrix<int64>();

 	// Calculate max segment_id
    std::vector<int64> segment_ids(num_indices);
    int64 max_seg_id = 0;
    for (int64 i = 0; i < num_indices; ++i) {
      int64 seg_id = slice_input_mat(i, col);
      segment_ids[i] = seg_id;
      if (seg_id > max_seg_id) {
        max_seg_id = seg_id;
      }
    }

    const int64 batch_size = max_seg_id + 1;

    Tensor* output_shape = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({1}), &output_shape));
    output_shape->flat<int32>()(0) = static_cast<int32>(batch_size);

    std::vector<std::pair<int64, float>> results;
    int64 num_nonzero = 0;
    absl::flat_hash_map<int64, float> segment_sums;
    absl::flat_hash_map<int64, int32> segment_counts;
    std::vector<int64> segment_order;

    if (is_mean_) {
      for (int64 i = 0; i < num_indices; ++i) {
        const int64 seg_id = segment_ids[i];
        const Tidx data_row = indices_vec(i);
        
        if (segment_sums.find(seg_id) == segment_sums.end()) {
          segment_order.push_back(seg_id);
        }
        
        segment_sums[seg_id] += input_data(data_row);
        segment_counts[seg_id] += 1;
      }

      for (int64 seg_id : segment_order) {
        const int32_t count = segment_counts[seg_id];
        if (count > 0) {
          const float inv_count = 1.0f / static_cast<float>(count);
          float value = segment_sums[seg_id];
          if (value != 0) {
            results.push_back({seg_id, value * inv_count});
            num_nonzero++;
          }
        }
      }
    } else {
      for (int64 i = 0; i < num_indices; ++i) {
        const int64 seg_id = segment_ids[i];
        const Tidx data_row = indices_vec(i);
        
        if (segment_sums.find(seg_id) == segment_sums.end()) {
          segment_order.push_back(seg_id);
        }
        
        segment_sums[seg_id] += input_data(data_row);
      }

      for (int64 seg_id : segment_order) {
        float value = segment_sums[seg_id];
        if (value != 0) {
          results.push_back({seg_id, value});
          num_nonzero++;
        }
      }
    }
    Tensor* output_indices = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({num_nonzero, 1}),
                                            &output_indices));
    auto output_indices_data = output_indices->flat<int32>();

    Tensor* output_nonzero = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, TensorShape({num_nonzero}),
                                            &output_nonzero));
    auto output_nonzero_data = output_nonzero->flat<float>();
    for (int64 i = 0; i < num_nonzero; ++i) {
      output_indices_data(i) = static_cast<int32>(results[i].first);
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