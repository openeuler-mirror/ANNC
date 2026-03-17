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

    const int input_dims = input_tensor.dims();
    OP_REQUIRES(context, input_dims == 1 || input_dims == 2,
                errors::InvalidArgument("Input data must be a 1-D vector or 2-D matrix"));
    OP_REQUIRES(context, slice_input.dims() == 2, errors::InvalidArgument("slice input must be 2-D"));
    OP_REQUIRES(context, begin.NumElements() == 2,  errors::InvalidArgument("begin must have 2 elements"));

    int64 num_indices = indices.dim_size(0);
    int32 col = begin.flat<int32>().data()[1];

    OP_REQUIRES(context, col >= 0 && col < slice_input.dim_size(1),
                 errors::InvalidArgument("Column index out of range"));
    OP_REQUIRES(context, num_indices == slice_input.dim_size(0),
                errors::InvalidArgument("indices and slice_input.dim_size(0) should have same size"));

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

    // 获取线程池
    auto* worker_threads = context->device()->tensorflow_cpu_worker_threads();
    const int num_threads = worker_threads ? worker_threads->num_threads : 1;

    if (input_dims == 1) {
      // -------------------------------------------------------
      // 1-D input: each segment reduces to a scalar
      // output_shape: [batch_size]
      // output_indices: [num_nonzero, 1]  (segment_id)
      // output_nonzero: [num_nonzero]
      // -------------------------------------------------------
      auto input_data = input_tensor.flat<float>();

      Tensor* output_shape = nullptr;
      OP_REQUIRES_OK(
          context, context->allocate_output(0, TensorShape({1}), &output_shape));
      output_shape->flat<int32>()(0) = static_cast<int32>(batch_size);

      // 优化：使用map管理索引 + 连续内存存储
      absl::flat_hash_map<int64, int64> seg_id_to_idx;
      seg_id_to_idx.reserve(num_indices / 4 + 1);
      
      std::vector<float> segment_sums_data;
      std::vector<int32_t> segment_counts_data;
      std::vector<int64> segment_order;

      if (is_mean_) {
        // 聚合阶段 - 单线程（reduce维度）
        for (int64 i = 0; i < num_indices; ++i) {
          const int64 seg_id = segment_ids[i];
          const Tidx data_row = indices_vec(i);
          
          auto it = seg_id_to_idx.find(seg_id);
          int64 idx;
          if (it == seg_id_to_idx.end()) {
            idx = segment_order.size();
            seg_id_to_idx.emplace(seg_id, idx);
            segment_order.push_back(seg_id);
            segment_sums_data.push_back(0.0f);
            segment_counts_data.push_back(0);
          } else {
            idx = it->second;
          }
          
          segment_sums_data[idx] += input_data(data_row);
          segment_counts_data[idx]++;
        }
        
        // 预计算逆元
        std::vector<float> inv_counts(segment_order.size());
        for (size_t s = 0; s < segment_order.size(); ++s) {
          inv_counts[s] = (segment_counts_data[s] > 0) ? 
              (1.0f / static_cast<float>(segment_counts_data[s])) : 0.0f;
        }
        
        // 统计非零数量
        int64 num_nonzero = 0;
        for (size_t s = 0; s < segment_order.size(); ++s) {
          float val = segment_sums_data[s] * inv_counts[s];
          if (val != 0.0f) num_nonzero++;
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
        
        // 直接填充输出 - 无中间容器
        int64 idx = 0;
        for (size_t s = 0; s < segment_order.size(); ++s) {
          float val = segment_sums_data[s] * inv_counts[s];
          if (val != 0.0f) {
            output_indices_data(idx) = static_cast<int32>(segment_order[s]);
            output_nonzero_data(idx) = val;
            idx++;
          }
        }
      } else {
        // Sum模式
        for (int64 i = 0; i < num_indices; ++i) {
          const int64 seg_id = segment_ids[i];
          const Tidx data_row = indices_vec(i);
          
          auto it = seg_id_to_idx.find(seg_id);
          int64 idx;
          if (it == seg_id_to_idx.end()) {
            idx = segment_order.size();
            seg_id_to_idx.emplace(seg_id, idx);
            segment_order.push_back(seg_id);
            segment_sums_data.push_back(0.0f);
          } else {
            idx = it->second;
          }
          
          segment_sums_data[idx] += input_data(data_row);
        }
        
        // 统计非零数量
        int64 num_nonzero = 0;
        for (size_t s = 0; s < segment_order.size(); ++s) {
          if (segment_sums_data[s] != 0.0f) num_nonzero++;
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
        
        // 直接填充输出
        int64 idx = 0;
        for (size_t s = 0; s < segment_order.size(); ++s) {
          float val = segment_sums_data[s];
          if (val != 0.0f) {
            output_indices_data(idx) = static_cast<int32>(segment_order[s]);
            output_nonzero_data(idx) = val;
            idx++;
          }
        }
      }

    } else {
      // -------------------------------------------------------
      // 2-D input: each segment reduces to a vector of size embed_dim
      // output_shape: [batch_size, embed_dim]
      // output_indices: [num_nonzero, 2]  (segment_id, dim_index)
      // output_nonzero: [num_nonzero]
      // -------------------------------------------------------
      const int64 embed_dim = input_tensor.dim_size(1);
      auto input_data = input_tensor.matrix<float>();

      Tensor* output_shape = nullptr;
      OP_REQUIRES_OK(
          context, context->allocate_output(0, TensorShape({2}), &output_shape));
      output_shape->flat<int32>()(0) = static_cast<int32>(batch_size);
      output_shape->flat<int32>()(1) = static_cast<int32>(embed_dim);

      // 优化：map管理索引 + 连续内存存储数据
      absl::flat_hash_map<int64, int64> seg_id_to_idx;
      seg_id_to_idx.reserve(num_indices / 4 + 1);
      
      std::vector<float> segment_sums_data;
      std::vector<int32_t> segment_counts_data;
      std::vector<int64> segment_order;

      // ========== 阶段1：单线程聚合（Reduce维度，避免数据竞争）==========
      for (int64 i = 0; i < num_indices; ++i) {
        const int64 seg_id = segment_ids[i];
        const Tidx data_row = indices_vec(i);
        
        auto it = seg_id_to_idx.find(seg_id);
        int64 idx;
        if (it == seg_id_to_idx.end()) {
          idx = segment_order.size();
          seg_id_to_idx.emplace(seg_id, idx);
          segment_order.push_back(seg_id);
          segment_sums_data.resize((idx + 1) * embed_dim, 0.0f);
          if (is_mean_) segment_counts_data.push_back(0);
        } else {
          idx = it->second;
        }
        
        float* sum_ptr = &segment_sums_data[idx * embed_dim];
        
        // SIMD 加速聚合
        int64 d = 0;
        for (; d + 4 <= embed_dim; d += 4) {
          float32x4_t input_vec = vld1q_f32(&input_data(data_row, d));
          float32x4_t sum_vec = vld1q_f32(&sum_ptr[d]);
          sum_vec = vaddq_f32(sum_vec, input_vec);
          vst1q_f32(&sum_ptr[d], sum_vec);
        }
        // 处理剩余元素
        for (; d < embed_dim; ++d) {
          sum_ptr[d] += input_data(data_row, d);
        }
        
        if (is_mean_) {
          segment_counts_data[idx]++;
        }
      }
      
      const int64 num_unique_segments = segment_order.size();
      
      // 预计算逆元
      std::vector<float> inv_counts(num_unique_segments);
      if (is_mean_) {
        for (int64 s = 0; s < num_unique_segments; ++s) {
          inv_counts[s] = (segment_counts_data[s] > 0) ? 
              (1.0f / static_cast<float>(segment_counts_data[s])) : 0.0f;
        }
      }
      
      // ========== 阶段2：多线程统计非零数量（非Reduce维度）==========
      std::vector<int64> segment_nz_counts(num_unique_segments, 0);
      
      if (num_unique_segments >= 16 && num_threads > 1) {
        auto count_work = [&](int64 start_seg, int64 end_seg) {
          for (int64 s = start_seg; s < end_seg; ++s) {
            float* sum_ptr = &segment_sums_data[s * embed_dim];
            float inv_count = is_mean_ ? inv_counts[s] : 1.0f;
            int64 count = 0;
            for (int64 d = 0; d < embed_dim; ++d) {
              if (sum_ptr[d] * inv_count != 0.0f) count++;
            }
            segment_nz_counts[s] = count;
          }
        };
        
        Shard(num_threads, worker_threads->workers, num_unique_segments,
              num_unique_segments * embed_dim / num_threads, count_work);
      } else {
        for (int64 s = 0; s < num_unique_segments; ++s) {
          float* sum_ptr = &segment_sums_data[s * embed_dim];
          float inv_count = is_mean_ ? inv_counts[s] : 1.0f;
          int64 count = 0;
          for (int64 d = 0; d < embed_dim; ++d) {
            if (sum_ptr[d] * inv_count != 0.0f) count++;
          }
          segment_nz_counts[s] = count;
        }
      }
      
      // 计算前缀和
      std::vector<int64> segment_offsets(num_unique_segments + 1, 0);
      for (int64 s = 0; s < num_unique_segments; ++s) {
        segment_offsets[s + 1] = segment_offsets[s] + segment_nz_counts[s];
      }
      int64 total_nz = segment_offsets[num_unique_segments];
      
      // 分配输出 Tensor
      Tensor* output_indices = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(1, TensorShape({total_nz, 2}),
                                              &output_indices));
      auto output_indices_data = output_indices->matrix<int32>();

      Tensor* output_nonzero = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(2, TensorShape({total_nz}),
                                              &output_nonzero));
      auto output_nonzero_data = output_nonzero->flat<float>();
      
      // ========== 阶段3：多线程填充输出（非Reduce维度）==========
      if (num_unique_segments >= 16 && num_threads > 1) {
        auto output_work = [&](int64 start_seg, int64 end_seg) {
          for (int64 s = start_seg; s < end_seg; ++s) {
            int64 seg_id = segment_order[s];
            float* sum_ptr = &segment_sums_data[s * embed_dim];
            float inv_count = is_mean_ ? inv_counts[s] : 1.0f;
            int64 out_idx = segment_offsets[s];
            
            for (int64 d = 0; d < embed_dim; ++d) {
              float val = sum_ptr[d] * inv_count;
              if (val != 0.0f) {
                output_indices_data(out_idx, 0) = static_cast<int32>(seg_id);
                output_indices_data(out_idx, 1) = static_cast<int32>(d);
                output_nonzero_data(out_idx) = val;
                out_idx++;
              }
            }
          }
        };
        
        Shard(num_threads, worker_threads->workers, num_unique_segments,
              num_unique_segments * embed_dim / num_threads, output_work);
      } else {
        int64 idx = 0;
        for (int64 s = 0; s < num_unique_segments; ++s) {
          int64 seg_id = segment_order[s];
          float* sum_ptr = &segment_sums_data[s * embed_dim];
          float inv_count = is_mean_ ? inv_counts[s] : 1.0f;
          
          for (int64 d = 0; d < embed_dim; ++d) {
            float val = sum_ptr[d] * inv_count;
            if (val != 0.0f) {
              output_indices_data(idx, 0) = static_cast<int32>(seg_id);
              output_indices_data(idx, 1) = static_cast<int32>(d);
              output_nonzero_data(idx) = val;
              idx++;
            }
          }
        }
      }
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
