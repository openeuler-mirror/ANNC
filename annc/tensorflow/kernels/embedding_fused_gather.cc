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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;

class KPFusedGather : public OpKernel {
 public:
  explicit KPFusedGather(OpKernelConstruction* context) : OpKernel(context) { }

  void Compute(OpKernelContext* context) override {
    const Tensor& data = context->input(0);
    const Tensor& keys = context->input(1);
    const Tensor& begin = context->input(2);
    VLOG(1) << "Embedding table size: " << data.shape().DebugString();
    VLOG(1) << "Input key shape: " << keys.shape().DebugString();
    VLOG(1) << "Slice begin value: " << begin.DebugString();

    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(keys.shape()), 
                errors::Internal("Input key must be 2D"));
    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(data.shape()),
                errors::Internal("Embedding table shape must be 2D"));
    OP_REQUIRES(context, begin.NumElements() == 2, errors::Internal("begin must be same as keys rank"));
    int32 col = begin.flat<int32>().data()[1];
    OP_REQUIRES(context, col < keys.dim_size(1), errors::Internal("slice cols out of keys range"));
    
    Tensor* out_indices = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                   1, TensorShape({static_cast<int32>(keys.dim_size(0))}), &out_indices));
    int32 *out_indices_data = out_indices->flat<int32>().data();

    auto keys_mat = keys.matrix<int64>();
    std::vector<int64_t> unique_values;
    std::unordered_map<int64_t, int32_t> value_to_index;
    int current_index = 0;
    for (int64_t i = 0; i < keys.dim_size(0); ++i) {
        auto it = value_to_index.find(keys_mat(i, col));
        if (it == value_to_index.end()) {
            value_to_index[keys_mat(i, col)] = current_index;
            unique_values.push_back(keys_mat(i, col));
            out_indices_data[i] = current_index;
            ++current_index;
        } else {
            out_indices_data[i] = it->second;
        }
    }

    Tensor* out_unique_value = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                   0, TensorShape({static_cast<int32>(unique_values.size())}), &out_unique_value));
    std::memcpy(out_unique_value->data(), unique_values.data(), unique_values.size() * sizeof(int64_t));

    Tensor* out_data = nullptr;
    int embedding_dims = data.dim_size(1);
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                   2, TensorShape({static_cast<int32>(unique_values.size()), embedding_dims}), &out_data));

    const float *data_mat = data.flat<float>().data();
    for (int64_t cur_row = 0; cur_row < unique_values.size(); ++cur_row) {
        int64_t idx = unique_values[cur_row];
        OP_REQUIRES(context, idx < data.dim_size(0), errors::Internal("idx out of table range"));
        const float* src = data_mat + idx * embedding_dims;
        float* dst = out_data->flat<float>().data() + cur_row * embedding_dims;
        std::memcpy(dst, src, embedding_dims * sizeof(float));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("KPFusedGather").Device(DEVICE_CPU),
                        KPFusedGather);