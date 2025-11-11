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
#include <algorithm>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;

class KPFusedSparseSelect : public OpKernel {
public:
  explicit KPFusedSparseSelect(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_a = context->input(0);
    const Tensor& input_b = context->input(1);
    const Tensor& input_c = context->input(2);
    const Tensor& greater = context->input(3); 
    const Tensor& equal1 = context->input(4);
    const Tensor& equal2 = context->input(5);
    const Tensor& equal3 = context->input(6);

    int32_t equal1_val = equal1.flat<int32_t>()(0);
    int32_t equal2_val = equal2.flat<int32_t>()(0);
    int32_t equal3_val = equal3.flat<int32_t>()(0);
    VLOG(1) << "equal1_val: " << equal1_val;
    VLOG(1) << "equal2_val: " << equal2_val;
    VLOG(1) << "equal3_val: " << equal3_val;

    int32_t greater_val = greater.flat<int32_t>()(0);
    auto a_flat = input_a.flat<int32_t>();
    auto b_flat = input_b.flat<int32_t>();
    auto c_flat = input_c.flat<int32_t>();
    VLOG(1) << "input_a shape: " << input_a.shape().DebugString();
    VLOG(1) << "input_b shape: " << input_b.shape().DebugString();
    VLOG(1) << "input_c shape: " << input_c.shape().DebugString();
    OP_REQUIRES(context, input_a.NumElements() == input_b.NumElements(),
                errors::InvalidArgument("Input num elements of a and b must match"));
    OP_REQUIRES(context, input_a.NumElements() == input_c.NumElements(),
                errors::InvalidArgument("Input num elements of a and c must match"));
    auto N = input_a.NumElements();

    Eigen::TensorMap<Eigen::Tensor<const int32_t, 2, Eigen::RowMajor>> a_reshaped_tensor(a_flat.data(), N, 1);
    Eigen::TensorMap<Eigen::Tensor<const int32_t, 2, Eigen::RowMajor>> b_reshaped_tensor(b_flat.data(), N, 1);
    Eigen::TensorMap<Eigen::Tensor<const int32_t, 2, Eigen::RowMajor>> c_reshaped_tensor(c_flat.data(), N, 1);

    Tensor* output_x = nullptr;
    Tensor* output_y = nullptr;
    Tensor* output_w = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N, 1}), &output_x));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({N, 1}), &output_y));
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({N, 2}), &output_w));

    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> out_x(
        output_x->flat<float>().data(),
        output_x->dim_size(0),
        output_x->dim_size(1)
    );

    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> out_y(
        output_y->flat<float>().data(),
        output_y->dim_size(0),
        output_y->dim_size(1)
    );

    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> out_w(
        output_w->flat<float>().data(),
        output_w->dim_size(0),
        output_w->dim_size(1)
    );

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    const int64 cost_per_unit = std::max(N / worker_threads->num_threads, int64(10));
    
    auto work = [&](int64 start, int64 end) {
      for (int64 i = start; i < end; i++) {
        // Greater(bool)+Cast.2406(float) --> 1.0f / 0.0f
        float a_greater = (a_reshaped_tensor(i, 0) > greater_val) ? 1.0f : 0.0f;
        float res_equal1 = (b_reshaped_tensor(i, 0) == equal1_val) ? 1.0f : a_greater;  // Fill.2409-->1.0f
        float res_equal2 = (b_reshaped_tensor(i, 0) == equal2_val) ? 1.0f : res_equal1;  // Fill.2409-->1.0f
        out_x(i, 0) = a_reshaped_tensor(i, 0);  // Reshape.2401
        out_y(i, 0) = res_equal2;
        out_w(i, 0) = res_equal2;  // Mul.2419 硬编码 1.0f * input
        out_w(i, 1) = 1.0f;  // select_2427被消除，直接使用Fill.2422-->1.0f
      }
    };
    Shard(worker_threads->num_threads, worker_threads->workers, N, cost_per_unit, work);
  }
};

REGISTER_KERNEL_BUILDER(Name("KPFusedSparseSelect").Device(DEVICE_CPU),
                        KPFusedSparseSelect);