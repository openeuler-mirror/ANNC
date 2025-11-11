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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;

class KPFusedSparseDynamicStitchOp : public OpKernel {
public:
  explicit KPFusedSparseDynamicStitchOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& x = context->input(0);
    auto x_flat = x.flat<int64>();
    int64_t num_elems = x_flat.size();

    const int num_inputs = context->num_inputs();
    const int num_partitions = num_inputs - 1;
    OP_REQUIRES(context, num_partitions > 1, errors::InvalidArgument("num partitions must > 1"));
    int64_t output_stride = 0;
    std::vector<const float*> variables(num_partitions);
    std::vector<int64_t> variable_rows(num_partitions);
    for (int i = 1; i < num_inputs; ++i) {
      const Tensor& input_tensor = context->input(i);
      OP_REQUIRES(context, input_tensor.dims() == 2, errors::InvalidArgument("input dims must == 2"));
      if (i == 1) {
        output_stride = input_tensor.dim_size(1);
      } else {
        OP_REQUIRES(context, input_tensor.dim_size(1)  == output_stride,
                    errors::InvalidArgument("All inputs must have same second dimension"));
      }
      variables[i - 1] = context->input(i).flat<float>().data();
      variable_rows[i - 1] = input_tensor.dim_size(0);
    }

    OP_REQUIRES(context, output_stride > 0, errors::InvalidArgument("output_stride must > 0"));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({num_elems, output_stride}),
                                            &output_tensor));
    float* output = (float*)output_tensor->tensor_data().data();

    const size_t copy_size = output_stride * sizeof(float);

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    const int64 cost_per_unit = 120; // Actual single cycle execution time 
    auto work = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        const int64_t global_id = x_flat(i);
        const int64_t table_id = global_id % num_partitions;
        const int64_t row_id = global_id / num_partitions;

        OP_REQUIRES(context, row_id < variable_rows[table_id], errors::InvalidArgument(
          "row_id out of range."));

        std::memcpy(output + i * output_stride,
                    variables[table_id] + row_id * output_stride, copy_size);
      }
    };

    Shard(worker_threads->num_threads, worker_threads->workers, num_elems,
          cost_per_unit, work);
  }
};

REGISTER_KERNEL_BUILDER(Name("KPFusedSparseDynamicStitch").Device(DEVICE_CPU),
                        KPFusedSparseDynamicStitchOp);
