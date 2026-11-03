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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "Context/KernelContextUtils.h"

namespace {

// Parallelism is useful only after the operation is large enough to amortize
// worker-pool scheduling cost. The threshold is expressed in float elements.
constexpr int64_t kParallelThresholdElements = 1 << 20;
constexpr int kMaxParallelThreads = 8;

bool MultiplyFitsInt64(int64_t lhs, int64_t rhs, int64_t* product) {
  if (lhs < 0 || rhs < 0) {
    return false;
  }
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
    return false;
  }
  *product = lhs * rhs;
  return true;
}

bool FloatBufferFitsSizeT(int64_t elements) {
  return elements >= 0 &&
         static_cast<uint64_t>(elements) <=
             std::numeric_limits<size_t>::max() / sizeof(float);
}

void CopyAndPadSerial(const float* input_data, float* output_data,
                      int64_t copy_elements, int64_t padding_elements) {
  if (copy_elements != 0) {
    std::memcpy(output_data, input_data,
                static_cast<size_t>(copy_elements) * sizeof(float));
  }
  if (padding_elements != 0) {
    std::memset(output_data + copy_elements, 0,
                static_cast<size_t>(padding_elements) * sizeof(float));
  }
}

void CopyAndPad(annc::kernels::KernelContext& context, const float* input_data,
                float* output_data, int64_t input_rows, int64_t padding_rows,
                int64_t cols) {
  int64_t copy_elements = 0;
  int64_t padding_elements = 0;
  if (!MultiplyFitsInt64(input_rows, cols, &copy_elements) ||
      !MultiplyFitsInt64(padding_rows, cols, &padding_elements)) {
    // Shape validation in Compute normally catches this. Keep the helper
    // defensive for callers that use it independently.
    return;
  }

  if (padding_elements > std::numeric_limits<int64_t>::max() - copy_elements) {
    return;
  }
  const int64_t total_elements = copy_elements + padding_elements;
  // Keep small copies on the libc path; this also avoids worker-pool overhead.
  if (total_elements < kParallelThresholdElements) {
    CopyAndPadSerial(input_data, output_data, copy_elements, padding_elements);
    return;
  }

  auto workers = context.getThreadPool();
  const int64_t work_threads = total_elements / kParallelThresholdElements;
  const int requested_threads = static_cast<int>(std::min<int64_t>(
      {work_threads, workers.numThreads(), kMaxParallelThreads}));
  if (requested_threads > 1) {
    auto copy_and_zero = [&](int64_t begin, int64_t end) {
      if (begin < copy_elements) {
        const int64_t copy_end = std::min(end, copy_elements);
        std::memcpy(output_data + begin, input_data + begin,
                    static_cast<size_t>(copy_end - begin) * sizeof(float));
      }
      if (end > copy_elements) {
        const int64_t padding_begin = std::max(begin, copy_elements);
        std::memset(output_data + padding_begin, 0,
                    static_cast<size_t>(end - padding_begin) * sizeof(float));
      }
    };
    annc::kernels::Shard(requested_threads, workers, total_elements, 1,
                         copy_and_zero);
    return;
  }

  CopyAndPadSerial(input_data, output_data, copy_elements, padding_elements);
}

// Preserve TensorShape::DebugString diagnostics without a TensorFlow
// dependency.
std::string ShapeDebugString(const AnncTensorView& tensor) {
  std::string shape = "[";
  for (uint32_t i = 0; i < tensor.rank; ++i) {
    if (i != 0) shape += ',';
    shape += std::to_string(tensor.sizes[i]);
  }
  return shape + ']';
}

AnncStatusCode kpFusedEmbeddingPadding(annc::kernels::KernelContext& context,
                                       bool fast) {
  // Context slots follow the shared schema, rather than the TF op input order.
  const auto* origin_shape = context.getInput(0);
  const auto* input_rows = context.getInput(1);
  const auto* pack = context.getInput(2);
  const auto* input = context.getInput(3);
  const auto* reshape_sizes = context.getInput(4);
  if (!origin_shape || !input_rows || !pack || !input || !reshape_sizes)
    return context.status();

  if (origin_shape->rank != 1)
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        ("origin_shape dims must 1D, not " + ShapeDebugString(*origin_shape))
            .c_str());
  if (origin_shape->sizes[0] != 2)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         ("origin_shape NumElements must == 2, not " +
                          std::to_string(origin_shape->sizes[0]))
                             .c_str());
  if (input->rank != 2)
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        ("input dims must 2D, not " + ShapeDebugString(*input)).c_str());
  if (input_rows->rank != 0)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "input_rows must be a scalar");
  if (reshape_sizes->rank != 1)
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        ("sizes input must be 1-D, not " + ShapeDebugString(*reshape_sizes))
            .c_str());
  if (reshape_sizes->sizes[0] != 2)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "reshape_sizes NumElements must == 2");

  const int64_t origin_rows =
      annc::kernels::contextData<int64_t>(*origin_shape)[0];
  const int64_t input_rows_value =
      *annc::kernels::contextData<int32_t>(*input_rows);
  const int64_t reshape_cols =
      annc::kernels::contextData<int32_t>(*reshape_sizes)[1];
  const int64_t output_cols = input->sizes[1];
  if (!(origin_rows >= 0 && input_rows_value >= 0 &&
        input_rows_value <= origin_rows))
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        ("Pooling size(" + std::to_string(input_rows_value) +
         ") is greater than Input size(" + std::to_string(origin_rows) + ")")
            .c_str());
  const int64_t padding_rows = origin_rows - input_rows_value;
  if (input_rows_value != input->sizes[0])
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         ("input_rows(" + std::to_string(input_rows_value) +
                          ") is not equal to input rows(" +
                          std::to_string(input->sizes[0]) + ")")
                             .c_str());
  const int64_t output_rows = padding_rows + input_rows_value;
  if (reshape_cols <= 0)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT, "reshape_cols must > 0");
  if (annc::kernels::contextData<int32_t>(*reshape_sizes)[0] != -1)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT, "reshape[0] is not -1");
  // TensorFlow's scalar accessor enforces this precondition implicitly.
  if (pack->rank != 0)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT, "pack must be a scalar");
  if (*annc::kernels::contextData<int32_t>(*pack) != output_cols)
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        ("pack(" + std::to_string(*annc::kernels::contextData<int32_t>(*pack)) +
         ") is not equal to embedding dims")
            .c_str());
  if (padding_rows > std::numeric_limits<int32_t>::max())
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         ("padding rows exceed int32 output range: " +
                          std::to_string(padding_rows))
                             .c_str());

  int64_t output_elements = 0;
  if (!MultiplyFitsInt64(output_rows, output_cols, &output_elements))
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "output element count overflows int64");
  if (!FloatBufferFitsSizeT(output_elements))
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "output byte size exceeds size_t range");

  auto* output0 = context.allocateOutput<int32_t>(0, {});
  if (!output0) return context.status();
  *annc::kernels::contextData<int32_t>(*output0) =
      static_cast<int32_t>(padding_rows);
  if (output_elements % reshape_cols != 0)
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        ("padding cannot reshape to [-1, " + std::to_string(reshape_cols) + "]")
            .c_str());
  const int64_t reshape_rows = output_elements / reshape_cols;
  if (fast) {
    if (reshape_rows > std::numeric_limits<int32_t>::max())
      return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                           ("reshape rows exceed int32 output range: " +
                            std::to_string(reshape_rows))
                               .c_str());
    auto* output1 = context.allocateOutput<int32_t>(1, {});
    if (!output1) return context.status();
    *annc::kernels::contextData<int32_t>(*output1) =
        static_cast<int32_t>(reshape_rows);
    return context.status();
  }

  auto* output1 =
      context.allocateOutput<float>(1, {reshape_rows, reshape_cols});
  if (!output1) return context.status();
  float* output_data = annc::kernels::contextData<float>(*output1);
  const float* input_data = annc::kernels::contextData<float>(*input);
  CopyAndPad(context, input_data, output_data, input_rows_value, padding_rows,
             output_cols);
  return context.status();
}

AnncStatusCode kpFusedEmbeddingPaddingImpl(
    annc::kernels::KernelContext& context) {
  return kpFusedEmbeddingPadding(context, false);
}

AnncStatusCode kpFusedEmbeddingPaddingFastImpl(
    annc::kernels::KernelContext& context) {
  return kpFusedEmbeddingPadding(context, true);
}

}  // namespace
