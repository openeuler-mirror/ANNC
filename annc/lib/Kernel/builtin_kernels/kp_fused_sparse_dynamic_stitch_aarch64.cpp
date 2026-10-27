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

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "Context/KernelContextUtils.h"

namespace {

// Computation follows embedding_fused_sparse_dynamic_stitch_new.cc;
// only tensor access, allocation, scheduling and errors use Context services.
// TensorFlow's sharder uses this value as an estimate of the work for one
// input element. Keep the baseline estimate: a larger value for wide rows
// creates too many tiny shards because the table lookup is usually limited by
// random memory latency rather than by the number of copied floats.
constexpr int64_t kCostPerUnit = 120;

inline int64_t ShardCost(int64_t /*stride*/) { return kCostPerUnit; }

inline void RecordInvalidIndex(std::atomic<int64_t>* first_bad_index,
                               int64_t index) {
  int64_t expected = -1;
  first_bad_index->compare_exchange_strong(expected, index,
                                           std::memory_order_relaxed);
}

template <int Stride>
struct RowCopier {
  static inline void Copy(float* destination, const float* source) {
    __builtin_memcpy(destination, source, sizeof(float) * Stride);
  }
};

template <>
struct RowCopier<3> {
  static inline void Copy(float* destination, const float* source) {
    vst1_f32(destination, vld1_f32(source));
    destination[2] = source[2];
  }
};

template <>
struct RowCopier<4> {
  static inline void Copy(float* destination, const float* source) {
    vst1q_f32(destination, vld1q_f32(source));
  }
};

template <>
struct RowCopier<6> {
  static inline void Copy(float* destination, const float* source) {
    vst1q_f32(destination, vld1q_f32(source));
    vst1_f32(destination + 4, vld1_f32(source + 4));
  }
};

template <>
struct RowCopier<8> {
  static inline void Copy(float* destination, const float* source) {
    vst1q_f32(destination, vld1q_f32(source));
    vst1q_f32(destination + 4, vld1q_f32(source + 4));
  }
};

template <>
struct RowCopier<10> {
  static inline void Copy(float* destination, const float* source) {
    vst1q_f32(destination, vld1q_f32(source));
    vst1q_f32(destination + 4, vld1q_f32(source + 4));
    vst1_f32(destination + 8, vld1_f32(source + 8));
  }
};

template <>
struct RowCopier<16> {
  static inline void Copy(float* destination, const float* source) {
    vst1q_f32(destination, vld1q_f32(source));
    vst1q_f32(destination + 4, vld1q_f32(source + 4));
    vst1q_f32(destination + 8, vld1q_f32(source + 8));
    vst1q_f32(destination + 12, vld1q_f32(source + 12));
  }
};

template <int NumPartitions>
inline int64_t DecodeTable(int64_t global_id, int64_t* row_id) {
  if constexpr (NumPartitions == 8) {
    *row_id = global_id >> 3;
    return global_id & 7;
  } else {
    *row_id = global_id / NumPartitions;
    return global_id - *row_id * NumPartitions;
  }
}

template <int NumPartitions, int Stride>
void CopyFixedStride(const int64_t* indices, int64_t start, int64_t end,
                     const float* const* variables,
                     const int64_t* variable_rows, float* output,
                     std::atomic<int64_t>* first_bad_index) {
  for (int64_t i = start; i < end; ++i) {
    const int64_t global_id = indices[i];
    if (global_id < 0) {
      RecordInvalidIndex(first_bad_index, i);
      continue;
    }

    int64_t row_id;
    const int64_t table_id = DecodeTable<NumPartitions>(global_id, &row_id);
    if (row_id < 0 || row_id >= variable_rows[table_id]) {
      RecordInvalidIndex(first_bad_index, i);
      continue;
    }

    float* destination = output + i * Stride;
    const float* source = variables[table_id] + row_id * Stride;
    RowCopier<Stride>::Copy(destination, source);
  }
}

template <int NumPartitions>
void CopyFixedPartitionRuntimeStride(const int64_t* indices, int64_t start,
                                     int64_t end, const float* const* variables,
                                     const int64_t* variable_rows,
                                     int64_t stride, float* output,
                                     std::atomic<int64_t>* first_bad_index) {
  const size_t copy_size = static_cast<size_t>(stride) * sizeof(float);
  for (int64_t i = start; i < end; ++i) {
    const int64_t global_id = indices[i];
    if (global_id < 0) {
      RecordInvalidIndex(first_bad_index, i);
      continue;
    }

    int64_t row_id;
    const int64_t table_id = DecodeTable<NumPartitions>(global_id, &row_id);
    if (row_id < 0 || row_id >= variable_rows[table_id]) {
      RecordInvalidIndex(first_bad_index, i);
      continue;
    }

    std::memcpy(output + i * stride, variables[table_id] + row_id * stride,
                copy_size);
  }
}

void CopyRuntime(const int64_t* indices, int64_t start, int64_t end,
                 int num_partitions, const float* const* variables,
                 const int64_t* variable_rows, int64_t stride, float* output,
                 std::atomic<int64_t>* first_bad_index) {
  const size_t copy_size = static_cast<size_t>(stride) * sizeof(float);
  for (int64_t i = start; i < end; ++i) {
    const int64_t global_id = indices[i];
    if (global_id < 0) {
      RecordInvalidIndex(first_bad_index, i);
      continue;
    }

    const int64_t row_id = global_id / num_partitions;
    const int64_t table_id = global_id - row_id * num_partitions;
    if (row_id < 0 || row_id >= variable_rows[table_id]) {
      RecordInvalidIndex(first_bad_index, i);
      continue;
    }

    std::memcpy(output + i * stride, variables[table_id] + row_id * stride,
                copy_size);
  }
}

template <typename Work>
void RunSharded(annc::kernels::KernelContext* context, int64_t num_elems,
                int64_t stride, Work work) {
  auto workers = context->getThreadPool();
  annc::kernels::Shard(workers.numThreads(), workers, num_elems,
                       ShardCost(stride), work);
}

template <int NumPartitions, int Stride>
void RunFixedStride(annc::kernels::KernelContext* context, int64_t num_elems,
                    const int64_t* indices, const float* const* variables,
                    const int64_t* variable_rows, float* output,
                    std::atomic<int64_t>* first_bad_index) {
  auto work = [&](int64_t start, int64_t end) {
    CopyFixedStride<NumPartitions, Stride>(
        indices, start, end, variables, variable_rows, output, first_bad_index);
  };
  RunSharded(context, num_elems, Stride, work);
}

template <int NumPartitions>
void RunFixedPartitionRuntimeStride(annc::kernels::KernelContext* context,
                                    int64_t num_elems, const int64_t* indices,
                                    const float* const* variables,
                                    const int64_t* variable_rows,
                                    int64_t stride, float* output,
                                    std::atomic<int64_t>* first_bad_index) {
  auto work = [&](int64_t start, int64_t end) {
    CopyFixedPartitionRuntimeStride<NumPartitions>(
        indices, start, end, variables, variable_rows, stride, output,
        first_bad_index);
  };
  RunSharded(context, num_elems, stride, work);
}

void RunRuntime(annc::kernels::KernelContext* context, int64_t num_elems,
                const int64_t* indices, int num_partitions,
                const float* const* variables, const int64_t* variable_rows,
                int64_t stride, float* output,
                std::atomic<int64_t>* first_bad_index) {
  auto work = [&](int64_t start, int64_t end) {
    CopyRuntime(indices, start, end, num_partitions, variables, variable_rows,
                stride, output, first_bad_index);
  };
  RunSharded(context, num_elems, stride, work);
}

template <int NumPartitions>
void DispatchStride(annc::kernels::KernelContext* context, int64_t num_elems,
                    const int64_t* indices, const float* const* variables,
                    const int64_t* variable_rows, int64_t stride, float* output,
                    std::atomic<int64_t>* first_bad_index) {
  switch (stride) {
    case 3:
      RunFixedStride<NumPartitions, 3>(context, num_elems, indices, variables,
                                       variable_rows, output, first_bad_index);
      return;
    case 4:
      RunFixedStride<NumPartitions, 4>(context, num_elems, indices, variables,
                                       variable_rows, output, first_bad_index);
      return;
    case 6:
      RunFixedStride<NumPartitions, 6>(context, num_elems, indices, variables,
                                       variable_rows, output, first_bad_index);
      return;
    case 8:
      RunFixedStride<NumPartitions, 8>(context, num_elems, indices, variables,
                                       variable_rows, output, first_bad_index);
      return;
    case 10:
      RunFixedStride<NumPartitions, 10>(context, num_elems, indices, variables,
                                        variable_rows, output, first_bad_index);
      return;
    case 16:
      RunFixedStride<NumPartitions, 16>(context, num_elems, indices, variables,
                                        variable_rows, output, first_bad_index);
      return;
    default:
      RunFixedPartitionRuntimeStride<NumPartitions>(
          context, num_elems, indices, variables, variable_rows, stride, output,
          first_bad_index);
  }
}

void Dispatch(annc::kernels::KernelContext* context, int64_t num_elems,
              const int64_t* indices, int num_partitions,
              const float* const* variables, const int64_t* variable_rows,
              int64_t stride, float* output,
              std::atomic<int64_t>* first_bad_index) {
  switch (num_partitions) {
    case 8:
      DispatchStride<8>(context, num_elems, indices, variables, variable_rows,
                        stride, output, first_bad_index);
      return;
    case 12:
      DispatchStride<12>(context, num_elems, indices, variables, variable_rows,
                         stride, output, first_bad_index);
      return;
    case 36:
      DispatchStride<36>(context, num_elems, indices, variables, variable_rows,
                         stride, output, first_bad_index);
      return;
    default:
      RunRuntime(context, num_elems, indices, num_partitions, variables,
                 variable_rows, stride, output, first_bad_index);
  }
}

AnncStatusCode kp_fused_sparse_dynamic_stitch_impl(
    annc::kernels::KernelContext& context) {
  const auto declaredN = context.getI64Attr(0);
  if (!declaredN) return context.status();
  const int64_t n = static_cast<int64_t>(context.numInputs()) - 1;
  if (n <= 1 || n != *declaredN || n > std::numeric_limits<int>::max())
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "DynamicStitch input count does not match N > 1");
  const int num_partitions = static_cast<int>(n);
  const auto* x = context.getInput(0);
  if (!x) return context.status();
  if (x->element_type != ANNC_ELEMENT_TYPE_I64)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "DynamicStitch x must be int64");
  // Tensor::flat<int64>() in the reference accepts every dense input rank.
  int64_t num_elems = 1;
  for (uint32_t i = 0; i < x->rank; ++i) num_elems *= x->sizes[i];

  // Match the reference's InlinedVector<..., 36> without a TF dependency.
  std::array<const float*, 36> inline_variables;
  std::array<int64_t, 36> inline_rows;
  std::vector<const float*> heap_variables(num_partitions > 36 ? num_partitions
                                                               : 0);
  std::vector<int64_t> heap_rows(num_partitions > 36 ? num_partitions : 0);
  auto* variables =
      num_partitions > 36 ? heap_variables.data() : inline_variables.data();
  auto* variable_rows =
      num_partitions > 36 ? heap_rows.data() : inline_rows.data();
  int64_t output_stride = 0;
  for (int i = 0; i < num_partitions; ++i) {
    const auto* input = context.getInput(static_cast<uint32_t>(i) + 1);
    if (!input) return context.status();
    if (input->rank != 2 || input->element_type != ANNC_ELEMENT_TYPE_F32)
      return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                           "DynamicStitch variables must be rank-2 float32");
    if (i == 0)
      output_stride = input->sizes[1];
    else if (input->sizes[1] != output_stride)
      return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                           "DynamicStitch variable widths differ");
    variables[i] = annc::kernels::contextData<float>(*input);
    variable_rows[i] = input->sizes[0];
  }
  if (output_stride <= 0)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "DynamicStitch output width must be positive");
  auto* output = context.allocateOutput<float>(0, {num_elems, output_stride});
  if (!output) return context.status();
  std::atomic<int64_t> first_bad_index(-1);
  Dispatch(&context, num_elems, annc::kernels::contextData<int64_t>(*x),
           num_partitions, variables, variable_rows, output_stride,
           annc::kernels::contextData<float>(*output), &first_bad_index);
  if (context.status() != ANNC_STATUS_OK) return context.status();
  if (first_bad_index.load(std::memory_order_relaxed) >= 0)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "DynamicStitch row index out of range");
  return context.status();
}

}  // namespace
