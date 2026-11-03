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

#include "Context/KernelContextUtils.h"

namespace {

bool multiplyFits(int64_t lhs, int64_t rhs, int64_t *product) {
  if (lhs < 0 || rhs < 0 ||
      (lhs && rhs > std::numeric_limits<int64_t>::max() / lhs))
    return false;
  *product = lhs * rhs;
  return true;
}

inline void ZeroPadding(float *output, int64_t count) {
  if (count == 1) {
    output[0] = 0.0f;
  } else if (count != 0) {
    std::memset(output, 0, static_cast<size_t>(count) * sizeof(float));
  }
}

template <int64_t EmbeddingDim, typename Tindices1, typename Tindices2>
static AnncStatusCode GatherOneEmbeddingPerRow(
    const float *params_data, const Tindices1 *indices1_data,
    const Tindices2 *indices2_data, int64_t row_count, int64_t output_stride,
    int64_t pack_const, float *output, int max_parallelism,
    const annc::kernels::KernelThreadPool &workers) {
  auto work = [&](int64_t start_row, int64_t end_row) {
    for (int64_t row = start_row; row < end_row; ++row) {
      const int64_t idx2 = static_cast<int64_t>(indices2_data[row]);
      const int64_t idx1 = static_cast<int64_t>(indices1_data[idx2]);
      float *dst = output + row * output_stride;
      std::memcpy(dst, params_data + idx1 * EmbeddingDim,
                  EmbeddingDim * sizeof(float));
      ZeroPadding(dst + EmbeddingDim, pack_const);
    }
  };
  return annc::kernels::Shard(max_parallelism, workers, row_count,
                              std::max<int64_t>(1, output_stride), work);
}

template <typename T1, typename T2>
AnncStatusCode kp_fused_action_id_gather_typed(
    annc::kernels::KernelContext &context) {
  const auto *params = context.getInput(0);
  const auto *indices1 = context.getInput(1);
  const auto *indices2 = context.getInput(2);
  const auto *packDim = context.getInput(3);
  const auto *pack = context.getInput(4);
  if (!params || !indices1 || !indices2 || !packDim || !pack)
    return context.status();
  if (params->rank != 2 || (indices1->rank != 1 && indices1->rank != 2) ||
      (indices2->rank != 1 && indices2->rank != 2) || packDim->rank != 0 ||
      pack->rank != 0)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "ActionIdGather input rank mismatch");

  const int64_t paramsRows = params->sizes[0];
  const int64_t embeddingDim = params->sizes[1];
  const int64_t indices1Rows = indices1->sizes[0];
  const int64_t indices1Cols = indices1->rank == 1 ? 1 : indices1->sizes[1];
  int64_t indices2Count = indices2->sizes[0];
  if (indices2->rank == 2 &&
      !multiplyFits(indices2->sizes[0], indices2->sizes[1], &indices2Count))
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "ActionIdGather indices2 shape overflow");
  const int64_t packSize = *annc::kernels::contextData<int32_t>(*packDim);
  const int64_t packConst = *annc::kernels::contextData<int32_t>(*pack);
  if (paramsRows < 0 || embeddingDim < 0 || indices1Rows < 0 ||
      indices1Cols < 0 || indices2Count < 0 || packSize <= 0 || packConst < 0)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "ActionIdGather shape or pack value is invalid");

  const T1 *indices1Data = annc::kernels::contextData<T1>(*indices1);
  const T2 *indices2Data = annc::kernels::contextData<T2>(*indices2);
  int64_t indices1Count = 0;
  if (!multiplyFits(indices1Rows, indices1Cols, &indices1Count))
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "ActionIdGather indices1 shape overflow");
  for (int64_t i = 0; i < indices1Count; ++i) {
    const int64_t index = static_cast<int64_t>(indices1Data[i]);
    if (index < 0 || index >= paramsRows)
      return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                           "ActionIdGather indices1 index out of range");
  }
  for (int64_t i = 0; i < indices2Count; ++i) {
    const int64_t index = static_cast<int64_t>(indices2Data[i]);
    if (index < 0 || index >= indices1Rows)
      return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                           "ActionIdGather indices2 index out of range");
  }

  int64_t vectorCount = 0;
  int64_t gatheredElements = 0;
  if (!multiplyFits(indices2Count, indices1Cols, &vectorCount) ||
      !multiplyFits(vectorCount, embeddingDim, &gatheredElements) ||
      gatheredElements % packSize != 0)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "ActionIdGather output shape overflow");
  const int64_t packedCols = gatheredElements / packSize;
  int64_t outputCols = 0;
  if (packedCols > std::numeric_limits<int64_t>::max() - packConst)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "ActionIdGather output shape overflow");
  outputCols = packedCols + packConst;
  auto *output = context.allocateOutput<float>(0, {packSize, outputCols});
  if (!output) return context.status();

  float *outputData = annc::kernels::contextData<float>(*output);
  const float *paramsData = annc::kernels::contextData<float>(*params);
  const int64_t outputStride = outputCols;
  if (!vectorCount || !embeddingDim) {
    int64_t outputElements = 0;
    if (!multiplyFits(packSize, outputStride, &outputElements))
      return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                           "ActionIdGather output size overflow");
    if (outputElements && !outputData)
      return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                           "ActionIdGather output data is null");
    if (outputElements)
      std::memset(outputData, 0,
                  static_cast<size_t>(outputElements) * sizeof(float));
    return context.status();
  }
  if (!paramsData || !outputData)
    return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                         "ActionIdGather data is null");

  try {
    auto workers = context.getThreadPool();
    const int maxParallelism = workers.numThreads();
    const size_t embedding_bytes =
        static_cast<size_t>(embeddingDim) * sizeof(float);
    const int64_t cost_per_row = std::max<int64_t>(1, outputStride);

    // The graph rewriter usually produces packSize == indices2.NumElements()
    // and indices1Cols == 1. Keep that path free of division and modulo in its
    // inner loop.
    if (vectorCount % packSize == 0) {
      const int64_t vectors_per_row = vectorCount / packSize;
      if (indices1Cols == 1) {
        if (vectors_per_row == 1) {
          switch (embeddingDim) {
            case 32:
              return GatherOneEmbeddingPerRow<32>(
                  paramsData, indices1Data, indices2Data, packSize,
                  outputStride, packConst, outputData, maxParallelism, workers);
            case 64:
              return GatherOneEmbeddingPerRow<64>(
                  paramsData, indices1Data, indices2Data, packSize,
                  outputStride, packConst, outputData, maxParallelism, workers);
            default:
              break;
          }
        }

        auto work = [&](int64_t start_row, int64_t end_row) {
          for (int64_t row = start_row; row < end_row; ++row) {
            float *dst = outputData + row * outputStride;
            const int64_t first_vector = row * vectors_per_row;
            for (int64_t vector = 0; vector < vectors_per_row; ++vector) {
              const int64_t gather_index = first_vector + vector;
              const int64_t idx2 =
                  static_cast<int64_t>(indices2Data[gather_index]);
              const int64_t idx1 = static_cast<int64_t>(indices1Data[idx2]);
              std::memcpy(dst + vector * embeddingDim,
                          paramsData + idx1 * embeddingDim, embedding_bytes);
            }
            ZeroPadding(dst + packedCols, packConst);
          }
        };
        return annc::kernels::Shard(maxParallelism, workers, packSize,
                                    cost_per_row, work);
      }

      auto work = [&](int64_t start_row, int64_t end_row) {
        for (int64_t row = start_row; row < end_row; ++row) {
          float *dst = outputData + row * outputStride;
          const int64_t first_vector = row * vectors_per_row;
          for (int64_t vector = 0; vector < vectors_per_row; ++vector) {
            const int64_t gather_index = first_vector + vector;
            const int64_t index1_col = gather_index % indices1Cols;
            const int64_t index2_pos = gather_index / indices1Cols;
            const int64_t idx2 = static_cast<int64_t>(indices2Data[index2_pos]);
            const int64_t idx1 = static_cast<int64_t>(
                indices1Data[idx2 * indices1Cols + index1_col]);
            std::memcpy(dst + vector * embeddingDim,
                        paramsData + idx1 * embeddingDim, embedding_bytes);
          }
          ZeroPadding(dst + packedCols, packConst);
        }
      };
      return annc::kernels::Shard(maxParallelism, workers, packSize,
                                  cost_per_row, work);
    }

    // Generic path for a pack size that splits one or more embedding vectors.
    // Copy the largest contiguous source segment that fits in each output row.
    auto work = [&](int64_t start_row, int64_t end_row) {
      for (int64_t row = start_row; row < end_row; ++row) {
        float *dst = outputData + row * outputStride;
        ZeroPadding(dst + packedCols, packConst);

        int64_t source_offset = row * packedCols;
        int64_t destination_offset = 0;
        int64_t remaining = packedCols;
        while (remaining != 0) {
          const int64_t vector = source_offset / embeddingDim;
          const int64_t vector_offset = source_offset % embeddingDim;
          const int64_t index1_col = vector % indices1Cols;
          const int64_t index2_pos = vector / indices1Cols;
          const int64_t idx2 = static_cast<int64_t>(indices2Data[index2_pos]);
          const int64_t idx1 = static_cast<int64_t>(
              indices1Data[idx2 * indices1Cols + index1_col]);
          const int64_t copy_count =
              std::min(embeddingDim - vector_offset, remaining);
          std::memcpy(dst + destination_offset,
                      paramsData + idx1 * embeddingDim + vector_offset,
                      static_cast<size_t>(copy_count) * sizeof(float));
          source_offset += copy_count;
          destination_offset += copy_count;
          remaining -= copy_count;
        }
      }
    };
    return annc::kernels::Shard(maxParallelism, workers, packSize, cost_per_row,
                                work);
  } catch (...) {
    return context.error(ANNC_STATUS_INTERNAL, "ActionIdGather failed");
  }
}

AnncStatusCode kp_fused_action_id_gather_impl(
    annc::kernels::KernelContext &context) {
  const auto *indices1 = context.getInput(1);
  const auto *indices2 = context.getInput(2);
  if (!indices1 || !indices2) return context.status();
  if (indices1->element_type == ANNC_ELEMENT_TYPE_I32) {
    if (indices2->element_type == ANNC_ELEMENT_TYPE_I32)
      return kp_fused_action_id_gather_typed<int32_t, int32_t>(context);
    if (indices2->element_type == ANNC_ELEMENT_TYPE_I64)
      return kp_fused_action_id_gather_typed<int32_t, int64_t>(context);
  } else if (indices1->element_type == ANNC_ELEMENT_TYPE_I64) {
    if (indices2->element_type == ANNC_ELEMENT_TYPE_I32)
      return kp_fused_action_id_gather_typed<int64_t, int32_t>(context);
    if (indices2->element_type == ANNC_ELEMENT_TYPE_I64)
      return kp_fused_action_id_gather_typed<int64_t, int64_t>(context);
  }
  return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                       "ActionIdGather indices must be i32 or i64");
}

}  // namespace
