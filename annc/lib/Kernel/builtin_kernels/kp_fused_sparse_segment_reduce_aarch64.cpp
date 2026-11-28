#include <cstdint>
#include <exception>
#include <vector>

#include "Context/KernelContextUtils.h"

namespace {

// Fused sparse segment reduce, ported from the 812 KPFusedSparseSegmentReduce
// TF kernel, adapted to the context ABI (outputs are allocated through
// the context services).
//
//   seg_id = keys[i, col]  (col = begin[1], runtime)
//   batch_size = max(seg_id) + 1
//   slot 0: output[seg_id] = SUM/MEAN over data[indices[i]]
//   slot 1: slice_output = (begin_1[0] == 0) ? batch_size : embedding_size
template <typename Tidx, bool IsMean>
AnncStatusCode kpFusedSparseSegmentReduceTyped(
    annc::kernels::KernelContext& context) {
  const auto* keys = context.getInput(0);
  const auto* begin = context.getInput(1);
  const auto* data = context.getInput(2);
  const auto* indices = context.getInput(3);
  const auto* begin_1 = context.getInput(4);
  if (!keys || !begin || !data || !indices || !begin_1) return context.status();
  const Tidx* indices_data = annc::kernels::contextData<Tidx>(*indices);
  const int64_t num_indices = indices->sizes[0];

  if (!indices_data) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }

  const int64_t num_rows = data->sizes[0];
  const int64_t embed = data->sizes[1];
  const int64_t num_keys = keys->sizes[0];
  const int64_t key_width = keys->sizes[1];

  if (num_rows < 0 || embed <= 0 || num_keys < 0 || key_width < 0 ||
      num_indices < 0) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }
  if (begin->sizes[0] != 2 || begin_1->sizes[0] != 1) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }
  const int32_t col = annc::kernels::contextData<int32_t>(*begin)[1];
  const int32_t out_dim = annc::kernels::contextData<int32_t>(*begin_1)[0];
  if (col < 0 || col >= key_width) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }
  if (num_indices != num_keys) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }
  if (data->offset < 0 || keys->offset < 0 || begin->offset < 0 ||
      begin_1->offset < 0) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }
  if ((num_rows > 0 && !data->aligned) || (num_keys > 0 && !keys->aligned) ||
      !begin->aligned || !begin_1->aligned) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }
  if (data->strides[0] != embed || data->strides[1] != 1 ||
      keys->strides[0] != key_width || keys->strides[1] != 1 ||
      begin->strides[0] != 1 || begin_1->strides[0] != 1) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }

  const int64_t* keys_data = annc::kernels::contextData<int64_t>(*keys);
  const float* data_ptr = annc::kernels::contextData<float>(*data);

  try {
    // batch_size = max(seg_id) + 1 (value-dependent output shape; context
    // services allocate it after this scan).
    int64_t max_seg = -1;
    for (int64_t i = 0; i < num_keys; ++i) {
      const int64_t seg = keys_data[i * key_width + col];
      if (seg < 0) {
        return ANNC_STATUS_INVALID_ARGUMENT;
      }
      if (seg > max_seg) max_seg = seg;
    }
    const int64_t batch_size = max_seg + 1;

    auto* output = context.allocateOutput<float>(
        0, {batch_size, embed}, ANNC_ALLOCATION_ZERO);
    if (!output) return context.status();
    auto* slice_output = context.allocateOutput<int32_t>(1, {});
    if (!slice_output) return context.status();

    float* out = annc::kernels::contextData<float>(*output);
    std::vector<int32_t> counts;
    if (IsMean) {
      counts.assign(static_cast<size_t>(batch_size), 0);
    }

    for (int64_t i = 0; i < num_indices; ++i) {
      const int64_t seg = keys_data[i * key_width + col];
      const int64_t data_row = static_cast<int64_t>(indices_data[i]);
      if (data_row < 0 || data_row >= num_rows) {
        return ANNC_STATUS_INVALID_ARGUMENT;
      }
      if (IsMean) counts[static_cast<size_t>(seg)] += 1;
      float* out_row = out + seg * embed;
      const float* in_row = data_ptr + data_row * embed;
      for (int64_t j = 0; j < embed; ++j) {
        out_row[j] += in_row[j];
      }
    }
    if (IsMean) {
      for (int64_t seg = 0; seg < batch_size; ++seg) {
        if (counts[static_cast<size_t>(seg)] > 0) {
          const float inv =
              1.0f / static_cast<float>(counts[static_cast<size_t>(seg)]);
          float* row = out + seg * embed;
          for (int64_t j = 0; j < embed; ++j) {
            row[j] *= inv;
          }
        }
      }
    }

    *annc::kernels::contextData<int32_t>(*slice_output) =
        (out_dim == 0) ? static_cast<int32_t>(batch_size)
                       : static_cast<int32_t>(embed);
    return context.status();
  } catch (const std::exception&) {
    return ANNC_STATUS_INTERNAL;
  } catch (...) {
    return ANNC_STATUS_INTERNAL;
  }
}

template <bool IsMean>
AnncStatusCode kpFusedSparseSegmentReduceDispatch(
    annc::kernels::KernelContext& context) {
  const auto* indices = context.getInput(3);
  if (!indices) return context.status();
  switch (indices->element_type) {
    case ANNC_ELEMENT_TYPE_I32:
      return kpFusedSparseSegmentReduceTyped<int32_t, IsMean>(context);
    case ANNC_ELEMENT_TYPE_I64:
      return kpFusedSparseSegmentReduceTyped<int64_t, IsMean>(context);
    default:
      return context.error(ANNC_STATUS_INVALID_ARGUMENT,
                           "SparseSegmentReduce indices must be i32 or i64");
  }
}

AnncStatusCode kpFusedSparseSegmentSumImpl(
    annc::kernels::KernelContext& context) {
  return kpFusedSparseSegmentReduceDispatch<false>(context);
}

AnncStatusCode kpFusedSparseSegmentMeanImpl(
    annc::kernels::KernelContext& context) {
  return kpFusedSparseSegmentReduceDispatch<true>(context);
}

}  // namespace
