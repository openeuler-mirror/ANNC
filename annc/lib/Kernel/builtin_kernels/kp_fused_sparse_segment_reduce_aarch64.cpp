#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include "Kernel/ExecutionContextUtils.h"
#include "Kernel/KernelStatus.h"
#include "Kernel/MemRefTypes.h"
#include "Support/ThreadPool/Parallel.h"
#include "Support/ThreadPool/ThreadPool.h"

namespace {

// Fused sparse segment reduce, ported from the 812 KPFusedSparseSegmentReduce
// TF kernel, adapted to the Execution V2 ABI (outputs are allocated through
// the execution context callbacks).
//
//   seg_id = keys[i, col]  (col = begin[1], runtime)
//   batch_size = max(seg_id) + 1
//   slot 0: output[seg_id] = SUM/MEAN over data[indices[i]]
//   slot 1: slice_output = (begin_1[0] == 0) ? batch_size : embedding_size
template <typename Tidx, bool IsMean>
annc::kernels::KernelStatus kpFusedSparseSegmentReduceImpl(
    annc::threadpool::AnncThreadPool* thread_pool,
    const AnncExecutionContext* execution, AnncMemRef2DI64* keys,
    AnncMemRef1DI32* begin, AnncMemRef2DF32* data, const Tidx* indices_data,
    int64_t num_indices, AnncMemRef1DI32* begin_1) {
  (void)thread_pool;
  if (!execution || !indices_data || !keys || !begin || !data || !begin_1) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  const int64_t num_rows = data->sizes[0];
  const int64_t embed = data->sizes[1];
  const int64_t num_keys = keys->sizes[0];
  const int64_t key_width = keys->sizes[1];

  if (num_rows < 0 || embed <= 0 || num_keys < 0 || key_width < 0 ||
      num_indices < 0) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  if (begin->sizes[0] != 2 || begin_1->sizes[0] != 1) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  const int32_t col = ANNC_MEMREF_DATA(*begin)[1];
  const int32_t out_dim = ANNC_MEMREF_DATA(*begin_1)[0];
  if (col < 0 || col >= key_width) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  if (num_indices != num_keys) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  if (data->offset < 0 || keys->offset < 0 || begin->offset < 0 ||
      begin_1->offset < 0) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  if ((num_rows > 0 && !data->aligned) ||
      (num_keys > 0 && !keys->aligned) || !begin->aligned ||
      !begin_1->aligned) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  if (data->strides[0] != embed || data->strides[1] != 1 ||
      keys->strides[0] != key_width || keys->strides[1] != 1 ||
      begin->strides[0] != 1 || begin_1->strides[0] != 1) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  const int64_t* keys_data = ANNC_MEMREF_DATA(*keys);
  const float* data_ptr = ANNC_MEMREF_DATA(*data);

  try {
    // batch_size = max(seg_id) + 1 (value-dependent output shape; the V2
    // execution API allocates it after this scan).
    int64_t max_seg = -1;
    for (int64_t i = 0; i < num_keys; ++i) {
      const int64_t seg = keys_data[i * key_width + col];
      if (seg < 0) {
        return annc::kernels::KernelStatus::InvalidArgument;
      }
      if (seg > max_seg) max_seg = seg;
    }
    const int64_t batch_size = max_seg + 1;

    auto output = annc::kernels::execution::allocateOutput2DF32(
        execution, 0, batch_size, embed, ANNC_ALLOCATION_ZERO);
    if (!output) {
      llvm::consumeError(output.takeError());
      return annc::kernels::KernelStatus::RuntimeError;
    }
    auto slice_output = annc::kernels::execution::allocateOutput0DI32(
        execution, 1, ANNC_ALLOCATION_NONE);
    if (!slice_output) {
      llvm::consumeError(slice_output.takeError());
      return annc::kernels::KernelStatus::RuntimeError;
    }

    float* out = ANNC_MEMREF_DATA(*output);
    // ANNC_ALLOCATION_ZERO declares zeroed memory, but keep the explicit
    // memset so the kernel is also correct with allocators that ignore the
    // flag (e.g. the ctypes test harness).
    std::memset(out, 0,
                static_cast<size_t>(batch_size) * embed * sizeof(float));
    std::vector<int32_t> counts;
    if (IsMean) {
      counts.assign(static_cast<size_t>(batch_size), 0);
    }

    for (int64_t i = 0; i < num_indices; ++i) {
      const int64_t seg = keys_data[i * key_width + col];
      const int64_t data_row = static_cast<int64_t>(indices_data[i]);
      if (data_row < 0 || data_row >= num_rows) {
        return annc::kernels::KernelStatus::InvalidArgument;
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
          const float inv = 1.0f / static_cast<float>(
                                       counts[static_cast<size_t>(seg)]);
          float* row = out + seg * embed;
          for (int64_t j = 0; j < embed; ++j) {
            row[j] *= inv;
          }
        }
      }
    }

    *ANNC_MEMREF_DATA(*slice_output) =
        (out_dim == 0) ? static_cast<int32_t>(batch_size)
                       : static_cast<int32_t>(embed);
    return annc::kernels::KernelStatus::Success;
  } catch (const std::exception&) {
    return annc::kernels::KernelStatus::RuntimeError;
  } catch (...) {
    return annc::kernels::KernelStatus::UnknownError;
  }
}

// Non-template dispatch entry points (macro-safe: all commas in parens).
annc::kernels::KernelStatus kpFusedSparseSegmentReduceI64Sum(
    annc::threadpool::AnncThreadPool* tp, const AnncExecutionContext* execution,
    AnncMemRef2DI64* keys, AnncMemRef1DI32* begin, AnncMemRef2DF32* data,
    AnncMemRef1DI64* indices, AnncMemRef1DI32* begin_1) {
  return kpFusedSparseSegmentReduceImpl<int64_t, false>(
      tp, execution, keys, begin, data, ANNC_MEMREF_DATA(*indices),
      indices->sizes[0], begin_1);
}
annc::kernels::KernelStatus kpFusedSparseSegmentReduceI64Mean(
    annc::threadpool::AnncThreadPool* tp, const AnncExecutionContext* execution,
    AnncMemRef2DI64* keys, AnncMemRef1DI32* begin, AnncMemRef2DF32* data,
    AnncMemRef1DI64* indices, AnncMemRef1DI32* begin_1) {
  return kpFusedSparseSegmentReduceImpl<int64_t, true>(
      tp, execution, keys, begin, data, ANNC_MEMREF_DATA(*indices),
      indices->sizes[0], begin_1);
}
annc::kernels::KernelStatus kpFusedSparseSegmentReduceI32Sum(
    annc::threadpool::AnncThreadPool* tp, const AnncExecutionContext* execution,
    AnncMemRef2DI64* keys, AnncMemRef1DI32* begin, AnncMemRef2DF32* data,
    AnncMemRef1DI32* indices, AnncMemRef1DI32* begin_1) {
  return kpFusedSparseSegmentReduceImpl<int32_t, false>(
      tp, execution, keys, begin, data, ANNC_MEMREF_DATA(*indices),
      indices->sizes[0], begin_1);
}
annc::kernels::KernelStatus kpFusedSparseSegmentReduceI32Mean(
    annc::threadpool::AnncThreadPool* tp, const AnncExecutionContext* execution,
    AnncMemRef2DI64* keys, AnncMemRef1DI32* begin, AnncMemRef2DF32* data,
    AnncMemRef1DI32* indices, AnncMemRef1DI32* begin_1) {
  return kpFusedSparseSegmentReduceImpl<int32_t, true>(
      tp, execution, keys, begin, data, ANNC_MEMREF_DATA(*indices),
      indices->sizes[0], begin_1);
}

}  // namespace
