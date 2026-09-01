#include <cstdint>
#include <exception>
#include <vector>

#include "Kernel/ExecutionContextUtils.h"
#include "Kernel/KernelStatus.h"
#include "Kernel/MemRefTypes.h"
#include "Support/ThreadPool/Parallel.h"
#include "Support/ThreadPool/ThreadPool.h"

namespace {

// Fused sparse reshape, ported from the 812 KPFusedSparseReshape TF kernel,
// adapted to the Execution V2 ABI (outputs allocated through the execution
// context callbacks).  Semantics:
//
//   col = begin[1]                          (begin is 1-D i32, 2 elements)
//   shape_in = [num_rows, pack_const]       (num_rows = slice_input rows)
//   indices_in[i] = [i, slice_input[i, col]]
//   ReshapeKp(indices_in, shape_in, new_shape) — standard TF SparseReshape
//     row-major coordinate remap, one -1 dimension allowed.
//
// Outputs: slot 0 = out_indices (i64 [num_rows, output_rank]),
//          slot 1 = out_shape (i64 [output_rank]).
// T is the pack_const element type (int32/int64); new_shape is always i64.
template <typename T>
annc::kernels::KernelStatus kpFusedSparseReshapeImpl(
    annc::threadpool::AnncThreadPool* thread_pool,
    const AnncExecutionContext* execution, AnncMemRef2DI64* slice_input,
    AnncMemRef1DI32* begin, const T* pack_const_data,
    const int64_t* new_shape_data, int64_t new_shape_len) {
  (void)thread_pool;
  if (!execution || !slice_input || !begin || !pack_const_data ||
      !new_shape_data) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  const int64_t num_rows = slice_input->sizes[0];
  const int64_t dims = slice_input->sizes[1];
  const int64_t output_rank = new_shape_len;

  if (num_rows < 0 || dims < 0 || output_rank < 0) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  // 812 checks: begin 1-D with 2 elements, new_shape 2 elements.
  if (begin->sizes[0] != 2 || new_shape_len != 2) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  if (slice_input->offset < 0 || begin->offset < 0) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  if ((num_rows > 0 && dims > 0 && !slice_input->aligned) ||
      !begin->aligned) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  if (slice_input->strides[0] != dims || slice_input->strides[1] != 1 ||
      begin->strides[0] != 1) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  const int32_t col = ANNC_MEMREF_DATA(*begin)[1];
  // 812: begin[1] must < slice_input.dim_size(1).
  if (col < 0 || col >= dims) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  try {
    // ReshapeKp: input shape is always [num_rows, pack_const] (rank 2).
    const int64_t pack_const = static_cast<int64_t>(*pack_const_data);
    const int64_t input_rank = 2;
    const int64_t input_shape[2] = {num_rows, pack_const};
    const int64_t dense_size = num_rows * pack_const;
    if (pack_const < 0) {
      return annc::kernels::KernelStatus::InvalidArgument;
    }

    // Resolve the target shape, inferring a single -1 dimension.
    std::vector<int64_t> output_shape_vec;
    output_shape_vec.reserve(output_rank);
    int64_t product = 1;
    int64_t unknown_index = -1;
    for (int64_t d = 0; d < output_rank; ++d) {
      const int64_t size = static_cast<int64_t>(new_shape_data[d]);
      if (size == -1) {
        if (unknown_index != -1) {
          return annc::kernels::KernelStatus::InvalidArgument;
        }
        unknown_index = d;
        output_shape_vec.push_back(1);
      } else {
        if (size < 0) {
          return annc::kernels::KernelStatus::InvalidArgument;
        }
        product *= size;
        output_shape_vec.push_back(size);
      }
    }
    if (unknown_index != -1) {
      if (product <= 0) {
        return annc::kernels::KernelStatus::InvalidArgument;
      }
      const int64_t missing = dense_size / product;
      if (product * missing != dense_size) {
        return annc::kernels::KernelStatus::InvalidArgument;
      }
      output_shape_vec[unknown_index] = missing;
    }
    int64_t output_num_elems = 1;
    for (int64_t d = 0; d < output_rank; ++d) {
      output_num_elems *= output_shape_vec[d];
    }
    if (output_num_elems != dense_size) {
      return annc::kernels::KernelStatus::InvalidArgument;
    }

    // V2: value-dependent out_indices shape [num_rows, output_rank].
    auto out_indices = annc::kernels::execution::allocateOutput2DI64(
        execution, 0, num_rows, output_rank, ANNC_ALLOCATION_NONE);
    if (!out_indices) {
      llvm::consumeError(out_indices.takeError());
      return annc::kernels::KernelStatus::RuntimeError;
    }
    auto out_shape = annc::kernels::execution::allocateOutput1DI64(
        execution, 1, output_rank, ANNC_ALLOCATION_NONE);
    if (!out_shape) {
      llvm::consumeError(out_shape.takeError());
      return annc::kernels::KernelStatus::RuntimeError;
    }

    const int64_t* keys_data = ANNC_MEMREF_DATA(*slice_input);
    int64_t* out_idx_data = ANNC_MEMREF_DATA(*out_indices);
    int64_t* out_shape_data = ANNC_MEMREF_DATA(*out_shape);

    // 812 shortcut: input_shape == output_shape → indices unchanged.
    const bool same_shape =
        output_rank == input_rank && output_shape_vec[0] == input_shape[0] &&
        output_shape_vec[1] == input_shape[1];
    if (same_shape) {
      for (int64_t i = 0; i < num_rows; ++i) {
        out_idx_data[i * 2 + 0] = i;
        out_idx_data[i * 2 + 1] = keys_data[i * dims + col];
      }
      out_shape_data[0] = input_shape[0];
      out_shape_data[1] = input_shape[1];
      return annc::kernels::KernelStatus::Success;
    }

    // Row-major coordinate remap (812 ReshapeKp).
    const int64_t input_strides[2] = {input_shape[1], 1};
    std::vector<int64_t> output_strides(output_rank);
    if (output_rank > 0) {
      output_strides[output_rank - 1] = 1;
      for (int64_t d = output_rank - 2; d >= 0; --d) {
        output_strides[d] =
            output_strides[d + 1] * output_shape_vec[d + 1];
      }
    }

    for (int64_t i = 0; i < num_rows; ++i) {
      int64_t id = static_cast<int64_t>(i) * input_strides[0] +
                   keys_data[i * dims + col] * input_strides[1];
      for (int64_t j = 0; j < output_rank; ++j) {
        out_idx_data[i * output_rank + j] = id / output_strides[j];
        id %= output_strides[j];
      }
    }
    for (int64_t j = 0; j < output_rank; ++j) {
      out_shape_data[j] = output_shape_vec[j];
    }
    return annc::kernels::KernelStatus::Success;
  } catch (const std::exception&) {
    return annc::kernels::KernelStatus::RuntimeError;
  } catch (...) {
    return annc::kernels::KernelStatus::UnknownError;
  }
}

// Non-template dispatch entry points (macro-safe: all commas in parens).
annc::kernels::KernelStatus kpFusedSparseReshapeI64(
    annc::threadpool::AnncThreadPool* tp, const AnncExecutionContext* execution,
    AnncMemRef2DI64* slice_input, AnncMemRef1DI32* begin,
    AnncMemRef0DI64* pack_const, AnncMemRef1DI64* new_shape) {
  return kpFusedSparseReshapeImpl<int64_t>(
      tp, execution, slice_input, begin, ANNC_MEMREF_DATA(*pack_const),
      ANNC_MEMREF_DATA(*new_shape), new_shape->sizes[0]);
}

annc::kernels::KernelStatus kpFusedSparseReshapeI32(
    annc::threadpool::AnncThreadPool* tp, const AnncExecutionContext* execution,
    AnncMemRef2DI64* slice_input, AnncMemRef1DI32* begin,
    AnncMemRef0DI32* pack_const, AnncMemRef1DI64* new_shape) {
  return kpFusedSparseReshapeImpl<int32_t>(
      tp, execution, slice_input, begin, ANNC_MEMREF_DATA(*pack_const),
      ANNC_MEMREF_DATA(*new_shape), new_shape->sizes[0]);
}

// DynLimit variants: the trailing 0-D i32 range_limit is the dynamic
// Range.limit from varlen-batch graphs.  The kernel regenerates the
// row-number column internally and never reads it; the parameter exists only
// to match the ABI wrapper order (keys, begin, pack_const, new_shape,
// range_limit).
annc::kernels::KernelStatus kpFusedSparseReshapeI64DynLimit(
    annc::threadpool::AnncThreadPool* tp, const AnncExecutionContext* execution,
    AnncMemRef2DI64* slice_input, AnncMemRef1DI32* begin,
    AnncMemRef0DI64* pack_const, AnncMemRef1DI64* new_shape,
    AnncMemRef0DI32* range_limit) {
  (void)range_limit;
  return kpFusedSparseReshapeImpl<int64_t>(
      tp, execution, slice_input, begin, ANNC_MEMREF_DATA(*pack_const),
      ANNC_MEMREF_DATA(*new_shape), new_shape->sizes[0]);
}

annc::kernels::KernelStatus kpFusedSparseReshapeI32DynLimit(
    annc::threadpool::AnncThreadPool* tp, const AnncExecutionContext* execution,
    AnncMemRef2DI64* slice_input, AnncMemRef1DI32* begin,
    AnncMemRef0DI32* pack_const, AnncMemRef1DI64* new_shape,
    AnncMemRef0DI32* range_limit) {
  (void)range_limit;
  return kpFusedSparseReshapeImpl<int32_t>(
      tp, execution, slice_input, begin, ANNC_MEMREF_DATA(*pack_const),
      ANNC_MEMREF_DATA(*new_shape), new_shape->sizes[0]);
}

}  // namespace
