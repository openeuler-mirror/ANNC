#include <cstdint>
#include <cstring>
#include <exception>

#include "Kernel/ExecutionContextUtils.h"
#include "Kernel/KernelStatus.h"
#include "Kernel/MemRefTypes.h"
#include "Support/ThreadPool/Parallel.h"
#include "Support/ThreadPool/ThreadPool.h"

namespace {

// Fused embedding padding, ported from the 812 KPFusedEmbeddingPadding TF
// kernel, adapted to the Execution V2 ABI (outputs allocated through the
// execution context callbacks).
//
//   padding_rows = origin_shape[0] - input_rows
//   reshape_rows = (padding_rows + data.rows) * data.cols / reshape_sizes[1]
//   pack must equal data.cols (812 check: embedding dims)
// Fast variant outputs two scalar slots only; the full variant also writes
// slot 1 as the padded [reshape_rows, reshape_cols] data.
//
// 移植补查 (812 未查): data.rows == input_rows (不等时 812 会越界读或留未
// 初始化区), 拒绝而非静默错误。
template <bool HasOutData>
annc::kernels::KernelStatus kpFusedEmbeddingPaddingImpl(
    annc::threadpool::AnncThreadPool* thread_pool,
    const AnncExecutionContext* execution, AnncMemRef1DI64* origin_shape,
    AnncMemRef0DI32* input_rows, AnncMemRef0DI32* pack, AnncMemRef2DF32* data,
    AnncMemRef1DI32* reshape_sizes) {
  (void)thread_pool;
  if (!execution || !origin_shape || !input_rows || !pack || !data ||
      !reshape_sizes) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  const int64_t data_rows = data->sizes[0];
  const int64_t data_cols = data->sizes[1];
  if (origin_shape->sizes[0] != 2 || reshape_sizes->sizes[0] != 2) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  if (data_rows < 0 || data_cols <= 0) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  const int64_t origin_rows = ANNC_MEMREF_DATA(*origin_shape)[0];
  const int32_t input_rows_v = *ANNC_MEMREF_DATA(*input_rows);
  const int32_t r0 = ANNC_MEMREF_DATA(*reshape_sizes)[0];
  const int32_t reshape_cols = ANNC_MEMREF_DATA(*reshape_sizes)[1];
  const int32_t pack_v = *ANNC_MEMREF_DATA(*pack);

  const int64_t padding_rows = origin_rows - input_rows_v;
  if (padding_rows < 0 || input_rows_v < 0 || r0 != -1 || reshape_cols <= 0) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  // 812: pack 必须等于 embedding dims (Fill 形状与 data 一致的 ConcatV2
  // 语义要求)。
  if (pack_v != data_cols) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  if (input_rows_v != data_rows) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  if (data->strides[0] != data_cols || data->strides[1] != 1 ||
      !data->aligned) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }

  const int64_t output_rows = padding_rows + data_rows;
  const int64_t output_cols = data_cols;
  if (output_rows * output_cols % reshape_cols != 0) {
    return annc::kernels::KernelStatus::InvalidArgument;
  }
  const int64_t reshape_rows = output_rows * output_cols / reshape_cols;

  try {
    auto out_padding_rows =
        annc::kernels::execution::allocateOutput0DI32(
            execution, 0, ANNC_ALLOCATION_NONE);
    if (!out_padding_rows) {
      llvm::consumeError(out_padding_rows.takeError());
      return annc::kernels::KernelStatus::RuntimeError;
    }
    if (HasOutData) {
      auto out_data = annc::kernels::execution::allocateOutput2DF32(
          execution, 1, reshape_rows, reshape_cols, ANNC_ALLOCATION_NONE);
      if (!out_data) {
        llvm::consumeError(out_data.takeError());
        return annc::kernels::KernelStatus::RuntimeError;
      }
      float* out = ANNC_MEMREF_DATA(*out_data);
      const float* src = ANNC_MEMREF_DATA(*data);
      std::memcpy(out, src,
                  static_cast<size_t>(input_rows_v) * data_cols *
                      sizeof(float));
      std::memset(out + static_cast<size_t>(input_rows_v) * data_cols, 0,
                  static_cast<size_t>(padding_rows) * data_cols *
                      sizeof(float));
    } else {
      auto out_reshape_rows =
          annc::kernels::execution::allocateOutput0DI32(
              execution, 1, ANNC_ALLOCATION_NONE);
      if (!out_reshape_rows) {
        llvm::consumeError(out_reshape_rows.takeError());
        return annc::kernels::KernelStatus::RuntimeError;
      }
      *ANNC_MEMREF_DATA(*out_reshape_rows) =
          static_cast<int32_t>(reshape_rows);
    }
    *ANNC_MEMREF_DATA(*out_padding_rows) =
        static_cast<int32_t>(padding_rows);
    return annc::kernels::KernelStatus::Success;
  } catch (const std::exception&) {
    return annc::kernels::KernelStatus::RuntimeError;
  } catch (...) {
    return annc::kernels::KernelStatus::UnknownError;
  }
}

// Macro-safe dispatch helpers.
annc::kernels::KernelStatus kpFusedEmbeddingPaddingFast(
    annc::threadpool::AnncThreadPool* tp, const AnncExecutionContext* execution,
    AnncMemRef1DI64* origin_shape, AnncMemRef0DI32* input_rows,
    AnncMemRef0DI32* pack, AnncMemRef2DF32* data,
    AnncMemRef1DI32* reshape_sizes) {
  return kpFusedEmbeddingPaddingImpl<false>(
      tp, execution, origin_shape, input_rows, pack, data, reshape_sizes);
}

annc::kernels::KernelStatus kpFusedEmbeddingPaddingFull(
    annc::threadpool::AnncThreadPool* tp, const AnncExecutionContext* execution,
    AnncMemRef1DI64* origin_shape, AnncMemRef0DI32* input_rows,
    AnncMemRef0DI32* pack, AnncMemRef2DF32* data,
    AnncMemRef1DI32* reshape_sizes) {
  return kpFusedEmbeddingPaddingImpl<true>(
      tp, execution, origin_shape, input_rows, pack, data, reshape_sizes);
}

}  // namespace
