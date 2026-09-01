#include "Kernel/MemRefTypes.h"
#include "Kernel/KernelStatus.h"
#include "Support/ThreadPool/Parallel.h"
#include "Support/ThreadPool/ThreadPool.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>

#include "llvm/Support/raw_ostream.h"

namespace {

// Fused double gather, ported from the 812 KPFusedEmbeddingActionIdGather TF
// kernel (tf_custom_ops/kernels/embedding_fused_action_id_gather.cc).
//
// Step 1: temp[i, j, :] = params[indices1[i, j], :]      ([I10, I11, P_row])
// Step 2: output[i, j, (k), :] = temp[indices2[i, j], (k), :]
// Fused:  output[i, j, (k), :] = params[indices1[indices2[i, j], (k)], :]
//
// R1/R2: rank of indices1/indices2 (1 or 2; 1-D is normalized to the
// 2-D form [N, 1], i.e. I10 = N, I11 = 1 — matching the JD kernel's
// I10/I11 convention).
// T1/T2: element type of indices1/indices2 (int64/int32; the 4 (T1,T2)
// combos mirror the JD kernel registration: (i64,i32), (i32,i32),
// (i64,i64), (i32,i64)).
// params is 2D [P0, P_row].  The gathered tensor is reshaped to
// [pack_size, a_reshaped_cols] and zero-padded per row to
// [pack_size, a_reshaped_cols + pack_const].
template <int R1, int R2, typename T1, typename T2>
annc::kernels::KernelStatus kp_fused_action_id_gather_impl(
    annc::threadpool::AnncThreadPool* thread_pool,
    AnncMemRef2DF32* output,
    const T1* indices1_data, int64_t I10, int64_t I11,
    AnncMemRef2DF32* params,
    const T2* indices2_data, int64_t I20, int64_t I21,
    int32_t pack_size, int32_t pack_const) {
    static_assert(R1 == 1 || R1 == 2, "indices1 rank must be 1 or 2");
    static_assert(R2 == 1 || R2 == 2, "indices2 rank must be 1 or 2");

    if (!output || !indices1_data || !params || !indices2_data) {
        llvm::errs() << "[ANNC Kernel] KPFusedEmbeddingActionIdGather failed: "
                     "null memref argument\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }

    const int64_t P0 = params->sizes[0];
    const int64_t P_row = params->sizes[1];
    const int64_t out_rows = output->sizes[0];
    const int64_t out_cols = output->sizes[1];

    if (pack_size <= 0 || P0 <= 0 || P_row <= 0 || out_rows <= 0 ||
        out_cols <= 0 || pack_const < 0 || I10 <= 0 || I11 <= 0 || I20 <= 0 ||
        I21 <= 0) {
        llvm::errs() << "[ANNC Kernel] KPFusedEmbeddingActionIdGather failed: "
                     "invalid shape\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    if (params->offset < 0 || output->offset < 0) {
        llvm::errs() << "[ANNC Kernel] KPFusedEmbeddingActionIdGather failed: "
                     "negative memref offset\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    if (!params->aligned || !output->aligned) {
        llvm::errs() << "[ANNC Kernel] KPFusedEmbeddingActionIdGather failed: "
                     "null memref data\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    if (params->strides[0] != P_row || params->strides[1] != 1 ||
        output->strides[0] != out_cols || output->strides[1] != 1) {
        llvm::errs() << "[ANNC Kernel] KPFusedEmbeddingActionIdGather failed: "
                     "non-contiguous memref is unsupported\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    if (pack_size != out_rows) {
        llvm::errs() << "[ANNC Kernel] KPFusedEmbeddingActionIdGather failed: "
                     "pack_size mismatch\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }

    // gathered tensor: [I20*I21, I11, P_row] (R1==2) or [I20*I21, P_row]
    // (R1==1, I11 normalized to 1).  Reshaped row-major to
    // [pack_size, a_reshaped_cols].
    const int64_t gathered_rows = I20 * I21;
    const int64_t gathered_row_elems = I11 * P_row;
    const int64_t gathered_elems = gathered_rows * gathered_row_elems;
    if (gathered_elems % pack_size != 0) {
        llvm::errs() << "[ANNC Kernel] KPFusedEmbeddingActionIdGather failed: "
                     "gathered size not divisible by pack_size\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    const int64_t a_reshaped_cols = gathered_elems / pack_size;
    if (out_cols != a_reshaped_cols + pack_const) {
        llvm::errs() << "[ANNC Kernel] KPFusedEmbeddingActionIdGather failed: "
                     "output column count mismatch\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }

    const float* params_data = ANNC_MEMREF_DATA(*params);
    float* out = ANNC_MEMREF_DATA(*output);

    try {
        std::atomic<bool> failed{false};
        annc::threadpool::parallel_for(
            thread_pool, gathered_rows, [&](int64_t begin, int64_t end) {
                for (int64_t flat = begin; flat < end; ++flat) {
                    const T2 idx2 = indices2_data[flat];
                    if (idx2 < 0 || idx2 >= I10) {
                        failed.store(true, std::memory_order_relaxed);
                        return;
                    }
                    for (int64_t k = 0; k < I11; ++k) {
                        const T1 idx1 = indices1_data[idx2 * I11 + k];
                        if (idx1 < 0 || idx1 >= P0) {
                            failed.store(true, std::memory_order_relaxed);
                            return;
                        }
                        // gathered 张量按 float 元素 reshape 到
                        // [pack_size, a_reshaped_cols]: 块 (flat, k) 的
                        // float 起点 = (flat * I11 + k) * P_row.
                        const int64_t float_lin = (flat * I11 + k) * P_row;
                        const int64_t row = float_lin / a_reshaped_cols;
                        const int64_t col = float_lin % a_reshaped_cols;
                        std::memcpy(out + row * out_cols + col,
                                    params_data + idx1 * P_row,
                                    static_cast<size_t>(P_row) * sizeof(float));
                    }
                }
            });
        if (failed.load(std::memory_order_relaxed)) {
            llvm::errs() << "[ANNC Kernel] KPFusedEmbeddingActionIdGather "
                            "failed: index out of range\n";
            return annc::kernels::KernelStatus::InvalidArgument;
        }
        if (pack_const > 0) {
            annc::threadpool::parallel_for(
                thread_pool, pack_size, [&](int64_t begin, int64_t end) {
                    for (int64_t row = begin; row < end; ++row) {
                        std::memset(out + row * out_cols + a_reshaped_cols, 0,
                                    static_cast<size_t>(pack_const) *
                                        sizeof(float));
                    }
                });
        }
        return annc::kernels::KernelStatus::Success;
    } catch (const std::exception& ex) {
        llvm::errs() << "[ANNC Kernel] KPFusedEmbeddingActionIdGather failed: "
                     << ex.what() << '\n';
        return annc::kernels::KernelStatus::RuntimeError;
    } catch (...) {
        llvm::errs() << "[ANNC Kernel] KPFusedEmbeddingActionIdGather failed: "
                     "unknown exception\n";
        return annc::kernels::KernelStatus::UnknownError;
    }
}

// Non-template dispatch entry points.  The ANNC_KERNEL macro body is
// comma-split at the preprocessor level, so template argument lists like
// <1, 1> cannot appear there; these function-pointer constants give the
// specs a comma-free way to select the rank/dtype specialization.
template <typename T1, typename T2>
using ActionIdGatherFn = annc::kernels::KernelStatus (*)(
    annc::threadpool::AnncThreadPool*, AnncMemRef2DF32*, const T1*,
    int64_t, int64_t, AnncMemRef2DF32*, const T2*, int64_t, int64_t,
    int32_t, int32_t);

// v1 默认组合 (indices1 int64, indices2 int32): 保留原符号名。
static constexpr ActionIdGatherFn<int64_t, int32_t> kActionIdGather11 =
    &kp_fused_action_id_gather_impl<1, 1, int64_t, int32_t>;
static constexpr ActionIdGatherFn<int64_t, int32_t> kActionIdGather21 =
    &kp_fused_action_id_gather_impl<2, 1, int64_t, int32_t>;
static constexpr ActionIdGatherFn<int64_t, int32_t> kActionIdGather12 =
    &kp_fused_action_id_gather_impl<1, 2, int64_t, int32_t>;
static constexpr ActionIdGatherFn<int64_t, int32_t> kActionIdGather22 =
    &kp_fused_action_id_gather_impl<2, 2, int64_t, int32_t>;

// JD 其余 3 组 (Tindices1, Tindices2) 组合 — 后缀 <T1><T2>。
static constexpr ActionIdGatherFn<int32_t, int32_t> kActionIdGather11_I32I32 =
    &kp_fused_action_id_gather_impl<1, 1, int32_t, int32_t>;
static constexpr ActionIdGatherFn<int32_t, int32_t> kActionIdGather21_I32I32 =
    &kp_fused_action_id_gather_impl<2, 1, int32_t, int32_t>;
static constexpr ActionIdGatherFn<int32_t, int32_t> kActionIdGather12_I32I32 =
    &kp_fused_action_id_gather_impl<1, 2, int32_t, int32_t>;
static constexpr ActionIdGatherFn<int32_t, int32_t> kActionIdGather22_I32I32 =
    &kp_fused_action_id_gather_impl<2, 2, int32_t, int32_t>;

static constexpr ActionIdGatherFn<int64_t, int64_t> kActionIdGather11_I64I64 =
    &kp_fused_action_id_gather_impl<1, 1, int64_t, int64_t>;
static constexpr ActionIdGatherFn<int64_t, int64_t> kActionIdGather21_I64I64 =
    &kp_fused_action_id_gather_impl<2, 1, int64_t, int64_t>;
static constexpr ActionIdGatherFn<int64_t, int64_t> kActionIdGather12_I64I64 =
    &kp_fused_action_id_gather_impl<1, 2, int64_t, int64_t>;
static constexpr ActionIdGatherFn<int64_t, int64_t> kActionIdGather22_I64I64 =
    &kp_fused_action_id_gather_impl<2, 2, int64_t, int64_t>;

static constexpr ActionIdGatherFn<int32_t, int64_t> kActionIdGather11_I32I64 =
    &kp_fused_action_id_gather_impl<1, 1, int32_t, int64_t>;
static constexpr ActionIdGatherFn<int32_t, int64_t> kActionIdGather21_I32I64 =
    &kp_fused_action_id_gather_impl<2, 1, int32_t, int64_t>;
static constexpr ActionIdGatherFn<int32_t, int64_t> kActionIdGather12_I32I64 =
    &kp_fused_action_id_gather_impl<1, 2, int32_t, int64_t>;
static constexpr ActionIdGatherFn<int32_t, int64_t> kActionIdGather22_I32I64 =
    &kp_fused_action_id_gather_impl<2, 2, int32_t, int64_t>;

} // namespace
