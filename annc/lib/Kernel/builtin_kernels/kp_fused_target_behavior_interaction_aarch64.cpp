#include "Kernel/MemRefTypes.h"
#include "Kernel/KernelStatus.h"
#include "Support/ThreadPool/Parallel.h"
#include "Support/ThreadPool/ThreadPool.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>

#include "llvm/Support/raw_ostream.h"

namespace {

// Fused target-behavior interaction, ported from the 812
// KPFusedTargetBehaviorInteraction TF kernel (v0 scalar path).
//
//   relu_out = ReLU(BatchMatMul(batch_input, weight) + bias)
//   output = Concat([relu_out, tile_input, relu_out - tile_input,
//                    relu_out * tile_input], axis=-1)
//
// Shapes: batch_input [B,M,K], weight [K,N], bias [N], tile_input [B,M,N],
// output [B,M,4*N].
annc::kernels::KernelStatus kp_fused_target_behavior_interaction_aarch64_impl(
    annc::threadpool::AnncThreadPool* thread_pool,
    AnncMemRef3DF32* output,
    AnncMemRef3DF32* batch_input,
    AnncMemRef2DF32* weight,
    AnncMemRef1DF32* bias,
    AnncMemRef3DF32* tile_input) {
    if (!output || !batch_input || !weight || !bias || !tile_input) {
        llvm::errs() << "[ANNC Kernel] KPFusedTargetBehaviorInteraction failed: "
                     "null memref argument\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }

    const int64_t B = batch_input->sizes[0];
    const int64_t M = batch_input->sizes[1];
    const int64_t K = batch_input->sizes[2];
    const int64_t N = weight->sizes[1];

    if (B <= 0 || M <= 0 || K <= 0 || N <= 0) {
        llvm::errs() << "[ANNC Kernel] KPFusedTargetBehaviorInteraction failed: "
                     "invalid shape\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    if (weight->sizes[0] != K || bias->sizes[0] != N ||
        tile_input->sizes[0] != B || tile_input->sizes[1] != M ||
        tile_input->sizes[2] != N ||
        output->sizes[0] != B || output->sizes[1] != M ||
        output->sizes[2] != 4 * N) {
        llvm::errs() << "[ANNC Kernel] KPFusedTargetBehaviorInteraction failed: "
                     "shape mismatch\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    if (output->offset < 0 || batch_input->offset < 0 || weight->offset < 0 ||
        bias->offset < 0 || tile_input->offset < 0) {
        llvm::errs() << "[ANNC Kernel] KPFusedTargetBehaviorInteraction failed: "
                     "negative memref offset\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    if (!output->aligned || !batch_input->aligned || !weight->aligned ||
        !bias->aligned || !tile_input->aligned) {
        llvm::errs() << "[ANNC Kernel] KPFusedTargetBehaviorInteraction failed: "
                     "null memref data\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }
    if (batch_input->strides[0] != M * K ||
        batch_input->strides[1] != K || batch_input->strides[2] != 1 ||
        weight->strides[0] != N || weight->strides[1] != 1 ||
        bias->strides[0] != 1 ||
        tile_input->strides[0] != M * N || tile_input->strides[1] != N ||
        tile_input->strides[2] != 1 ||
        output->strides[0] != M * 4 * N || output->strides[1] != 4 * N ||
        output->strides[2] != 1) {
        llvm::errs() << "[ANNC Kernel] KPFusedTargetBehaviorInteraction failed: "
                     "non-contiguous memref is unsupported\n";
        return annc::kernels::KernelStatus::InvalidArgument;
    }

    const float* A = ANNC_MEMREF_DATA(*batch_input);
    const float* w = ANNC_MEMREF_DATA(*weight);
    const float* b = ANNC_MEMREF_DATA(*bias);
    const float* tile = ANNC_MEMREF_DATA(*tile_input);
    float* out = ANNC_MEMREF_DATA(*output);

    try {
        const int64_t out_cols = 4 * N;
        annc::threadpool::parallel_for(
            thread_pool, B, [&](int64_t begin, int64_t end) {
                for (int64_t batch = begin; batch < end; ++batch) {
                    const float* A_b = A + batch * M * K;
                    const float* tile_b = tile + batch * M * N;
                    float* out_b = out + batch * M * out_cols;
                    for (int64_t i = 0; i < M; ++i) {
                        const float* A_row = A_b + i * K;
                        const float* tile_row = tile_b + i * N;
                        float* out_row = out_b + i * out_cols;
                        for (int64_t j = 0; j < N; ++j) {
                            float sum = 0.0f;
                            for (int64_t k = 0; k < K; ++k) {
                                sum += A_row[k] * w[k * N + j];
                            }
                            const float val = sum + b[j];
                            const float relu = (val > 0.0f) ? val : 0.0f;
                            const float tv = tile_row[j];
                            out_row[j] = relu;
                            out_row[N + j] = tv;
                            out_row[2 * N + j] = relu - tv;
                            out_row[3 * N + j] = relu * tv;
                        }
                    }
                }
            });
        return annc::kernels::KernelStatus::Success;
    } catch (const std::exception& ex) {
        llvm::errs() << "[ANNC Kernel] KPFusedTargetBehaviorInteraction failed: "
                     << ex.what() << '\n';
        return annc::kernels::KernelStatus::RuntimeError;
    } catch (...) {
        llvm::errs() << "[ANNC Kernel] KPFusedTargetBehaviorInteraction failed: "
                     "unknown exception\n";
        return annc::kernels::KernelStatus::UnknownError;
    }
}

} // namespace
