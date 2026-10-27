#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>

#include "Context/KernelContextUtils.h"

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
AnncStatusCode kp_fused_target_behavior_interaction_aarch64_impl(
    annc::kernels::KernelContext& context) {
  const auto* batch_input = context.getInput(0);
  const auto* weight = context.getInput(1);
  const auto* bias = context.getInput(2);
  const auto* tile_input = context.getInput(3);
  if (!batch_input || !weight || !bias || !tile_input) return context.status();

  const int64_t B = batch_input->sizes[0];
  const int64_t M = batch_input->sizes[1];
  const int64_t K = batch_input->sizes[2];
  const int64_t N = weight->sizes[1];

  if (B <= 0 || M <= 0 || K <= 0 || N <= 0 ||
      N > std::numeric_limits<int64_t>::max() / 4) {
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        "[ANNC Kernel] KPFusedTargetBehaviorInteraction failed: "
        "invalid shape\n");
  }
  if (weight->sizes[0] != K || bias->sizes[0] != N ||
      tile_input->sizes[0] != B || tile_input->sizes[1] != M ||
      tile_input->sizes[2] != N) {
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        "[ANNC Kernel] KPFusedTargetBehaviorInteraction failed: "
        "shape mismatch\n");
  }
  if (batch_input->offset < 0 || weight->offset < 0 || bias->offset < 0 ||
      tile_input->offset < 0) {
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        "[ANNC Kernel] KPFusedTargetBehaviorInteraction failed: "
        "negative memref offset\n");
  }
  if (!batch_input->aligned || !weight->aligned || !bias->aligned ||
      !tile_input->aligned) {
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        "[ANNC Kernel] KPFusedTargetBehaviorInteraction failed: "
        "null memref data\n");
  }
  if (batch_input->strides[0] != M * K || batch_input->strides[1] != K ||
      batch_input->strides[2] != 1 || weight->strides[0] != N ||
      weight->strides[1] != 1 || bias->strides[0] != 1 ||
      tile_input->strides[0] != M * N || tile_input->strides[1] != N ||
      tile_input->strides[2] != 1) {
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        "[ANNC Kernel] KPFusedTargetBehaviorInteraction failed: "
        "non-contiguous memref is unsupported\n");
  }

  auto* output = context.allocateOutput<float>(0, {B, M, 4 * N});
  if (!output) return context.status();
  if (output->sizes[0] != B || output->sizes[1] != M ||
      output->sizes[2] != 4 * N || output->offset < 0 || !output->aligned ||
      output->strides[0] != M * 4 * N || output->strides[1] != 4 * N ||
      output->strides[2] != 1) {
    return context.error(
        ANNC_STATUS_INVALID_ARGUMENT,
        "[ANNC Kernel] KPFusedTargetBehaviorInteraction failed: "
        "invalid output layout\n");
  }

  const float* A = annc::kernels::contextData<float>(*batch_input);
  const float* w = annc::kernels::contextData<float>(*weight);
  const float* b = annc::kernels::contextData<float>(*bias);
  const float* tile = annc::kernels::contextData<float>(*tile_input);
  float* out = annc::kernels::contextData<float>(*output);

  try {
    const int64_t out_cols = 4 * N;
    auto workers = context.getThreadPool();
    workers.parallelFor(B, [&](int64_t begin, int64_t end) {
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
    return context.status();
  } catch (const std::exception& ex) {
    return context.error(ANNC_STATUS_INTERNAL, ex.what());
  } catch (...) {
    return context.error(
        ANNC_STATUS_INTERNAL,
        "[ANNC Kernel] KPFusedTargetBehaviorInteraction failed: "
        "unknown exception\n");
  }
}

}  // namespace
