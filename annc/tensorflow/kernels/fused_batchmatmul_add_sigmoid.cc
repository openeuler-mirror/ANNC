/* Copyright 2026 The Huawei Technologies Co. Authors. All Rights Reserved.

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
#include <array>
#include <cmath>
#include <cstring>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

using namespace tensorflow;

namespace {

constexpr int kFastPathLaneSize = 4;
constexpr int kFastPathMaxVecCount = 4;
constexpr int kFastPathMaxCols = kFastPathLaneSize * kFastPathMaxVecCount;

template <int VecCount>
struct FastPathTraits {
  static constexpr int kRowTile = 4;
};

template <>
struct FastPathTraits<1> {
  static constexpr int kRowTile = 16;
};

template <>
struct FastPathTraits<2> {
  static constexpr int kRowTile = 8;
};

template <>
struct FastPathTraits<3> {
  static constexpr int kRowTile = 4;
};

template <>
struct FastPathTraits<4> {
  static constexpr int kRowTile = 4;
};

#ifdef __ARM_NEON
inline float32x4_t FusedMulAdd(float32x4_t acc, float32x4_t rhs, float lhs) {
#if defined(__aarch64__)
  return vfmaq_n_f32(acc, rhs, lhs);
#else
  return vmlaq_n_f32(acc, rhs, lhs);
#endif
}

inline float32x4_t SigmoidEigenPacket(float32x4_t values) {
  return Eigen::internal::scalar_logistic_op<float>().packetOp(values);
}
#endif

#ifdef __ARM_NEON
template <int VecCount, int Vec = 0>
struct CopyBiasPackets {
  static inline void Run(
      float32x4_t* acc_row,
      const std::array<float32x4_t, VecCount>& bias_vecs) {
    acc_row[Vec] = bias_vecs[Vec];
    CopyBiasPackets<VecCount, Vec + 1>::Run(acc_row, bias_vecs);
  }
};

template <int VecCount>
struct CopyBiasPackets<VecCount, VecCount> {
  static inline void Run(float32x4_t*,
                         const std::array<float32x4_t, VecCount>&) {}
};

template <int RowTile, int VecCount, int Row = 0>
struct InitAccumulatorRows {
  static inline void Run(
      float32x4_t (&acc)[RowTile][VecCount],
      const std::array<float32x4_t, VecCount>& bias_vecs) {
    CopyBiasPackets<VecCount>::Run(acc[Row], bias_vecs);
    InitAccumulatorRows<RowTile, VecCount, Row + 1>::Run(acc, bias_vecs);
  }
};

template <int RowTile, int VecCount>
struct InitAccumulatorRows<RowTile, VecCount, RowTile> {
  static inline void Run(
      float32x4_t (&)[RowTile][VecCount],
      const std::array<float32x4_t, VecCount>&) {}
};

template <int VecCount, int Vec = 0>
struct LoadRhsPackets {
  static inline void Run(const float* rhs_row, float32x4_t (&rhs_vecs)[VecCount]) {
    rhs_vecs[Vec] = vld1q_f32(rhs_row + Vec * kFastPathLaneSize);
    LoadRhsPackets<VecCount, Vec + 1>::Run(rhs_row, rhs_vecs);
  }
};

template <int VecCount>
struct LoadRhsPackets<VecCount, VecCount> {
  static inline void Run(const float*, float32x4_t (&)[VecCount]) {}
};

template <int VecCount, int Vec = 0>
struct FmaVecLoop {
  static inline void Run(float32x4_t* acc_row,
                         const float32x4_t (&rhs_vecs)[VecCount],
                         float lhs_value) {
    acc_row[Vec] = FusedMulAdd(acc_row[Vec], rhs_vecs[Vec], lhs_value);
    FmaVecLoop<VecCount, Vec + 1>::Run(acc_row, rhs_vecs, lhs_value);
  }
};

template <int VecCount>
struct FmaVecLoop<VecCount, VecCount> {
  static inline void Run(float32x4_t*, const float32x4_t (&)[VecCount], float) {}
};

template <int RowTile, int VecCount, int Row = 0>
struct FmaRowLoop {
  static inline void Run(float32x4_t (&acc)[RowTile][VecCount],
                         const float32x4_t (&rhs_vecs)[VecCount],
                         const float* lhs_block, int64 k, int64 kk) {
    const float lhs_value = lhs_block[Row * k + kk];
    FmaVecLoop<VecCount>::Run(acc[Row], rhs_vecs, lhs_value);
    FmaRowLoop<RowTile, VecCount, Row + 1>::Run(acc, rhs_vecs, lhs_block, k,
                                                kk);
  }
};

template <int RowTile, int VecCount>
struct FmaRowLoop<RowTile, VecCount, RowTile> {
  static inline void Run(float32x4_t (&)[RowTile][VecCount],
                         const float32x4_t (&)[VecCount], const float*, int64,
                         int64) {}
};

template <int VecCount, int Vec = 0>
struct StoreActivatedPackets {
  static inline void Run(float* out_row, const float32x4_t* acc_row) {
    vst1q_f32(out_row + Vec * kFastPathLaneSize,
              SigmoidEigenPacket(acc_row[Vec]));
    StoreActivatedPackets<VecCount, Vec + 1>::Run(out_row, acc_row);
  }
};

template <int VecCount>
struct StoreActivatedPackets<VecCount, VecCount> {
  static inline void Run(float*, const float32x4_t*) {}
};

template <int RowTile, int VecCount, int Row = 0>
struct StoreActivatedRows {
  static inline void Run(float* out_block,
                         float32x4_t (&acc)[RowTile][VecCount]) {
    StoreActivatedPackets<VecCount>::Run(
        out_block + Row * VecCount * kFastPathLaneSize, acc[Row]);
    StoreActivatedRows<RowTile, VecCount, Row + 1>::Run(out_block, acc);
  }
};

template <int RowTile, int VecCount>
struct StoreActivatedRows<RowTile, VecCount, RowTile> {
  static inline void Run(float*, float32x4_t (&)[RowTile][VecCount]) {}
};
#endif

struct BiasInfo {
  int64 dims[3] = {1, 1, 1};
  int64 strides[3] = {0, 0, 0};
};

inline float SigmoidEigenScalar(float x) {
  return Eigen::internal::scalar_logistic_op<float>()(x);
}

Status BuildBiasInfo(const Tensor& bias, int64 batch, int64 m, int64 n,
                     BiasInfo* info) {
  for (int i = 0; i < bias.dims(); ++i) {
    info->dims[3 - bias.dims() + i] = bias.dim_size(i);
  }

  if (!((info->dims[0] == 1 || info->dims[0] == batch) &&
        (info->dims[1] == 1 || info->dims[1] == m) &&
        (info->dims[2] == 1 || info->dims[2] == n))) {
    return errors::InvalidArgument("bias shape is not broadcast-compatible");
  }

  info->strides[2] = (info->dims[2] == 1) ? 0 : 1;
  info->strides[1] = (info->dims[1] == 1) ? 0 : info->dims[2];
  info->strides[0] =
      (info->dims[0] == 1) ? 0 : info->dims[1] * info->dims[2];
  return OkStatus();
}

void ComputeGeneric(const float* lhs_ptr, const float* rhs_ptr,
                    const float* bias_ptr, const BiasInfo& bias, int64 batch,
                    int64 m, int64 k, int64 n, bool rhs_is_batched,
                    float* out_ptr) {
  const int64 lhs_batch_stride = m * k;
  const int64 rhs_batch_stride = rhs_is_batched ? k * n : 0;
  const int64 out_batch_stride = m * n;

  for (int64 b = 0; b < batch; ++b) {
    const float* lhs_batch = lhs_ptr + b * lhs_batch_stride;
    const float* rhs_batch = rhs_ptr + b * rhs_batch_stride;
    float* out_batch = out_ptr + b * out_batch_stride;

    for (int64 i = 0; i < m; ++i) {
      for (int64 j_block = 0; j_block < n; j_block += 4) {
        const int64 width = std::min<int64>(4, n - j_block);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int64 t = 0; t < width; ++t) {
          const int64 bias_offset = b * bias.strides[0] +
                                    i * bias.strides[1] +
                                    (j_block + t) * bias.strides[2];
          acc[t] = bias_ptr[bias_offset];
        }

        for (int64 kk = 0; kk < k; ++kk) {
          const float lhs_value = lhs_batch[i * k + kk];
          const float* rhs_row = rhs_batch + kk * n + j_block;
          for (int64 t = 0; t < width; ++t) {
            acc[t] += lhs_value * rhs_row[t];
          }
        }

        for (int64 t = 0; t < width; ++t) {
          out_batch[i * n + j_block + t] = SigmoidEigenScalar(acc[t]);
        }
      }
    }
  }
}

bool CanUseFastPath(const Tensor& rhs, const Tensor& bias, int64 m, int64 k,
                    int64 n) {
  // Target the common "feature squeeze" stages: small-N rhs shared across batch.
  return rhs.dims() == 2 && bias.dims() == 1 && bias.dim_size(0) == n &&
         m <= 32 && k <= 64 && n > 0 && n <= kFastPathMaxCols &&
         (n % kFastPathLaneSize) == 0;
}

template <int VecCount>
void ComputeFastTileScalar(const float* lhs_block, const float* rhs_ptr,
                           const float* bias_ptr, int64 rows, int64 k,
                           float* out_block) {
  constexpr int kCols = VecCount * kFastPathLaneSize;

  for (int64 r = 0; r < rows; ++r) {
    alignas(64) float acc[kFastPathMaxCols];
    std::memcpy(acc, bias_ptr, kCols * sizeof(float));
    const float* lhs_row = lhs_block + r * k;

    for (int64 kk = 0; kk < k; ++kk) {
      const float lhs_value = lhs_row[kk];
      const float* rhs_row = rhs_ptr + kk * kCols;
      for (int v = 0; v < VecCount; ++v) {
        for (int lane = 0; lane < kFastPathLaneSize; ++lane) {
          acc[v * kFastPathLaneSize + lane] +=
              lhs_value * rhs_row[v * kFastPathLaneSize + lane];
        }
      }
    }

    float* out_row = out_block + r * kCols;
    for (int c = 0; c < kCols; ++c) {
      out_row[c] = SigmoidEigenScalar(acc[c]);
    }
  }
}

#ifdef __ARM_NEON
template <int VecCount, int RowTile>
void ComputeFastTileNeon(const float* lhs_block, const float* rhs_ptr,
    const std::array<float32x4_t, VecCount>& bias_vecs, int64 k,
    float* out_block) {
  constexpr int kCols = VecCount * kFastPathLaneSize;
  float32x4_t acc[RowTile][VecCount];
  InitAccumulatorRows<RowTile, VecCount>::Run(acc, bias_vecs);

  int64 kk = 0;
  for (; kk + 3 < k; kk += 4) {
    for (int unroll = 0; unroll < 4; ++unroll) {
      float32x4_t rhs_vecs[VecCount];
      const float* rhs_row = rhs_ptr + (kk + unroll) * kCols;
      LoadRhsPackets<VecCount>::Run(rhs_row, rhs_vecs);
      FmaRowLoop<RowTile, VecCount>::Run(acc, rhs_vecs, lhs_block, k,
                                         kk + unroll);
    }
  }

  for (; kk < k; ++kk) {
    float32x4_t rhs_vecs[VecCount];
    const float* rhs_row = rhs_ptr + kk * kCols;
    LoadRhsPackets<VecCount>::Run(rhs_row, rhs_vecs);
    FmaRowLoop<RowTile, VecCount>::Run(acc, rhs_vecs, lhs_block, k, kk);
  }
  StoreActivatedRows<RowTile, VecCount>::Run(out_block, acc);
}
#endif

template <int VecCount>
void ComputeFastPathImpl(OpKernelContext* /*ctx*/, const float* lhs_ptr,
                         const float* rhs_ptr, const float* bias_ptr,
                         int64 batch, int64 m, int64 k, float* out_ptr) {
  constexpr int kRowTile = FastPathTraits<VecCount>::kRowTile;
  constexpr int kCols = VecCount * kFastPathLaneSize;
  const int64 row_tiles = (m + kRowTile - 1) / kRowTile;

#ifdef __ARM_NEON
  std::array<float32x4_t, VecCount> bias_vecs;
  for (int v = 0; v < VecCount; ++v) {
    bias_vecs[v] = vld1q_f32(bias_ptr + v * kFastPathLaneSize);
  }
#endif

  for (int64 b = 0; b < batch; ++b) {
    for (int64 tile = 0; tile < row_tiles; ++tile) {
      const int64 row_start = tile * kRowTile;
      const int64 rows = std::min<int64>(kRowTile, m - row_start);
      const float* lhs_block = lhs_ptr + b * m * k + row_start * k;
      float* out_block = out_ptr + b * m * kCols + row_start * kCols;

#ifdef __ARM_NEON
      if (rows == kRowTile) {
        ComputeFastTileNeon<VecCount, kRowTile>(lhs_block, rhs_ptr, bias_vecs, k,
                                                out_block);
        continue;
      }
#endif

      ComputeFastTileScalar<VecCount>(lhs_block, rhs_ptr, bias_ptr, rows, k,
                                      out_block);
    }
  }
}

void ComputeFastPath(OpKernelContext* ctx, const float* lhs_ptr,
                     const float* rhs_ptr, const float* bias_ptr, int64 batch,
                     int64 m, int64 k, int64 n, float* out_ptr) {
  const int vec_count = static_cast<int>(n / kFastPathLaneSize);
  switch (vec_count) {
    case 1:
      ComputeFastPathImpl<1>(ctx, lhs_ptr, rhs_ptr, bias_ptr, batch, m, k,
                             out_ptr);
      break;
    case 2:
      ComputeFastPathImpl<2>(ctx, lhs_ptr, rhs_ptr, bias_ptr, batch, m, k,
                             out_ptr);
      break;
    case 3:
      ComputeFastPathImpl<3>(ctx, lhs_ptr, rhs_ptr, bias_ptr, batch, m, k,
                             out_ptr);
      break;
    case 4:
      ComputeFastPathImpl<4>(ctx, lhs_ptr, rhs_ptr, bias_ptr, batch, m, k,
                             out_ptr);
      break;
    default:
      break;
  }
}

}  // namespace

class KPFusedBatchMatMulAddSigmoidOp : public OpKernel {
 public:
  explicit KPFusedBatchMatMulAddSigmoidOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& lhs = ctx->input(0);
    const Tensor& rhs = ctx->input(1);
    const Tensor& bias = ctx->input(2);

    OP_REQUIRES(ctx, lhs.dims() == 3,
                errors::InvalidArgument("lhs must be rank 3"));
    OP_REQUIRES(ctx, rhs.dims() == 2 || rhs.dims() == 3,
                errors::InvalidArgument("rhs must be rank 2 or 3"));
    OP_REQUIRES(ctx, bias.dims() <= 3,
                errors::InvalidArgument("bias rank must be <= 3"));

    const int64 batch = lhs.dim_size(0);
    const int64 m = lhs.dim_size(1);
    const int64 k = lhs.dim_size(2);
    const bool rhs_is_batched = rhs.dims() == 3;

    OP_REQUIRES(ctx, !rhs_is_batched || rhs.dim_size(0) == batch,
                errors::InvalidArgument("lhs/rhs batch mismatch"));
    OP_REQUIRES(ctx, rhs.dim_size(rhs_is_batched ? 1 : 0) == k,
                errors::InvalidArgument("lhs/rhs contracting dim mismatch"));

    const int64 n = rhs.dim_size(rhs_is_batched ? 2 : 1);
    TensorShape out_shape({batch, m, n});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));

    const float* lhs_ptr = lhs.flat<float>().data();
    const float* rhs_ptr = rhs.flat<float>().data();
    const float* bias_ptr = bias.flat<float>().data();
    float* out_ptr = output->flat<float>().data();
    BiasInfo bias_info;
    OP_REQUIRES_OK(ctx, BuildBiasInfo(bias, batch, m, n, &bias_info));

    if (CanUseFastPath(rhs, bias, m, k, n)) {
      ComputeFastPath(ctx, lhs_ptr, rhs_ptr, bias_ptr, batch, m, k, n, out_ptr);
    } else {
      ComputeGeneric(lhs_ptr, rhs_ptr, bias_ptr, bias_info, batch, m, k, n,
                     rhs_is_batched, out_ptr);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("KPFusedBatchMatMulAddSigmoid").Device(DEVICE_CPU),
                        KPFusedBatchMatMulAddSigmoidOp);
