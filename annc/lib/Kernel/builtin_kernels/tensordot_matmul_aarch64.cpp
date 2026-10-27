#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#if defined(ANNC_KERNEL_MLAS)
// ORT's MLAS public header.  It is self-contained (forward declarations only:
// no ORT thread-pool headers, no Eigen), and in this ORT release
// MLAS_THREADPOOL is a type alias of onnxruntime::concurrency::ThreadPool, so
// nativePool("onnxruntime") hands over the pool object MlasGemmBatch wants.  The build supplies this include directory and the
// archives (annc/lib/Kernel/builtin_kernels/CMakeLists.txt).
#include "mlas.h"
#endif

#include "Context/KernelContextUtils.h"
#include "Kernel/PackedB.h"
#include "Context/RowParallel.h"

namespace {

// FusedTensordotMatMul, context-v3 form.
//
//   input : rank-3 [d0, d1, K] (contiguous row-major)
//   weight: rank-2 [K, N]
//   output: rank-3 [d0, d1, N] -- allocated here through the execution context
//           (slot 0) because the fused output's sizes follow the input's
//           leading dims, which are runtime values; the shape is only known
//           after reading the input descriptor.
//
//   out[i, j, n] = sum_k input[i, j, k] * weight[k, n]
//
// The fused region being replaced is the ONNX Tensordot block
// Reshape(MatMul(Reshape(x, [d0*d1, K]), W), [d0, d1, N]) plus the shape
// arithmetic that computes the two Reshape targets — 16 ops in ATIR, of which
// 15 are metadata (Transpose, the twelve shape-chain ops, and the two
// Reshapes).  In row-major layout collapsing the leading dimensions moves no
// data, so all of them are pure metadata: the fused op is the same matmul
// over the unchanged input buffer, with m = d0 * d1.  This mirrors ORT's
// FusedTensordotMatMul kernel
// (onnxruntime/contrib_ops/cpu/fused_tensordot_matmul.cc).
AnncStatusCode tensordotMatMulImpl(annc::kernels::KernelContext& context) {
  const auto* input = context.getInput(0);
  const auto* weight = context.getInput(1);
  if (!input || !weight) return context.status();
  auto thread_pool = context.getThreadPool();

  const int64_t d0 = input->sizes[0];
  const int64_t d1 = input->sizes[1];
  const int64_t k = input->sizes[2];
  const int64_t n = weight->sizes[1];
  if (d0 < 0 || d1 < 0 || k < 0 || n < 0) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }
  // The pattern only fuses sites whose contract dim matches the weight's
  // first dim; re-check here so a mis-specialized call fails loudly instead
  // of producing a wrong result.
  if (weight->sizes[0] != k) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }
  // The kernel indexes both operands as contiguous row-major (x + row * k,
  // w + t * n). The host builds the descriptors as contiguous row-major
  // (addons/onnxruntime/context/annc_ep_execution.cc), so this check guards
  // the host/kernel contract rather than an arbitrary input layout:
  // non-contiguous inputs are not supported and must be rejected here.
  if (input->strides[2] != 1 || input->strides[1] != k ||
      input->strides[0] != d1 * k || weight->strides[1] != 1 ||
      weight->strides[0] != n) {
    return ANNC_STATUS_INVALID_ARGUMENT;
  }

  const int64_t m = d0 * d1;
  // Output shape [d0, d1, n] is data-dependent (the leading dims come from the
  // input descriptor), so it is requested from the execution context rather
  // than received as a preallocated argument.  Allocate in every case -- also
  // when m or n is zero -- because the host checks that the call allocated
  // every declared output slot.
  auto output = context.allocateOutput<float>(0, {d0, d1, n});
  if (!output) {
    return ANNC_STATUS_ALLOCATION_FAILED;
  }
  if (m == 0 || n == 0) {
    return ANNC_STATUS_OK;
  }

  float* out = annc::kernels::contextData<float>(*output);
  if (k == 0) {
    for (int64_t i = 0; i < m * n; ++i) {
      out[i] = 0.0f;
    }
    return ANNC_STATUS_OK;
  }

  const float* x = annc::kernels::contextData<float>(*input);
  const float* w = annc::kernels::contextData<float>(*weight);

#if defined(ANNC_KERNEL_MLAS)
  // The host's native pool, when it has one to hand over (the ORT EP does:
  // ORT's operator pool, which is the object MlasGemmBatch takes).  One SGEMM
  // over the collapsed [m, K] x [K, N] view -- the same call shape ORT's own
  // FusedTensordotMatMul makes through math::MatMul with
  // GetOperatorThreadPool() -- so MLAS owns the tiling, the kernel choice and
  // the threading.  Without a native pool (the scalar-only build, another
  // host, or the version gate refusing) the row-parallel scalar path below
  // still runs.
  {
    if (void* native = thread_pool.nativePool("onnxruntime"); native != nullptr) {
      MLAS_SGEMM_DATA_PARAMS gemm;
      gemm.A = x;
      gemm.lda = static_cast<size_t>(k);
      const auto* packed = annc::kernels::contextPackedB(context, *weight);
      const float* packed_weight = annc::kernels::packedBSlice(packed, 0, 0);
      gemm.B = packed_weight != nullptr ? packed_weight : w;
      gemm.ldb = static_cast<size_t>(n);
      gemm.C = out;
      gemm.ldc = static_cast<size_t>(n);
      gemm.BIsPacked = packed_weight != nullptr;
      if (packed_weight != nullptr &&
          annc::kernels::packedBDebugEnabled()) {
        static std::atomic<int> packed_printed{0};
        if (packed_printed.fetch_add(1) == 0) {
          std::fprintf(stderr, "[annc-kernel] ftdmm: packed-B selected\\n");
        }
      }
      MlasGemmBatch(CblasNoTrans, CblasNoTrans, static_cast<size_t>(m),
                    static_cast<size_t>(n), static_cast<size_t>(k), &gemm,
                    /*BatchSize=*/1,
                    reinterpret_cast<MLAS_THREADPOOL*>(native),
                    /*BackendKernelSelectorConfig=*/nullptr);
      if (annc::kernels::packedBDebugEnabled()) {
        static std::atomic<int> printed{0};
        if (printed.fetch_add(1) < 4) {
          std::fprintf(stderr,
                       "[annc-kernel] ftdmm: MlasGemmBatch m=%lld n=%lld "
                       "k=%lld\n",
                       static_cast<long long>(m), static_cast<long long>(n),
                       static_cast<long long>(k));
        }
      }
      return ANNC_STATUS_OK;
    }
  }
#endif

  // Rows of the collapsed [m, K] view are independent, so the row loop is the
  // natural parallel split (same shape as the builtin MatMul kernel in
  // matmul_aarch64.cpp).  M is tiny at the sites this fusion targets
  // (batch * sequence), so the reported row cost -- the same as
  // batched_matmul_add's, a k-length dot per output element -- keeps one
  // thread on the sites whose k*n is small instead of paying for a dispatch.
  auto options = annc::threadpool::rowOptions(
      annc::threadpool::cost::dotRow(k, n), /*elements_per_row=*/k * n);
  annc::kernels::Shard(
      thread_pool.numThreads(), thread_pool, m, options.cost_per_unit.value_or(0),
      [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
          const float* xRow = x + row * k;
          float* outRow = out + row * n;
          for (int64_t col = 0; col < n; ++col) {
            const float* wCol = w + col;
            float sum = 0.0f;
            for (int64_t t = 0; t < k; ++t) {
              sum += xRow[t] * wCol[t * n];
            }
            outRow[col] = sum;
          }
        }
      });

  return ANNC_STATUS_OK;
}

}  // namespace
