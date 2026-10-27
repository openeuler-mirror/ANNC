#include "Context/KernelContextUtils.h"
#include "Context/RowParallel.h"

namespace {

AnncStatusCode matmul_aarch64_impl(annc::kernels::KernelContext& context) {
  const auto* lhs = context.getInput(0);
  const auto* rhs = context.getInput(1);
  if (!lhs || !rhs) return context.status();
  if (lhs->sizes[1] != rhs->sizes[0]) return ANNC_STATUS_INVALID_ARGUMENT;
  auto* output =
      context.allocateOutput<float>(0, {lhs->sizes[0], rhs->sizes[1]});
  if (!output) return context.status();
  float* out = annc::kernels::contextData<float>(*output);
  const float* A = annc::kernels::contextData<float>(*lhs);
  const float* B = annc::kernels::contextData<float>(*rhs);

  int64_t M = output->sizes[0];
  int64_t N = output->sizes[1];
  int64_t K = lhs->sizes[1];

  auto workers = context.getThreadPool();
  return annc::kernels::Shard(workers.numThreads(), workers, M,
                              annc::threadpool::cost::dotRow(K, N),
                              [&](int64_t begin, int64_t end) {
                                for (int64_t i = begin; i < end; ++i) {
                                  for (int64_t j = 0; j < N; ++j) {
                                    float sum = 0.0f;
                                    for (int64_t k = 0; k < K; ++k) {
                                      sum += A[i * K + k] * B[k * N + j];
                                    }
                                    out[i * N + j] = sum;
                                  }
                                }
                              });
}

}  // namespace
