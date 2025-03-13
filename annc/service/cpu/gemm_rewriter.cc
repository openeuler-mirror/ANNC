#include <algorithm>
#include <memory>

#include "annc/service/kdnn_util.h"
#include "kdnn_rewriter.h"

#if defined(ANNC_ENABLED_KDNN)
#include <kblas/kblas.h>
#endif

#if defined(ANNC_ENABLED_OPENBLAS)
#include <openblas/cblas.h>
#endif

namespace xla {
namespace cpu {

void register_matmul(std::vector<KDnnRewriter>& rewriters,
                     RewriterType rewrite_type, int benefit = 1) {
  RewritePattern pattern("__matmul", HloOpcode::kDot);
  pattern.dtypes = {PrimitiveType::F32, PrimitiveType::F32};
  pattern.dims = {2, 2};
  auto rewriter = KDnnRewriter(benefit, pattern, rewrite_type);
  rewriters.push_back(rewriter);
}

void register_batch_matmul(std::vector<KDnnRewriter>& rewriters,
                           RewriterType rewrite_type, int benefit = 1) {
  RewritePattern pattern("__batch_matmul", HloOpcode::kDot);
  pattern.dtypes = {PrimitiveType::F32, PrimitiveType::F32};
  pattern.dims = {3, 3};
  auto rewriter = KDnnRewriter(benefit, pattern, rewrite_type);
  rewriters.push_back(rewriter);
}

void register_matmul_add(std::vector<KDnnRewriter>& rewriters,
                         RewriterType rewrite_type, int benefit = 1) {
  RewritePattern pattern("__matmul_add", HloOpcode::kDot);
  pattern.dtypes = {PrimitiveType::F32, PrimitiveType::F32};
  pattern.dims = {2, 2};
  RewritePattern pattern_1("", HloOpcode::kAdd);
  pattern_1.dtypes = {PrimitiveType::F32, PrimitiveType::F32};
  pattern.next_patterns = {pattern_1};
  auto rewriter = KDnnRewriter(benefit, pattern, rewrite_type);
  rewriters.push_back(rewriter);
}

void register_matmul_add_relu(std::vector<KDnnRewriter>& rewriters,
                              RewriterType rewrite_type, int benefit = 1) {
  RewritePattern pattern("__matmul_add_relu", HloOpcode::kDot);
  pattern.dtypes = {PrimitiveType::F32, PrimitiveType::F32};
  pattern.dims = {2, 2};

  RewritePattern pattern_1("", HloOpcode::kAdd);
  pattern_1.dtypes = {PrimitiveType::F32, PrimitiveType::F32};
  pattern.next_patterns = {pattern_1};

  RewritePattern pattern_1_1("", HloOpcode::kMaximum);
  pattern_1_1.dtypes = {PrimitiveType::F32, PrimitiveType::F32};
  auto& add = pattern.next_patterns.at(0);
  add.next_patterns = {pattern_1_1};

  auto rewriter = KDnnRewriter(benefit, pattern, rewrite_type);
  rewriters.push_back(rewriter);
}

void register_gemm_rewriters(std::vector<KDnnRewriter>& rewriters) {
#if defined(ANNC_ENABLED_KDNN) || defined(ANNC_ENABLED_OPENBLAS)
  register_matmul(rewriters, RewriterType::DNN_CUSTOM_CALL);
  register_batch_matmul(rewriters, RewriterType::DNN_CUSTOM_CALL);
  // register_matmul_add(rewriters, RewriterType::DNN_CUSTOM_CALL, 2);
  // register_matmul_add_relu(rewriters, RewriterType::DNN_CUSTOM_CALL, 3);
#endif
  std::sort(rewriters.begin(), rewriters.end(), compare_rewriter);
}

void __matmul(void* out, const void** in) {
  float* out_buf = reinterpret_cast<float*>(out);
  const float* lhs = reinterpret_cast<const float*>(in[0]);
  const float* rhs = reinterpret_cast<const float*>(in[1]);
  const int64_t* lhs_shape = reinterpret_cast<const int64_t*>(in[2]);
  const int64_t* rhs_shape = reinterpret_cast<const int64_t*>(in[3]);
  int m = lhs_shape[0];
  int k = lhs_shape[1];
  int n = rhs_shape[1];

#if defined(ANNC_ENABLED_KDNN) || defined(ANNC_ENABLED_OPENBLAS)
  CBLAS_LAYOUT clayout = CblasRowMajor;
  CBLAS_TRANSPOSE transa = CblasNoTrans;
  CBLAS_TRANSPOSE transb = CblasNoTrans;

  int lda = k;
  int ldb = n;
  int ldc = n;

  float alpha = 1.0f;
  float beta = 0.0f;

  // C = alpha * A * B + beta * C;
  cblas_sgemm(clayout, transa, transb, m, n, k, alpha, lhs, lda, rhs, ldb, beta,
              out_buf, ldc);
#endif
}

void __batch_matmul(void* out, const void** in) {
  float* out_buf = reinterpret_cast<float*>(out);
  const float* lhs = reinterpret_cast<const float*>(in[0]);
  const float* rhs = reinterpret_cast<const float*>(in[1]);
  const int64_t* lhs_shape = reinterpret_cast<const int64_t*>(in[2]);
  const int64_t* rhs_shape = reinterpret_cast<const int64_t*>(in[3]);
  int batch = lhs_shape[0];
  int m = lhs_shape[1];
  int k = lhs_shape[2];
  int n = rhs_shape[2];

#if defined(ANNC_ENABLED_KDNN) || defined(ANNC_ENABLED_OPENBLAS)
  CBLAS_LAYOUT clayout = CblasRowMajor;
  CBLAS_TRANSPOSE transa = CblasNoTrans;
  CBLAS_TRANSPOSE transb = CblasNoTrans;

  int lda = k;
  int ldb = n;
  int ldc = n;

  float alpha = 1.0f;
  float beta = 0.0f;

  for (int i = 0; i < batch; ++i) {
    float* A = const_cast<float*>(&lhs[i * m * k]);
    float* B = const_cast<float*>(&rhs[i * k * n]);
    float* C = &out_buf[i * m * n];
    cblas_sgemm(clayout, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta,
                C, ldc);
  }
#endif
}

void __matmul_add(void* out, const void** in) {
  float* out_buf = reinterpret_cast<float*>(out);
  const float* elem_val = reinterpret_cast<const float*>(in[0]);
  const float* lhs = reinterpret_cast<const float*>(in[1]);
  const float* rhs = reinterpret_cast<const float*>(in[2]);

  const int64_t* lhs_shape = reinterpret_cast<const int64_t*>(in[4]);
  const int64_t* rhs_shape = reinterpret_cast<const int64_t*>(in[5]);

  int64_t m = lhs_shape[0];
  int64_t k = lhs_shape[1];
  int64_t n = rhs_shape[1];

#if defined(ANNC_ENABLED_KDNN) || defined(ANNC_ENABLED_OPENBLAS)
  CBLAS_LAYOUT clayout = CblasRowMajor;
  CBLAS_TRANSPOSE transa = CblasNoTrans;
  CBLAS_TRANSPOSE transb = CblasNoTrans;

  int lda = k;
  int ldb = n;
  int ldc = n;

  float alpha = 1.0f;
  float beta = 1.0f;

  memcpy(out_buf, elem_val, sizeof(float) * m * n);

  // C = alpha * A * B + beta * C;
  cblas_sgemm(clayout, transa, transb, m, n, k, alpha, lhs, lda, rhs, ldb, beta,
              out_buf, ldc);
#endif
}

void __matmul_add_relu(void* out, const void** in) {
  float* out_buf = reinterpret_cast<float*>(out);
  const float* max_val = reinterpret_cast<const float*>(in[0]);
  const float* elem_val = reinterpret_cast<const float*>(in[1]);
  const float* lhs = reinterpret_cast<const float*>(in[2]);
  const float* rhs = reinterpret_cast<const float*>(in[3]);
  const int64_t* lhs_shape = reinterpret_cast<const int64_t*>(in[6]);
  const int64_t* rhs_shape = reinterpret_cast<const int64_t*>(in[7]);

  int64_t m = lhs_shape[0];
  int64_t k = lhs_shape[1];
  int64_t n = rhs_shape[1];

#if defined(ANNC_ENABLED_KDNN) || defined(ANNC_ENABLED_OPENBLAS)
  CBLAS_LAYOUT clayout = CblasRowMajor;
  CBLAS_TRANSPOSE transa = CblasNoTrans;
  CBLAS_TRANSPOSE transb = CblasNoTrans;

  int lda = k;
  int ldb = n;
  int ldc = n;

  float alpha = 1.0f;
  float beta = 1.0f;

  memcpy(out_buf, elem_val, sizeof(float) * m * n);

  // C = alpha * A * B + beta * C;
  cblas_sgemm(clayout, transa, transb, m, n, k, alpha, lhs, lda, rhs, ldb, beta,
              out_buf, ldc);
  for (int i = 0; i < m * n; ++i) {
    out_buf[i] = std::max(out_buf[i], 0.0f);
  }
#endif
}

}  // namespace cpu
}  // namespace xla