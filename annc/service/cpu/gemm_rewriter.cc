#include "annc/service/kdnn_util.h"
#include "kdnn_rewriter.h"

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
  RewritePattern pattern("__matmul_relu_relu", HloOpcode::kDot);
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
  register_matmul(rewriters, RewriterType::FUSED_OPERATION);
  register_batch_matmul(rewriters, RewriterType::FUSED_OPERATION);
  register_matmul_add(rewriters, RewriterType::FUSED_OPERATION, 2);
  register_matmul_add_relu(rewriters, RewriterType::FUSED_OPERATION, 3);

  std::sort(rewriters.begin(), rewriters.end(), compare_rewriter);
}


void __matmul(void* out, const void** in) {
  float* out_buf = reinterpret_cast<float*>(out);
  const float* lhs = reinterpret_cast<const float*>(in[0]);
  const float* rhs = reinterpret_cast<const float*>(in[1]);
  int m = *reinterpret_cast<const int*>(in[2]);
  int k = *reinterpret_cast<const int*>(in[3]);
  int n = *reinterpret_cast<const int*>(in[4]);

  int lda = m;
  int ldb = k;
  int ldc = m;

  int alpha = 1;
  int beta = 0;

#if defined(ANNC_ENABLED_KDNN)
  CBLAS_LAYOUT clayout = CblasColMajor;
  CBLAS_TRANSPOSE transa = CblasNoTrans;
  CBLAS_TRANSPOSE transb = CblasNoTrans;
  if (trans_b) transb = CblasTrans;

  // C = alpha * A * B + beta * C;
  cblas_sgemm(clayout, transa, transb, m, n, k, alpha, lhs, lda, rhs, ldb, beta,
              out_buf, ldc);
#endif
}

void __batch_matmul(void* out, const void** in) {
  float* out_buf = reinterpret_cast<float*>(out);
  const float* lhs = reinterpret_cast<const float*>(in[0]);
  const float* rhs = reinterpret_cast<const float*>(in[1]);
  int batch = *reinterpret_cast<const int*>(in[2]);
  int m = *reinterpret_cast<const int*>(in[3]);
  int k = *reinterpret_cast<const int*>(in[4]);
  int n = *reinterpret_cast<const int*>(in[5]);

  int lda = m;
  int ldb = k;
  int ldc = m;

  int alpha = 1;
  int beta = 0;

#if defined(ANNC_ENABLED_KDNN)
  CBLAS_LAYOUT clayout = CblasColMajor;
  CBLAS_TRANSPOSE transa = CblasNoTrans;
  CBLAS_TRANSPOSE transb = CblasNoTrans;

  for (int i = 0; i < batch; ++i) {
    float* A = const_cast<float*>(&lhs[i * m * k]);
    float* B = const_cast<float*>(&rhs[i * k * n]);
    float* C = &out_buf[i * m * n];
    cblas_sgemm(clayout, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta,
                C, ldc);
  }
#endif
}

void __matmul_add(void* out, const void** in) {}

void register_gemm_kernels() {
  XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(__matmul);
  XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(__batch_matmul);
  XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(__matmul_add);
}

}  // namespace cpu
}  // namespace xla