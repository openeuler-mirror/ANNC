#include <algorithm>
#include <cmath>

#include "annc/service/kdnn_util.h"
#include "kdnn_rewriter.h"

#if defined(ANNC_ENABLED_KDNN)
#include <openblas/cblas.h>
#endif

namespace xla {
namespace cpu {

void register_reduce_mean(std::vector<KDnnRewriter>& rewriters,
                          RewriterType rewrite_type, int benefit = 1) {
  RewritePattern pattern("__reduce_mean", HloOpcode::kReduceWindow);
  pattern.dtypes = {PrimitiveType::F32, PrimitiveType::F32};
  pattern.dims = {2, 0};
  RewritePattern pattern_1("", HloOpcode::kReduce);
  pattern_1.dtypes = {PrimitiveType::F32, PrimitiveType::F32};
  pattern_1.dims = {2, 0};
  pattern.next_patterns = {pattern_1};
  auto rewriter = KDnnRewriter(benefit, pattern, rewrite_type);
  rewriters.push_back(rewriter);
}

void register_reduce_rewriters(std::vector<KDnnRewriter>& rewriters) {
  register_reduce_mean(rewriters, RewriterType::DNN_CUSTOM_CALL);

  std::sort(rewriters.begin(), rewriters.end(), compare_rewriter);
}

void __reduce_mean(void* out, const void** in) {
  float* out_buf = reinterpret_cast<float*>(out);
  float axis = *reinterpret_cast<const float*>(in[0]);
  const float* value = reinterpret_cast<const float*>(in[1]);
  const int64_t* shape = reinterpret_cast<const int64_t*>(in[3]);

  int m = shape[0];
  int n = shape[1];

#if defined(ANNC_ENABLED_KDNN)
  for (int i = 0; i < n; ++i) {
    out_buf[i] = cblas_sasum(m, value, 1) / m;
    value += m;
  }
#endif
}

}  // namespace cpu
}  // namespace xla