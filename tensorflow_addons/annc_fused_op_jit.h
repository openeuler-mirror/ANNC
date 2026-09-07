#ifndef TENSORFLOW_ADDON_ANNC_FUSED_OP_JIT_H_
#define TENSORFLOW_ADDON_ANNC_FUSED_OP_JIT_H_

#include <cstdint>
#include <string>
#include <vector>

#include "annc_jit_cache.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

struct AnncJitCompileRequest {
  std::string atir_module_path;
  std::string kernel_name;
  // Runtime intra-op thread budget for the GEMM plan; the TF intra-op pool is
  // fixed at session creation, so this is stable for the cache lifetime.
  int64_t intra_thread_count = 1;
  std::vector<std::vector<int64_t>> argument_shapes;
};

Status CompileAnncJitKernel(
    const AnncJitCompileRequest& request,
    std::shared_ptr<annc::jit::JitExecutable>* executable);

}  // namespace tensorflow

#endif  // TENSORFLOW_ADDON_ANNC_FUSED_OP_JIT_H_
