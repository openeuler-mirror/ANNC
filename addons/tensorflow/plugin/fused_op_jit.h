#ifndef ANNC_TENSORFLOW_PLUGIN_FUSED_OP_JIT_H
#define ANNC_TENSORFLOW_PLUGIN_FUSED_OP_JIT_H

#include <cstdint>
#include <string>
#include <vector>

#include "jit_cache.h"
#include "compat.h"

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

#endif  // ANNC_TENSORFLOW_PLUGIN_FUSED_OP_JIT_H
