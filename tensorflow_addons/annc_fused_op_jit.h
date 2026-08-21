#ifndef TENSORFLOW_ADDON_ANNC_FUSED_OP_JIT_H_
#define TENSORFLOW_ADDON_ANNC_FUSED_OP_JIT_H_

#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

struct AnncJitCompileRequest {
  std::string atir_module_path;
  std::string kernel_name;
  std::vector<std::vector<int64_t>> argument_shapes;
};

Status CompileAnncJitKernel(const AnncJitCompileRequest& request,
                            std::string* work_dir, std::string* so_path);
void CleanupAnncJitWorkDir(const std::string& work_dir);

}  // namespace tensorflow

#endif  // TENSORFLOW_ADDON_ANNC_FUSED_OP_JIT_H_
