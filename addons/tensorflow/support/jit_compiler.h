#ifndef ANNC_TENSORFLOW_SUPPORT_JIT_COMPILER_H
#define ANNC_TENSORFLOW_SUPPORT_JIT_COMPILER_H

#include <string>
#include <vector>

#include "c_api.h"

// Private C++17 representation. Only support sources include this definition.
struct AnncJitKernel {
  std::string work_dir;
  std::string shared_library_path;
  std::string error;
  void* library_handle = nullptr;
  void* kernel_function = nullptr;
  void* set_thread_pool_function = nullptr;
  void* get_thread_pool_function = nullptr;
  bool keep_temps = false;

  ~AnncJitKernel();
};

namespace annc {
namespace jit {

struct JitCompileRequest {
  std::string atir_module_path;
  std::string kernel_name;
  int64_t intra_thread_count;
  std::vector<std::vector<int64_t>> argument_shapes;
  AnncJitLogCallback log;
  void* log_context;
};

AnncJitStatus CompileAnncJitKernel(const JitCompileRequest& request,
                                   AnncJitKernel* kernel);

}  // namespace jit
}  // namespace annc

#endif  // ANNC_TENSORFLOW_SUPPORT_JIT_COMPILER_H
