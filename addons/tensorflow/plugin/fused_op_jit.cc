#include "fused_op_jit.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

Status CompileAnncJitKernel(
    const AnncJitCompileRequest& request,
    std::shared_ptr<annc::jit::JitExecutable>* executable) {
  if (!executable) {
    return errors::InvalidArgument("Missing ANNC JIT executable output");
  }
  executable->reset();
  // Pass borrowed arrays and strings through C, not this side's C++ objects.
  // anncCompileJitKernel is synchronous, so request/dims/ranks stay alive until
  // support has finished reading them and issuing log callbacks.
  std::vector<const int64_t*> dims;
  std::vector<size_t> ranks;
  dims.reserve(request.argument_shapes.size());
  ranks.reserve(request.argument_shapes.size());
  for (const auto& shape : request.argument_shapes) {
    dims.push_back(shape.data());
    ranks.push_back(shape.size());
  }
  ::AnncJitCompileRequest input{};
  input.atir_module_path = request.atir_module_path.c_str();
  input.kernel_name = request.kernel_name.c_str();
  input.intra_thread_count = request.intra_thread_count;
  input.argument_dims = dims.data();
  input.argument_ranks = ranks.data();
  input.argument_count = dims.size();
  input.log = [](void*, int warning, const char* message) {
    // Logging must not unwind through the C interface.
    try {
      if (warning) {
        LOG(WARNING) << message;
      } else {
        LOG(INFO) << message;
      }
    } catch (...) {
    }
  };
  AnncJitKernelResult result{};
  anncCompileJitKernel(&input, &result);
  // Own the handle before checking status: failures also return owned state.
  // TF errors copy result.error before this guard releases the support object.
  annc::tensorflow::JitKernelPtr kernel(result.kernel);
  switch (result.status) {
    case ANNC_JIT_INVALID_ARGUMENT:
      return errors::InvalidArgument(result.error);
    case ANNC_JIT_NOT_FOUND:
      return errors::NotFound(result.error);
    case ANNC_JIT_INTERNAL:
      return errors::Internal(result.error);
    case ANNC_JIT_OK:
      break;
  }
  // Transfer ownership to the cached executable. Its final release calls back
  // into support to unload kernel.so and clean up the compilation directory.
  *executable =
      std::make_shared<annc::jit::JitExecutable>(std::move(kernel), result);
  return OkStatus();
}

}  // namespace tensorflow
