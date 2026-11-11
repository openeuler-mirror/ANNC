#include "c_api.h"

#include <algorithm>
#include <exception>
#include <new>

#include "Support/GemmConfig.h"
#include "jit_compiler.h"

extern "C" void anncCompileJitKernel(const AnncJitCompileRequest* request,
                                     AnncJitKernelResult* result) {
  if (!result) return;
  *result = {};
  result->status = ANNC_JIT_INTERNAL;
  result->error = "Unable to allocate ANNC JIT result";
  try {
    // Allocate and release this object in support's ABI. The caller receives
    // only an opaque handle and must use anncReleaseJitKernel, even on failure.
    result->kernel = new AnncJitKernel;
    auto& kernel = *result->kernel;
    if (!request || !request->atir_module_path || !request->kernel_name ||
        (request->argument_count &&
         (!request->argument_dims || !request->argument_ranks)) ||
        (request->immutable_argument_count && !request->immutable_arguments)) {
      result->status = ANNC_JIT_INVALID_ARGUMENT;
      kernel.error = "Missing ANNC JIT request data";
    } else {
      // Copy borrowed C data into support's own strings/vectors before calling
      // the compiler implementation; no C++ container crosses the ABI boundary.
      annc::jit::JitCompileRequest input;
      input.atir_module_path = request->atir_module_path;
      input.kernel_name = request->kernel_name;
      input.intra_thread_count = request->intra_thread_count;
      input.enable_prepack = request->enable_prepack != 0;
      for (size_t i = 0; i < request->immutable_argument_count; ++i) {
        const int64_t index = request->immutable_arguments[i];
        if (index < 0 ||
            static_cast<size_t>(index) >= request->argument_count) {
          result->status = ANNC_JIT_INVALID_ARGUMENT;
          kernel.error = "Invalid immutable argument index";
          result->error = kernel.error.c_str();
          return;
        }
        input.immutable_arguments.push_back(index);
      }
      input.log = request->log;
      input.log_context = request->log_context;
      input.argument_shapes.resize(request->argument_count);
      for (size_t i = 0; i < request->argument_count; ++i) {
        if (request->argument_ranks[i]) {
          if (!request->argument_dims[i]) {
            result->status = ANNC_JIT_INVALID_ARGUMENT;
            kernel.error = "Missing ANNC JIT argument dimensions";
            result->error = kernel.error.c_str();
            return;
          }
          input.argument_shapes[i].assign(
              request->argument_dims[i],
              request->argument_dims[i] + request->argument_ranks[i]);
        }
      }
      result->status = annc::jit::CompileAnncJitKernel(input, &kernel);
    }
    // These strings and function addresses belong to kernel. They stay valid
    // until anncReleaseJitKernel; callers copy text or retain the handle.
    result->error = kernel.error.c_str();
    result->work_dir = kernel.work_dir.c_str();
    result->shared_library_path = kernel.shared_library_path.c_str();
    result->kernel_function = kernel.kernel_function;
    result->set_thread_pool_function = kernel.set_thread_pool_function;
    result->get_thread_pool_function = kernel.get_thread_pool_function;
    result->packed_rhs_specs = kernel.packed_rhs_specs;
    result->packed_rhs_count = kernel.packed_rhs_count;
    result->prepare_rhs = kernel.prepare_rhs;
  } catch (const std::bad_alloc&) {
    result->status = ANNC_JIT_INTERNAL;
    result->error = "Unable to allocate ANNC JIT result";
  } catch (const std::exception& error) {
    result->status = ANNC_JIT_INTERNAL;
    result->error = "ANNC JIT compilation threw an exception";
    // Keep the message alive after the exception object is destroyed.
    if (result->kernel) {
      try {
        result->kernel->error = error.what();
        result->error = result->kernel->error.c_str();
      } catch (...) {
      }
    }
  } catch (...) {
    result->status = ANNC_JIT_INTERNAL;
    result->error = "ANNC JIT compilation threw an unknown exception";
  }
}

extern "C" void anncReleaseJitKernel(AnncJitKernel* kernel) { delete kernel; }

extern "C" int anncTensorFlowPrepackEnabled(void) {
  return annc::gemmConfig().prepack;
}

extern "C" void anncFindImmutableInputs(
    const char* atir_path, const char* kernel_name, const char* fingerprint,
    int constants, int fixed, int dynamic, const size_t* input_bytes,
    uint8_t* eligible, AnncJitLogCallback log, void* log_context) {
  const int64_t count = static_cast<int64_t>(constants) + fixed + dynamic;
  if (constants < 0 || fixed < 0 || dynamic < 0 || count > INT32_MAX ||
      !eligible)
    return;
  std::fill(eligible, eligible + count, 0);
  if (!atir_path || !kernel_name || !fingerprint || !input_bytes) return;
  try {
    annc::jit::FindImmutableInputs(atir_path, kernel_name, fingerprint,
                                   constants, fixed, dynamic, input_bytes,
                                   eligible, log, log_context);
  } catch (...) {
    // Admission failure preserves ordinary execution; never unwind into TF.
    std::fill(eligible, eligible + count, 0);
  }
}
