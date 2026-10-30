#include "c_api.h"

#include <exception>
#include <new>

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
         (!request->argument_dims || !request->argument_ranks))) {
      result->status = ANNC_JIT_INVALID_ARGUMENT;
      kernel.error = "Missing ANNC JIT request data";
    } else {
      // Copy borrowed C data into support's own strings/vectors before calling
      // the compiler implementation; no C++ container crosses the ABI boundary.
      annc::jit::JitCompileRequest input;
      input.atir_module_path = request->atir_module_path;
      input.kernel_name = request->kernel_name;
      input.intra_thread_count = request->intra_thread_count;
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
