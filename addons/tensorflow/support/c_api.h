#ifndef ANNC_TENSORFLOW_SUPPORT_C_API_H
#define ANNC_TENSORFLOW_SUPPORT_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AnncJitKernel AnncJitKernel;
typedef struct AnncThreadPoolAdapter AnncThreadPoolAdapter;

typedef enum {
  ANNC_JIT_OK,
  ANNC_JIT_INVALID_ARGUMENT,
  ANNC_JIT_NOT_FOUND,
  ANNC_JIT_INTERNAL
} AnncJitStatus;

// Callbacks are synchronous, borrow their arguments, and must not throw.
typedef void (*AnncJitLogCallback)(void* context, int warning,
                                   const char* message);

typedef struct {
  const char* atir_module_path;
  const char* kernel_name;
  int64_t intra_thread_count;
  const int64_t* const* argument_dims;
  const size_t* argument_ranks;
  size_t argument_count;
  AnncJitLogCallback log;
  void* log_context;
} AnncJitCompileRequest;

// The caller owns kernel even on failure. All strings and function addresses
// are borrowed from it and remain valid until anncReleaseJitKernel(kernel).
typedef struct {
  AnncJitKernel* kernel;
  AnncJitStatus status;
  const char* error;
  const char* work_dir;
  const char* shared_library_path;
  void* kernel_function;
  void* set_thread_pool_function;
  void* get_thread_pool_function;
} AnncJitKernelResult;

void anncCompileJitKernel(const AnncJitCompileRequest* request,
                          AnncJitKernelResult* result);
void anncReleaseJitKernel(AnncJitKernel* kernel);

typedef void (*AnncThreadPoolTask)(void* task_context, int64_t begin,
                                   int64_t end);
typedef struct {
  void* context;
  int (*num_threads)(void*);
  int (*in_parallel_region)(void*);
  // All tasks finish before returning. Return zero on success. block_size == 0
  // selects cost-based scheduling; a positive value selects fixed blocks.
  int (*parallel_for)(void*, int64_t total, int64_t block_size,
                      int64_t cost_per_unit, AnncThreadPoolTask task,
                      void* task_context);
} AnncThreadPoolCallbacks;

// Copies the callbacks, but borrows context. Null means invalid callbacks or
// allocation failure. Restore the kernel's previous pool before destroying this
// adapter.
AnncThreadPoolAdapter* anncCreateThreadPoolAdapter(
    const AnncThreadPoolCallbacks* callbacks);
void anncDestroyThreadPoolAdapter(AnncThreadPoolAdapter* adapter);

#ifdef __cplusplus
}
#endif

#endif  // ANNC_TENSORFLOW_SUPPORT_C_API_H
