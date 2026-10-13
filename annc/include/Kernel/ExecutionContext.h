#ifndef ANNC_EXECUTION_CONTEXT_H
#define ANNC_EXECUTION_CONTEXT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Version history:
//   v2: handle + allocate_output/allocate_temp/report_error, no thread pool.
//   v3: appends `thread_pool` so the host can carry the pool in the context
//       itself (the ONNX Runtime EP does). A host that keeps its own channel
//       -- the TensorFlow addon injects the host pool into the kernel
//       library's TLS -- leaves the field NULL; kernels then fall back to the
//       TLS pool through resolveThreadPool, which keeps the TensorFlow
//       semantics unchanged. Kernels must accept both layouts: read the field
//       only after checking struct_size/abi_version.
enum { ANNC_EXECUTION_ABI_VERSION = 3 };
enum { ANNC_EXECUTION_V2_ABI_VERSION = 2 };

typedef uint32_t AnncStatusCode;
#define ANNC_STATUS_OK ((AnncStatusCode)0u)
#define ANNC_STATUS_INVALID_ARGUMENT ((AnncStatusCode)1u)
#define ANNC_STATUS_ALLOCATION_FAILED ((AnncStatusCode)2u)
#define ANNC_STATUS_INTERNAL ((AnncStatusCode)3u)

typedef uint32_t AnncElementType;
#define ANNC_ELEMENT_TYPE_F32 ((AnncElementType)1u)
#define ANNC_ELEMENT_TYPE_I32 ((AnncElementType)2u)
#define ANNC_ELEMENT_TYPE_I64 ((AnncElementType)3u)

typedef uint32_t AnncAllocationFlags;
#define ANNC_ALLOCATION_NONE ((AnncAllocationFlags)0u)
#define ANNC_ALLOCATION_ZERO ((AnncAllocationFlags)1u)

typedef struct {
  uint32_t struct_size;
  AnncElementType element_type;
  uint32_t rank;
  AnncAllocationFlags flags;
  const int64_t* dims;
  void* data;
} AnncTensorDesc;

typedef struct AnncExecutionHandle AnncExecutionHandle;

// Opaque handle for the host thread pool carried by the context (ABI v3).
// The kernel side casts it to annc::threadpool::AnncThreadPool
// (Support/ThreadPool/ThreadPool.h). The host guarantees the object stays
// alive for the duration of every call that receives the context.
typedef struct AnncThreadPoolHandle AnncThreadPoolHandle;

typedef AnncStatusCode (*AnncAllocateOutputCallback)(
    AnncExecutionHandle* handle, uint32_t slot, AnncTensorDesc* desc);
typedef AnncStatusCode (*AnncAllocateTempCallback)(
    AnncExecutionHandle* handle, AnncTensorDesc* desc);
typedef void (*AnncReportErrorCallback)(AnncExecutionHandle* handle,
                                        AnncStatusCode status,
                                        const char* message);

typedef struct {
  uint32_t struct_size;
  uint32_t abi_version;
  AnncExecutionHandle* handle;
  AnncAllocateOutputCallback allocate_output;
  AnncAllocateTempCallback allocate_temp;
  AnncReportErrorCallback report_error;
  // Optional host thread pool (ABI v3). NULL means "not provided": kernels
  // then use the pool injected into their library's TLS, if any. The handle
  // points at a C++ interface (annc::threadpool::AnncThreadPool) whose vtable
  // crosses the shared-library boundary -- host and kernel must be built from
  // the same tree, or the virtual layout may disagree (the size/version check
  // above does not cover it).
  AnncThreadPoolHandle* thread_pool;
} AnncExecutionContext;

// Size of the v2 layout (everything before `thread_pool`). A host compiled
// against the v2 header reports this as its struct_size.
#define ANNC_EXECUTION_V2_CONTEXT_SIZE \
  (offsetof(AnncExecutionContext, thread_pool))

#ifdef __cplusplus
}

#include <type_traits>
static_assert(std::is_standard_layout<AnncTensorDesc>::value,
              "AnncTensorDesc must have standard layout");
static_assert(std::is_standard_layout<AnncExecutionContext>::value,
              "AnncExecutionContext must have standard layout");
#endif

#endif
