#ifndef ANNC_KERNEL_CONTEXT_H
#define ANNC_KERNEL_CONTEXT_H

#include <stddef.h>
#include <stdint.h>

#include "Context/StringRef.h"

#ifdef __cplusplus
extern "C" {
#endif

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

enum { ANNC_CONTEXT_ABI_VERSION = 3, ANNC_CONTEXT_MAX_RANK = 8 };
#define ANNC_CONTEXT_ABI_NAME "annc_context_v3"
#define ANNC_ELEMENT_TYPE_STRING ((AnncElementType)4u)
#define ANNC_TENSOR_BOUND ((uint32_t)1u)
#define ANNC_TENSOR_READ_ONLY ((uint32_t)2u)
#define ANNC_LAYOUT_STRIDED ((uint32_t)0u)
#define ANNC_ATTRIBUTE_I64 ((uint32_t)1u)

// Borrowed metadata and data. Offset and strides are measured in elements.
// Rank-zero views have no sizes/strides; empty bound views may have null data.
// LLVM mirror: annc/lib/Conversion/Common/ContextInputOverrides.cpp (viewType).
// Changes to field types/order or packing/alignment require updating the
// mirror and its field indices.
typedef struct {
  void* allocated;
  void* aligned;
  int64_t offset;
  const int64_t* sizes;
  const int64_t* strides;
  AnncElementType element_type;
  uint32_t rank;
  uint32_t layout;
  uint32_t flags;
} AnncTensorView;

typedef struct {
  uint32_t kind;
  uint32_t reserved;
  const void* data;
  uint64_t count;
} AnncAttributeView;

typedef AnncStatusCode (*AnncContextAllocateOutput)(
    void* opaque, uint32_t slot, const AnncTensorDesc* request,
    AnncTensorView* result);
typedef AnncStatusCode (*AnncContextAllocateTemp)(void* opaque,
                                                  const AnncTensorDesc* request,
                                                  AnncTensorView* result);
typedef void (*AnncContextReportError)(void* opaque, AnncStatusCode status,
                                       const char* message);
typedef AnncStatusCode (*AnncContextGetStatus)(void* opaque);
typedef void (*AnncContextTask)(void* task, int64_t begin, int64_t end);
// Synchronous: all task invocations finish before this callback returns.
typedef AnncStatusCode (*AnncContextParallelFor)(void* opaque, int64_t total,
                                                 int64_t grain_size,
                                                 AnncContextTask callback,
                                                 void* task);
typedef int32_t (*AnncContextNumThreads)(void* opaque);
// Optional synchronous sharding service. max_parallelism >= 1, total >= 0,
// cost_per_unit >= 0; all task invocations finish before returning.
typedef AnncStatusCode (*AnncContextShard)(void* opaque,
                                           int32_t max_parallelism,
                                           int64_t total, int64_t cost_per_unit,
                                           AnncContextTask callback,
                                           void* task);

// Optional host-native interoperability. The provider tag identifies the
// library ABI; an unknown tag or unavailable/version-incompatible pool yields
// null. The borrowed handle is valid only during this synchronous kernel call.
typedef void* (*AnncContextNativeThreadPool)(void* opaque, const char* provider);

typedef struct {
  uint32_t struct_size;
  uint32_t abi_version;
  void* opaque;
  AnncContextAllocateOutput allocate_output;
  AnncContextAllocateTemp allocate_temp;
  AnncContextReportError report_error;
  AnncContextGetStatus get_status;
  AnncContextParallelFor parallel_for;
  AnncContextNumThreads num_threads;
  AnncContextShard shard;
  AnncContextNativeThreadPool native_thread_pool;
} AnncRuntimeServices;

// Allocation, error, and scheduling callbacks form the required prefix.
// Optional services may be read only when struct_size covers the complete
// field.
#define ANNC_RUNTIME_SERVICES_BASE_SIZE                   \
  ((uint32_t)(offsetof(AnncRuntimeServices, num_threads) + \
              sizeof(((AnncRuntimeServices*)0)->num_threads)))

#define ANNC_RUNTIME_SERVICES_SHARD_SIZE             \
  ((uint32_t)(offsetof(AnncRuntimeServices, shard) + \
              sizeof(((AnncRuntimeServices*)0)->shard)))

#define ANNC_RUNTIME_SERVICES_NATIVE_POOL_SIZE \
  ((uint32_t)(offsetof(AnncRuntimeServices, native_thread_pool) + \
              sizeof(((AnncRuntimeServices*)0)->native_thread_pool)))

// LLVM mirror: annc/lib/Conversion/Common/ContextInputOverrides.cpp (contextType).
// Changes to field types/order or packing/alignment require updating the
// mirror and its field indices.
typedef struct {
  uint32_t struct_size;
  uint32_t abi_version;
  const AnncTensorView* inputs;
  uint32_t num_inputs;
  AnncTensorView* outputs;
  uint32_t num_outputs;
  const AnncAttributeView* attrs;
  uint32_t num_attrs;
  const AnncRuntimeServices* services;
} AnncKernelContext;

#define ANNC_CONTEXT_BASE_SIZE                        \
  ((uint32_t)(offsetof(AnncKernelContext, services) + \
              sizeof(((AnncKernelContext*)0)->services)))
typedef AnncStatusCode (*AnncKernelFn)(AnncKernelContext* context);

#ifdef __cplusplus
}
#include <type_traits>
static_assert(std::is_standard_layout<AnncTensorDesc>::value,
              "AnncTensorDesc must have standard layout");
static_assert(std::is_standard_layout<AnncTensorView>::value,
              "tensor view must have C layout");
static_assert(std::is_standard_layout<AnncRuntimeServices>::value,
              "runtime services must have C layout");
static_assert(std::is_standard_layout<AnncKernelContext>::value,
              "kernel context must have C layout");
static_assert(ANNC_CONTEXT_BASE_SIZE == sizeof(AnncKernelContext),
              "context prefix must remain compatible");
#endif

#endif
