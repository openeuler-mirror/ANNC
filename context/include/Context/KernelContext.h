#ifndef ANNC_KERNEL_CONTEXT_H
#define ANNC_KERNEL_CONTEXT_H

#include <stddef.h>
#include <stdint.h>

#include "Context/StringRef.h"

#ifdef __cplusplus
namespace annc {
namespace threadpool {
class AnncThreadPool;
}
}
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

// Process-local, read-only kernel feature resource. The type tag and version
// identify the payload ABI; bytes is its descriptor size, not its allocation
// size. The instance owner keeps the descriptor, tag and referenced storage
// alive through all synchronous calls, and releases them after calls finish.
// Never serialize this view or its pointers into IR / sites.json. Resources
// have their own array in AnncKernelContext, separate from scalar attributes.
typedef struct {
  uint32_t struct_size;
  uint32_t version;
  const char* type;
  const void* data;
  uint64_t bytes;
} AnncResourceView;

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

// request and its dims are borrowed only until this synchronous callback
// returns. On success, result borrows host-owned storage and metadata valid
// throughout the kernel call. Before returning ANNC_STATUS_OK, the kernel must
// bind this returned view to context->outputs[slot]; passing that slot as
// result satisfies the requirement directly. Preserve its data/metadata
// pointers, offset, type, rank and layout, with BOUND set and READ_ONLY clear.
// The kernel may write tensor elements, but must not modify the host-owned
// sizes/strides. Already-bound outputs may instead be reused without invoking
// this callback.
typedef AnncStatusCode (*AnncContextAllocateOutput)(
    void* opaque, uint32_t slot, const AnncTensorDesc* request,
    AnncTensorView* result);
typedef AnncStatusCode (*AnncContextAllocateTemp)(void* opaque,
                                                  const AnncTensorDesc* request,
                                                  AnncTensorView* result);
typedef void (*AnncContextReportError)(void* opaque, AnncStatusCode status,
                                       const char* message);
typedef AnncStatusCode (*AnncContextGetStatus)(void* opaque);
typedef struct {
  uint32_t struct_size;
  uint32_t abi_version;
  void* opaque;
  AnncContextAllocateOutput allocate_output;
  AnncContextAllocateTemp allocate_temp;
  AnncContextReportError report_error;
  AnncContextGetStatus get_status;
  // Borrowed for the synchronous call; null selects serial execution. This
  // points to an AnncThreadPool adapter, never to a framework-native pool.
  // Host and kernel must share the C++ interface/toolchain ABI.
#ifdef __cplusplus
  annc::threadpool::AnncThreadPool* thread_pool;
#else
  void* thread_pool;
#endif
} AnncRuntimeServices;

#define ANNC_RUNTIME_SERVICES_BASE_SIZE                    \
  ((uint32_t)(offsetof(AnncRuntimeServices, thread_pool) + \
              sizeof(((AnncRuntimeServices*)0)->thread_pool)))

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
  // Borrowed instance resources, independent of positional attributes.
  // A nonzero count requires a nonnull array.
  const AnncResourceView* resources;
  uint32_t num_resources;
  const AnncRuntimeServices* services;
} AnncKernelContext;

#define ANNC_CONTEXT_BASE_SIZE                        \
  ((uint32_t)(offsetof(AnncKernelContext, services) + \
              sizeof(((AnncKernelContext*)0)->services)))
typedef AnncStatusCode (*AnncKernelFn)(AnncKernelContext* context);

// Export a standard ranked memref descriptor (native 64-bit pointers/index).
// Descriptors borrow tensor data and live in host allocate_temp storage until
// this synchronous execution ends. Each call allocates a new descriptor;
// sizes/strides are copied inline, tensor elements are never copied.
// Failure returns null and records a context error when the context is valid.
const void* annc_context_get_input_memref(
    const AnncKernelContext* context, uint32_t input_index);

// The host must initialize outputs[output_index].element_type before calling
// the kernel, even for an unbound output. Reuses a matching writable bound
// output or allocates/binds one through allocate_output. The caller may write
// tensor elements through the descriptor, but must not modify its metadata.
void* annc_context_alloc_output_memref(
    AnncKernelContext* context, uint32_t output_index, uint32_t rank,
    const int64_t* dims);

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
