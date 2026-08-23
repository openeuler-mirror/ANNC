#ifndef ANNC_EXECUTION_CONTEXT_H
#define ANNC_EXECUTION_CONTEXT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { ANNC_EXECUTION_ABI_VERSION = 2 };

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
} AnncExecutionContext;

#ifdef __cplusplus
}

#include <type_traits>
static_assert(std::is_standard_layout<AnncTensorDesc>::value,
              "AnncTensorDesc must have standard layout");
static_assert(std::is_standard_layout<AnncExecutionContext>::value,
              "AnncExecutionContext must have standard layout");
#endif

#endif
