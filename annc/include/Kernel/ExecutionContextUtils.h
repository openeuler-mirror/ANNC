#ifndef ANNC_EXECUTION_CONTEXT_UTILS_H
#define ANNC_EXECUTION_CONTEXT_UTILS_H

#include <limits>
#include <string>
#include <type_traits>

#include "Kernel/ExecutionContext.h"
#include "Kernel/MemRefTypes.h"
#include "Context/ThreadPool.h"
#include "llvm/Support/Error.h"

namespace annc::kernels::execution {

template <typename T>
using Expected = llvm::Expected<T>;

inline void reportError(const AnncExecutionContext* context,
                        AnncStatusCode status,
                        const char* message) {
  if (context != nullptr && context->report_error != nullptr) {
    context->report_error(context->handle, status, message);
  }
}

inline llvm::Error error(const AnncExecutionContext* context,
                         AnncStatusCode status,
                         const char* message) {
  reportError(context, status, message);
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s", message);
}

// v4 appends the optional packed-B field to v3; earlier fields retain offsets.
inline bool isV3OrV4Context(const AnncExecutionContext* context) {
  return context != nullptr &&
         ((context->struct_size == ANNC_EXECUTION_V3_CONTEXT_SIZE &&
           context->abi_version == ANNC_EXECUTION_V3_ABI_VERSION) ||
          (context->struct_size == ANNC_EXECUTION_V4_CONTEXT_SIZE &&
           context->abi_version == ANNC_EXECUTION_ABI_VERSION));
}

inline bool isV2Context(const AnncExecutionContext* context) {
  return context != nullptr &&
         context->struct_size == ANNC_EXECUTION_V2_CONTEXT_SIZE &&
         context->abi_version == ANNC_EXECUTION_V2_ABI_VERSION;
}

// Both context layouts are accepted: every field except `thread_pool` sits at
// the same offset, so the callbacks are usable from either. Reading
// `thread_pool` off a v2 context would overrun the host's object -- use
// threadPool() below, never the field directly.
inline llvm::Error validateContext(const AnncExecutionContext* context) {
  if (context == nullptr) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "execution context is null");
  }
  if (!isV2Context(context) && !isV3OrV4Context(context)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid execution context size or version");
  }
  if (context->handle == nullptr ||
      context->allocate_output == nullptr ||
      context->allocate_temp == nullptr ||
      context->report_error == nullptr) {
    return error(context, ANNC_STATUS_INVALID_ARGUMENT,
                 "invalid execution context");
  }
  return llvm::Error::success();
}

// Returns the thread pool carried by the context (ABI v3), or nullptr when the
// host supplied the v2 layout or left the field empty. The host guarantees the
// pool outlives the call that received the context.
inline annc::threadpool::AnncThreadPool* threadPool(
    const AnncExecutionContext* context) {
  constexpr uint32_t kThreadPoolFieldEnd =
      ANNC_EXECUTION_V2_CONTEXT_SIZE + sizeof(AnncThreadPoolHandle*);
  if (context == nullptr || context->struct_size < kThreadPoolFieldEnd ||
      (context->abi_version != ANNC_EXECUTION_V3_ABI_VERSION &&
       context->abi_version != ANNC_EXECUTION_ABI_VERSION)) {
    return nullptr;
  }
  return reinterpret_cast<annc::threadpool::AnncThreadPool*>(
      context->thread_pool);
}

// Pool lookup for kernels: the context pool wins; when the context carries
// none, fall back to the pool the host injected into this library's TLS (the
// TensorFlow path). Returns nullptr when neither channel is populated, and
// parallel_for then runs the callback serially.
inline annc::threadpool::AnncThreadPool* resolveThreadPool(
    const AnncExecutionContext* context) {
  if (auto* pool = threadPool(context)) return pool;
  return annc::threadpool::getCurrentThreadPool();
}

inline llvm::Expected<size_t> elementSize(AnncElementType type) {
  switch (type) {
    case ANNC_ELEMENT_TYPE_F32:
    case ANNC_ELEMENT_TYPE_I32:
      return 4;
    case ANNC_ELEMENT_TYPE_I64:
      return 8;
    default:
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "unsupported element type");
  }
}

inline llvm::Expected<size_t> checkedByteCount(AnncElementType type,
                                               uint32_t rank,
                                               const int64_t* dims) {
  if (dims == nullptr) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid tensor dimensions");
  }
  size_t elements = 1;
  for (uint32_t i = 0; i < rank; ++i) {
    if (dims[i] < 0 ||
        (dims[i] != 0 && elements > std::numeric_limits<size_t>::max() /
                                  static_cast<size_t>(dims[i]))) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "invalid tensor dimensions");
    }
    elements *= static_cast<size_t>(dims[i]);
  }
  auto size = elementSize(type);
  if (!size) return size.takeError();
  if (elements > std::numeric_limits<size_t>::max() / *size) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "tensor byte size overflow");
  }
  return elements * *size;
}

// Trait: whether the memref layout carries sizes/strides arrays.  0-D memrefs
// (AnncMemRef0DI32/0DF32/0DI64) match the MLIR C interface for memref<...> and
// only have {allocated, aligned, offset}, so the loop bodies are skipped via
// if constexpr.
template <typename MemRef, typename = void>
struct HasSizes : std::false_type {};
template <typename MemRef>
struct HasSizes<MemRef, std::void_t<decltype(std::declval<MemRef>().sizes)>>
    : std::true_type {};

template <typename MemRef>
inline MemRef makeMemRef(void* data, uint32_t rank, const int64_t* dims) {
  MemRef result{};
  result.allocated = static_cast<decltype(result.allocated)>(data);
  result.aligned = static_cast<decltype(result.aligned)>(data);
  result.offset = 0;
  if constexpr (HasSizes<MemRef>::value) {
    for (uint32_t i = 0; i < rank; ++i) result.sizes[i] = dims[i];
    if (rank > 0) {
      result.strides[rank - 1] = 1;
      for (uint32_t i = rank - 1; i > 0; --i)
        result.strides[i - 1] = result.strides[i] * result.sizes[i];
    }
  }
  return result;
}

template <typename MemRef>
inline Expected<MemRef> allocate(const AnncExecutionContext* context,
                                 uint32_t slot, AnncElementType type,
                                 uint32_t rank,
                                 const int64_t* dims,
                                 AnncAllocationFlags flags) {
  if (auto err = validateContext(context)) return std::move(err);
  auto byteCount = checkedByteCount(type, rank, dims);
  if (!byteCount) {
    llvm::consumeError(byteCount.takeError());
    return error(context, ANNC_STATUS_INVALID_ARGUMENT, "invalid tensor dimensions");
  }
  AnncTensorDesc desc{sizeof(AnncTensorDesc), type, rank, flags, dims, nullptr};
  AnncStatusCode status = context->allocate_output(context->handle, slot, &desc);
  if (status != ANNC_STATUS_OK) {
    reportError(context, status, "output allocation failed");
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "output allocation failed");
  }
  if (*byteCount != 0 && desc.data == nullptr) {
    return error(context, ANNC_STATUS_INTERNAL,
                 "output allocation returned null data");
  }
  return makeMemRef<MemRef>(desc.data, rank, dims);
}

template <typename MemRef>
inline Expected<MemRef> allocateTemp(const AnncExecutionContext* context,
                                     AnncElementType type, uint32_t rank,
                                     const int64_t* dims,
                                     AnncAllocationFlags flags) {
  if (auto err = validateContext(context)) return std::move(err);
  auto byteCount = checkedByteCount(type, rank, dims);
  if (!byteCount) {
    llvm::consumeError(byteCount.takeError());
    return error(context, ANNC_STATUS_INVALID_ARGUMENT, "invalid tensor dimensions");
  }
  AnncTensorDesc desc{sizeof(AnncTensorDesc), type, rank, flags, dims, nullptr};
  AnncStatusCode status = context->allocate_temp(context->handle, &desc);
  if (status != ANNC_STATUS_OK) {
    reportError(context, status, "temporary allocation failed");
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "temporary allocation failed");
  }
  if (*byteCount != 0 && desc.data == nullptr) {
    return error(context, ANNC_STATUS_INTERNAL,
                 "temporary allocation returned null data");
  }
  return makeMemRef<MemRef>(desc.data, rank, dims);
}

inline Expected<AnncMemRef1DF32> allocateOutput1DF32(
    const AnncExecutionContext* context, uint32_t slot, int64_t dim0,
    AnncAllocationFlags flags) {
  return allocate<AnncMemRef1DF32>(context, slot, ANNC_ELEMENT_TYPE_F32, 1, &dim0,
                                   flags);
}
inline Expected<AnncMemRef2DF32> allocateOutput2DF32(
    const AnncExecutionContext* context, uint32_t slot, int64_t dim0,
    int64_t dim1, AnncAllocationFlags flags) {
  const int64_t dims[] = {dim0, dim1};
  return allocate<AnncMemRef2DF32>(context, slot, ANNC_ELEMENT_TYPE_F32, 2, dims,
                                   flags);
}
inline Expected<AnncMemRef3DF32> allocateOutput3DF32(
    const AnncExecutionContext* context, uint32_t slot, int64_t dim0,
    int64_t dim1, int64_t dim2, AnncAllocationFlags flags) {
  const int64_t dims[] = {dim0, dim1, dim2};
  return allocate<AnncMemRef3DF32>(context, slot, ANNC_ELEMENT_TYPE_F32, 3, dims,
                                   flags);
}
inline Expected<AnncMemRef4DF32> allocateOutput4DF32(
    const AnncExecutionContext* context, uint32_t slot, int64_t dim0,
    int64_t dim1, int64_t dim2, int64_t dim3, AnncAllocationFlags flags) {
  const int64_t dims[] = {dim0, dim1, dim2, dim3};
  return allocate<AnncMemRef4DF32>(context, slot, ANNC_ELEMENT_TYPE_F32, 4, dims,
                                   flags);
}
inline Expected<AnncMemRef2DI32> allocateOutput2DI32(
    const AnncExecutionContext* context, uint32_t slot, int64_t dim0,
    int64_t dim1, AnncAllocationFlags flags) {
  const int64_t dims[] = {dim0, dim1};
  return allocate<AnncMemRef2DI32>(context, slot, ANNC_ELEMENT_TYPE_I32, 2,
                                   dims, flags);
}
inline Expected<AnncMemRef1DI32> allocateOutput1DI32(
    const AnncExecutionContext* context, uint32_t slot, int64_t dim0,
    AnncAllocationFlags flags) {
  return allocate<AnncMemRef1DI32>(context, slot, ANNC_ELEMENT_TYPE_I32, 1, &dim0,
                                   flags);
}
inline Expected<AnncMemRef1DI64> allocateOutput1DI64(
    const AnncExecutionContext* context, uint32_t slot, int64_t dim0,
    AnncAllocationFlags flags) {
  return allocate<AnncMemRef1DI64>(context, slot, ANNC_ELEMENT_TYPE_I64, 1, &dim0,
                                   flags);
}
inline Expected<AnncMemRef2DI64> allocateOutput2DI64(
    const AnncExecutionContext* context, uint32_t slot, int64_t dim0,
    int64_t dim1, AnncAllocationFlags flags) {
  const int64_t dims[] = {dim0, dim1};
  return allocate<AnncMemRef2DI64>(context, slot, ANNC_ELEMENT_TYPE_I64, 2, dims,
                                   flags);
}
inline Expected<AnncMemRef0DI32> allocateOutput0DI32(
    const AnncExecutionContext* context, uint32_t slot,
    AnncAllocationFlags flags) {
  // rank-0: dims is never read (checkedByteCount loops zero times), but the
  // pointer must stay non-null.
  const int64_t dims[] = {0};
  return allocate<AnncMemRef0DI32>(context, slot, ANNC_ELEMENT_TYPE_I32, 0,
                                   dims, flags);
}
inline Expected<AnncMemRef2DF32> allocateTemp2DF32(
    const AnncExecutionContext* context, int64_t dim0, int64_t dim1,
    AnncAllocationFlags flags) {
  const int64_t dims[] = {dim0, dim1};
  return allocateTemp<AnncMemRef2DF32>(context, ANNC_ELEMENT_TYPE_F32, 2, dims,
                                       flags);
}
inline Expected<AnncMemRef1DI64> allocateTemp1DI64(
    const AnncExecutionContext* context, int64_t dim0,
    AnncAllocationFlags flags) {
  return allocateTemp<AnncMemRef1DI64>(context, ANNC_ELEMENT_TYPE_I64, 1, &dim0,
                                       flags);
}
inline Expected<AnncMemRef2DI64> allocateTemp2DI64(
    const AnncExecutionContext* context, int64_t dim0, int64_t dim1,
    AnncAllocationFlags flags) {
  const int64_t dims[] = {dim0, dim1};
  return allocateTemp<AnncMemRef2DI64>(context, ANNC_ELEMENT_TYPE_I64, 2, dims,
                                       flags);
}

}  // namespace annc::kernels::execution

#endif
