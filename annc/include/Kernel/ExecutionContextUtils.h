#ifndef ANNC_EXECUTION_CONTEXT_UTILS_H
#define ANNC_EXECUTION_CONTEXT_UTILS_H

#include <limits>
#include <string>
#include <type_traits>

#include "Kernel/ExecutionContext.h"
#include "Kernel/MemRefTypes.h"
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

inline llvm::Error validateContext(const AnncExecutionContext* context) {
  if (context == nullptr) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "execution context is null");
  }
  if (context->struct_size != sizeof(AnncExecutionContext)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid execution context size");
  }
  if (context->abi_version != ANNC_EXECUTION_ABI_VERSION ||
      context->handle == nullptr ||
      context->allocate_output == nullptr || context->allocate_temp == nullptr ||
      context->report_error == nullptr) {
    return error(context, ANNC_STATUS_INVALID_ARGUMENT,
                 "invalid execution context");
  }
  return llvm::Error::success();
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
