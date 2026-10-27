#ifndef ANNC_KERNEL_CONTEXT_UTILS_H
#define ANNC_KERNEL_CONTEXT_UTILS_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

#include "Context/KernelContext.h"

namespace annc::kernels {

inline bool validContext(const AnncKernelContext* context) {
  if (!context || context->struct_size < ANNC_CONTEXT_BASE_SIZE ||
      context->abi_version != ANNC_CONTEXT_ABI_VERSION)
    return false;
  const auto* services = context->services;
  return services && services->struct_size >= ANNC_RUNTIME_SERVICES_BASE_SIZE &&
         services->abi_version == ANNC_CONTEXT_ABI_VERSION &&
         services->allocate_output && services->allocate_temp &&
         services->report_error && services->get_status &&
         services->parallel_for && services->num_threads &&
         (!context->num_inputs || context->inputs) &&
         (!context->num_outputs || context->outputs) &&
         (!context->num_attrs || context->attrs);
}

inline size_t contextElementSize(AnncElementType type) {
  switch (type) {
    case ANNC_ELEMENT_TYPE_F32:
    case ANNC_ELEMENT_TYPE_I32:
      return 4;
    case ANNC_ELEMENT_TYPE_I64:
      return 8;
    case ANNC_ELEMENT_TYPE_STRING:
      return sizeof(AnncStringRef);
    default:
      return 0;
  }
}

inline bool contextByteCount(AnncElementType type, uint32_t rank,
                             const int64_t* dims, size_t* bytes) {
  size_t size = contextElementSize(type);
  if (!size || rank > ANNC_CONTEXT_MAX_RANK || (rank && !dims)) return false;
  for (uint32_t i = 0; i < rank; ++i) {
    if (dims[i] < 0 || (dims[i] && size > std::numeric_limits<size_t>::max() /
                                              static_cast<size_t>(dims[i])))
      return false;
    size *= static_cast<size_t>(dims[i]);
  }
  *bytes = size;
  return true;
}

template <typename T>
constexpr AnncElementType contextElementType() {
  static_assert(std::is_same<T, float>::value ||
                    std::is_same<T, int32_t>::value ||
                    std::is_same<T, int64_t>::value,
                "unsupported context element type");
  return 0;
}
template <>
constexpr AnncElementType contextElementType<float>() {
  return ANNC_ELEMENT_TYPE_F32;
}
template <>
constexpr AnncElementType contextElementType<int32_t>() {
  return ANNC_ELEMENT_TYPE_I32;
}
template <>
constexpr AnncElementType contextElementType<int64_t>() {
  return ANNC_ELEMENT_TYPE_I64;
}
template <>
constexpr AnncElementType contextElementType<AnncStringRef>() {
  return ANNC_ELEMENT_TYPE_STRING;
}

// These kernels require dense views. Other layouts must be
// supported explicitly by their consumer rather than silently flattened.
inline bool validStridedView(const AnncTensorView& view) {
  size_t bytes;
  if (!(view.flags & ANNC_TENSOR_BOUND) || view.offset < 0 ||
      view.layout != ANNC_LAYOUT_STRIDED ||
      !contextByteCount(view.element_type, view.rank, view.sizes, &bytes) ||
      (view.rank && !view.strides) || (bytes && !view.aligned))
    return false;
  uint64_t last = static_cast<uint64_t>(view.offset);
  for (uint32_t i = 0; i < view.rank; ++i) {
    if (view.strides[i] < 0) return false;
    const uint64_t extent = view.sizes[i] > 0 ? view.sizes[i] - 1 : 0;
    const uint64_t stride = view.strides[i];
    if (extent &&
        stride > (std::numeric_limits<uint64_t>::max() - last) / extent)
      return false;
    last += extent * stride;
  }
  return last <= (std::numeric_limits<size_t>::max() -
                  contextElementSize(view.element_type)) /
                     contextElementSize(view.element_type);
}

inline bool validDenseView(const AnncTensorView& view) {
  if (!validStridedView(view)) return false;
  int64_t stride = 1;
  for (uint32_t i = view.rank; i > 0; --i) {
    if (view.strides[i - 1] != stride) return false;
    const int64_t dim = view.sizes[i - 1];
    if (dim && stride > std::numeric_limits<int64_t>::max() / dim) return false;
    stride *= dim;
  }
  return true;
}

template <typename T>
T* contextData(const AnncTensorView& view) {
  return view.aligned ? static_cast<T*>(view.aligned) + view.offset : nullptr;
}

class KernelThreadPool;

class KernelContext {
 public:
  explicit KernelContext(AnncKernelContext* context) : context_(context) {}

  uint32_t numInputs() const { return context_->num_inputs; }

  AnncStatusCode error(AnncStatusCode status, const char* message) const {
    context_->services->report_error(context_->services->opaque, status,
                                     message);
    return status;
  }

  // Indexed access is the hot path for builtin kernels. The slot order is
  // defined by the shared CustomOpSchema and does not require name metadata.
  const AnncTensorView* getStridedInput(uint32_t index) const {
    if (index >= context_->num_inputs)
      return failInput("input index out of range");
    const auto& view = context_->inputs[index];
    if (!validStridedView(view)) return failInput("invalid strided input view");
    return &view;
  }

  const AnncTensorView* getInput(uint32_t index) const {
    if (index >= context_->num_inputs)
      return failInput("input index out of range");
    const auto& view = context_->inputs[index];
    if (!validDenseView(view)) return failInput("input must be contiguous");
    return &view;
  }

  std::optional<int64_t> getI64Attr(uint32_t index) const {
    if (index >= context_->num_attrs) {
      error(ANNC_STATUS_INVALID_ARGUMENT, "missing i64 attribute");
      return std::nullopt;
    }
    const auto& attr = context_->attrs[index];
    if (attr.kind != ANNC_ATTRIBUTE_I64 || attr.reserved || attr.count != 1 ||
        !attr.data) {
      error(ANNC_STATUS_INVALID_ARGUMENT, "invalid i64 attribute");
      return std::nullopt;
    }
    int64_t value;
    std::memcpy(&value, attr.data, sizeof(value));
    if (status() != ANNC_STATUS_OK) return std::nullopt;
    return value;
  }

  KernelThreadPool getThreadPool() const;

  // Dimensions are borrowed only for the synchronous allocation callback.
  // The runtime owns the returned view's shape, strides and storage.
  template <typename T>
  AnncTensorView* allocateOutput(
      uint32_t slot, std::initializer_list<int64_t> dims,
      AnncAllocationFlags flags = ANNC_ALLOCATION_NONE) {
    if (status() != ANNC_STATUS_OK) return nullptr;
    if (dims.size() > ANNC_CONTEXT_MAX_RANK) {
      error(ANNC_STATUS_INVALID_ARGUMENT, "unsupported output rank");
      return nullptr;
    }
    AnncTensorView* result = nullptr;
    if (allocateOutputImpl(slot, contextElementType<T>(),
                           static_cast<uint32_t>(dims.size()), dims.begin(),
                           &result, flags) != ANNC_STATUS_OK)
      return nullptr;
    return result;
  }

  template <typename T, uint32_t Rank>
  AnncStatusCode allocateTemp(const int64_t* dims, AnncTensorView* result) {
    static_assert(Rank <= ANNC_CONTEXT_MAX_RANK, "unsupported context rank");
    return allocateTempImpl(contextElementType<T>(), Rank, dims, result);
  }

  AnncStatusCode status() const {
    return context_->services->get_status(context_->services->opaque);
  }

 private:
  AnncStatusCode allocateTempImpl(AnncElementType type, uint32_t rank,
                                  const int64_t* dims, AnncTensorView* result) {
    size_t bytes;
    if (!result || !contextByteCount(type, rank, dims, &bytes))
      return error(ANNC_STATUS_INVALID_ARGUMENT, "invalid temporary request");
    AnncTensorDesc request{sizeof(AnncTensorDesc), type, rank,
                           ANNC_ALLOCATION_NONE,   dims, nullptr};
    auto code = context_->services->allocate_temp(context_->services->opaque,
                                                  &request, result);
    if (code != ANNC_STATUS_OK)
      return error(code, "temporary allocation failed");
    if (result->element_type != type || result->rank != rank ||
        (result->flags & ANNC_TENSOR_READ_ONLY) || !validDenseView(*result))
      return error(ANNC_STATUS_INTERNAL, "invalid temporary allocation");
    for (uint32_t i = 0; i < rank; ++i)
      if (result->sizes[i] != dims[i])
        return error(ANNC_STATUS_INTERNAL, "temporary shape mismatch");
    return status();
  }

  AnncStatusCode allocateOutputImpl(
      uint32_t slot, AnncElementType type, uint32_t rank, const int64_t* dims,
      AnncTensorView** result, AnncAllocationFlags flags) {
    size_t bytes;
    if (!result || slot >= context_->num_outputs ||
        (flags & ~ANNC_ALLOCATION_ZERO) ||
        !contextByteCount(type, rank, dims, &bytes))
      return error(ANNC_STATUS_INVALID_ARGUMENT, "invalid output request");
    auto& view = context_->outputs[slot];
    if (!(view.flags & ANNC_TENSOR_BOUND)) {
      AnncTensorDesc request{
          sizeof(AnncTensorDesc), type, rank, flags, dims, nullptr};
      AnncStatusCode code = context_->services->allocate_output(
          context_->services->opaque, slot, &request, &view);
      if (code != ANNC_STATUS_OK)
        return error(code, "output allocation failed");
    }
    if (view.element_type != type || view.rank != rank ||
        (view.flags & ANNC_TENSOR_READ_ONLY) || !validDenseView(view))
      return error(ANNC_STATUS_INVALID_ARGUMENT, "invalid output binding");
    for (uint32_t i = 0; i < rank; ++i)
      if (view.sizes[i] != dims[i])
        return error(ANNC_STATUS_INVALID_ARGUMENT, "output shape mismatch");
    if (flags & ANNC_ALLOCATION_ZERO) {
      // Dense numeric tensors: value initialization also handles F32 zero.
      auto* data = static_cast<unsigned char*>(view.aligned);
      if (bytes) {
        data += static_cast<size_t>(view.offset) * contextElementSize(type);
        for (size_t i = 0; i < bytes; ++i) data[i] = 0;
      }
    }
    *result = &view;
    return status();
  }

  const AnncTensorView* failInput(const char* message) const {
    error(ANNC_STATUS_INVALID_ARGUMENT, message);
    return nullptr;
  }

  AnncKernelContext* context_;
};

// Borrowed handle to runtime scheduling services, not an owning worker pool.
// The context must remain alive until all synchronous calls return.
class KernelThreadPool {
 public:
  void* nativePool(const char* provider) const {
    if (services_->struct_size < ANNC_RUNTIME_SERVICES_NATIVE_POOL_SIZE ||
        !services_->native_thread_pool)
      return nullptr;
    return services_->native_thread_pool(services_->opaque, provider);
  }

  int32_t numThreads() const {
    const int32_t count = services_->num_threads(services_->opaque);
    return count > 0 ? count : 1;
  }

  template <typename Task>
  AnncStatusCode parallelFor(int64_t total, Task&& task) const {
    return run(total, 0, 0, std::forward<Task>(task));
  }

  template <typename Task>
  AnncStatusCode shard(int32_t maxParallelism, int64_t total,
                       int64_t costPerUnit, Task&& task) const {
    if (maxParallelism < 1 || costPerUnit < 0)
      return context_.error(ANNC_STATUS_INVALID_ARGUMENT,
                            "invalid shard scheduling options");
    return run(total, maxParallelism, costPerUnit, std::forward<Task>(task));
  }

 private:
  friend class KernelContext;
  KernelThreadPool(const KernelContext& context,
                   const AnncRuntimeServices* services)
      : context_(context), services_(services) {}

  template <typename Task>
  AnncStatusCode run(int64_t total, int32_t maxParallelism, int64_t costPerUnit,
                     Task&& task) const {
    if (total < 0)
      return context_.error(ANNC_STATUS_INVALID_ARGUMENT, "negative work size");
    if (context_.status() != ANNC_STATUS_OK) return context_.status();
    if (total == 0) return context_.status();
    auto callback = [](void* opaque, int64_t first, int64_t last) {
      (*static_cast<typename std::remove_reference<Task>::type*>(opaque))(first,
                                                                          last);
    };
    AnncStatusCode code;
    if (maxParallelism == 1) {
      task(0, total);
      return context_.status();
    }
    if (maxParallelism > 0 &&
        services_->struct_size >= ANNC_RUNTIME_SERVICES_SHARD_SIZE &&
        services_->shard) {
      code = services_->shard(
          services_->opaque, maxParallelism, total, costPerUnit, callback,
          const_cast<void*>(static_cast<const void*>(&task)));
    } else {
      if (maxParallelism > 0) {
        // Older runtimes cannot honor a parallelism limit; run serially.
        task(0, total);
        return context_.status();
      }
      code = services_->parallel_for(
          services_->opaque, total, 0, callback,
          const_cast<void*>(static_cast<const void*>(&task)));
    }
    if (code != ANNC_STATUS_OK)
      return context_.error(code, "parallel execution failed");
    return context_.status();
  }

  KernelContext context_;
  const AnncRuntimeServices* services_;
};

inline KernelThreadPool KernelContext::getThreadPool() const {
  return KernelThreadPool(*this, context_->services);
}

// Same scheduling inputs as TensorFlow Shard, with a framework-neutral handle.
template <typename Task>
AnncStatusCode Shard(int32_t maxParallelism, const KernelThreadPool& workers,
                     int64_t total, int64_t costPerUnit, Task&& task) {
  return workers.shard(maxParallelism, total, costPerUnit,
                       std::forward<Task>(task));
}

template <typename Implementation>
AnncStatusCode invokeContextKernel(AnncKernelContext* context,
                                   Implementation implementation) noexcept {
  if (!validContext(context)) return ANNC_STATUS_INVALID_ARGUMENT;
  KernelContext view(context);
  try {
    if (view.status() != ANNC_STATUS_OK) return view.status();
    // Callers provide inputs that satisfy the selected kernel contract.
    auto code = implementation(view);
    if (code != ANNC_STATUS_OK && view.status() == ANNC_STATUS_OK)
      return view.error(code, "kernel execution failed");
    return code == ANNC_STATUS_OK ? view.status() : code;
  } catch (const std::bad_alloc&) {
    return view.error(ANNC_STATUS_ALLOCATION_FAILED,
                      "kernel allocation failed");
  } catch (...) {
    return view.error(ANNC_STATUS_INTERNAL, "kernel raised an exception");
  }
}

}  // namespace annc::kernels

#endif
