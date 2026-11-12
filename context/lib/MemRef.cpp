#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#include "Context/KernelContextUtils.h"

namespace {

struct MemRefPrefix {
  void* allocated;
  void* aligned;
  int64_t offset;
};

static_assert(sizeof(void*) == 8 && sizeof(MemRefPrefix) == 24 &&
                  offsetof(MemRefPrefix, aligned) == 8 &&
                  offsetof(MemRefPrefix, offset) == 16,
              "Context memref access requires native 64-bit MLIR layout");

using annc::kernels::KernelContext;

void* fail(KernelContext& context, AnncStatusCode code,
           const char* message) noexcept {
  try {
    context.error(code, message);
  } catch (...) {
    // A broken error reporter still cannot unwind through the C boundary.
  }
  return nullptr;
}

void* exportMemRef(KernelContext& context, const AnncTensorView& view,
                   const AnncRuntimeServices& services) {
  // Host-owned temp storage avoids thread-local caches and retains previously
  // returned descriptors across subsequent input/output lookups.
  const int64_t words = 3 + 2 * view.rank;
  const AnncTensorDesc request{sizeof(AnncTensorDesc),
                               ANNC_ELEMENT_TYPE_I64,
                               1,
                               ANNC_ALLOCATION_NONE,
                               &words,
                               nullptr};
  AnncTensorView storage{};
  const auto code = services.allocate_temp(services.opaque, &request, &storage);
  if (code != ANNC_STATUS_OK)
    return fail(context, code, "memref descriptor allocation failed");
  if (context.status() != ANNC_STATUS_OK) return nullptr;
  if (storage.element_type != ANNC_ELEMENT_TYPE_I64 || storage.rank != 1 ||
      (storage.flags & ANNC_TENSOR_READ_ONLY) ||
      !annc::kernels::validDenseView(storage) || storage.sizes[0] != words)
    return fail(context, ANNC_STATUS_INTERNAL, "invalid memref storage");
  auto* descriptor =
      reinterpret_cast<char*>(annc::kernels::contextData<int64_t>(storage));
  if (reinterpret_cast<uintptr_t>(descriptor) % alignof(MemRefPrefix))
    return fail(context, ANNC_STATUS_INTERNAL, "unaligned memref storage");
  const MemRefPrefix prefix{view.allocated, view.aligned, view.offset};
  std::memcpy(descriptor, &prefix, sizeof(prefix));
  if (view.rank) {
    const size_t bytes = view.rank * sizeof(int64_t);
    std::memcpy(descriptor + sizeof(prefix), view.sizes, bytes);
    std::memcpy(descriptor + sizeof(prefix) + bytes, view.strides, bytes);
  }
  return descriptor;
}

// Exceptions from host callbacks must not unwind through generated C callers.
template <typename Task>
void* accessMemRef(const AnncKernelContext* abi, Task task) {
  if (!annc::kernels::validContext(abi)) return nullptr;
  // KernelContext only changes host service state; input access never writes
  // the borrowed AnncKernelContext or its input views.
  KernelContext context(const_cast<AnncKernelContext*>(abi));
  try {
    if (context.status() != ANNC_STATUS_OK) return nullptr;
    return task(context);
  } catch (const std::bad_alloc&) {
    return fail(context, ANNC_STATUS_ALLOCATION_FAILED,
                "memref access allocation failed");
  } catch (...) {
    return fail(context, ANNC_STATUS_INTERNAL, "memref access callback failed");
  }
}

}  // namespace

extern "C" const void* annc_context_get_input_memref(
    const AnncKernelContext* context, uint32_t input_index) {
  return accessMemRef(context, [&](KernelContext& accessor) -> void* {
    const auto* input = accessor.getStridedInput(input_index);
    return input ? exportMemRef(accessor, *input, *context->services) : nullptr;
  });
}

extern "C" void* annc_context_alloc_output_memref(AnncKernelContext* context,
                                                  uint32_t output_index,
                                                  uint32_t rank,
                                                  const int64_t* dims) {
  return accessMemRef(context, [&](KernelContext& accessor) -> void* {
    if (output_index >= context->num_outputs)
      return fail(accessor, ANNC_STATUS_INVALID_ARGUMENT,
                  "output index out of range");
    auto& output = context->outputs[output_index];
    size_t bytes;
    if ((output.flags & ANNC_TENSOR_READ_ONLY) ||
        !annc::kernels::contextByteCount(output.element_type, rank, dims,
                                         &bytes) ||
        output.element_type == ANNC_ELEMENT_TYPE_STRING)
      return fail(accessor, ANNC_STATUS_INVALID_ARGUMENT,
                  "invalid memref output request or missing output dtype");
    const auto type = output.element_type;
    if (!(output.flags & ANNC_TENSOR_BOUND)) {
      const AnncTensorDesc request{sizeof(AnncTensorDesc), type, rank,
                                   ANNC_ALLOCATION_NONE,   dims, nullptr};
      const auto code = context->services->allocate_output(
          context->services->opaque, output_index, &request, &output);
      if (code != ANNC_STATUS_OK)
        return fail(accessor, code, "memref output allocation failed");
      if (accessor.status() != ANNC_STATUS_OK) return nullptr;
    }
    if (output.element_type != type || output.rank != rank ||
        (output.flags & ANNC_TENSOR_READ_ONLY) ||
        !annc::kernels::validDenseView(output))
      return fail(accessor, ANNC_STATUS_INVALID_ARGUMENT,
                  "invalid memref output binding");
    for (uint32_t i = 0; i < rank; ++i)
      if (output.sizes[i] != dims[i])
        return fail(accessor, ANNC_STATUS_INVALID_ARGUMENT,
                    "memref output shape mismatch");
    return exportMemRef(accessor, output, *context->services);
  });
}
