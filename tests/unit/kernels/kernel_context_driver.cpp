// Standalone host for the two representative context kernels.
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "Context/KernelContextUtils.h"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Allocation {
  std::vector<int64_t> dims;
  std::vector<int64_t> strides;
  std::vector<uint64_t> storage;
};

struct Host {
  AnncStatusCode error = ANNC_STATUS_OK;
  std::vector<std::unique_ptr<Allocation>> allocations;
  std::vector<uint32_t> outputOrder;
  int shardCalls = 0;

  static AnncStatusCode allocate(void* opaque, const AnncTensorDesc* request,
                                 AnncTensorView* result) {
    auto& host = *static_cast<Host*>(opaque);
    size_t bytes = 0;
    if (!request || !result || request->struct_size != sizeof(AnncTensorDesc) ||
        !annc::kernels::contextByteCount(request->element_type, request->rank,
                                         request->dims, &bytes))
      return ANNC_STATUS_INVALID_ARGUMENT;
    auto owner = std::make_unique<Allocation>();
    if (request->rank)
      owner->dims.assign(request->dims, request->dims + request->rank);
    owner->strides.resize(request->rank);
    int64_t stride = 1;
    for (uint32_t i = request->rank; i != 0; --i) {
      owner->strides[i - 1] = stride;
      stride *= owner->dims[i - 1];
    }
    owner->storage.resize((bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
    void* data = bytes ? owner->storage.data() : nullptr;
    if (bytes) std::memset(data, (request->flags & ANNC_ALLOCATION_ZERO) ? 0 : 0xa5,
                           bytes);
    *result = {data, data, 0, owner->dims.data(), owner->strides.data(),
               request->element_type, request->rank, ANNC_LAYOUT_STRIDED,
               ANNC_TENSOR_BOUND};
    host.allocations.push_back(std::move(owner));
    return ANNC_STATUS_OK;
  }

  static AnncStatusCode output(void* opaque, uint32_t slot,
                               const AnncTensorDesc* request,
                               AnncTensorView* result) {
    static_cast<Host*>(opaque)->outputOrder.push_back(slot);
    return allocate(opaque, request, result);
  }

  static void report(void* opaque, AnncStatusCode code, const char*) {
    auto& host = *static_cast<Host*>(opaque);
    if (host.error == ANNC_STATUS_OK) {
      host.error = code;
    }
  }
  static AnncStatusCode status(void* opaque) {
    return static_cast<Host*>(opaque)->error;
  }
  static int32_t threads(void*) { return 2; }

  static AnncStatusCode parallel(void* opaque, int64_t total, int64_t,
                                 AnncContextTask task, void* data) {
    const int64_t middle = total / 2;
    task(data, 0, middle);
    task(data, middle, total);
    return static_cast<Host*>(opaque)->error;
  }

  static AnncStatusCode shard(void* opaque, int32_t maxParallelism,
                              int64_t total, int64_t,
                              AnncContextTask task, void* data) {
    auto& host = *static_cast<Host*>(opaque);
    ++host.shardCalls;
    if (total <= 1 || maxParallelism <= 1) {
      task(data, 0, total);
      return host.error;
    }
    const int64_t middle = total / 2;
    auto worker = std::async(std::launch::async,
                             [=] { task(data, 0, middle); });
    task(data, middle, total);
    worker.get();
    return host.error;
  }

  AnncRuntimeServices services() {
    return {sizeof(AnncRuntimeServices), ANNC_CONTEXT_ABI_VERSION, this,
            output, allocate, report, status, parallel, threads, shard,
            nullptr};
  }
};

template <typename T, size_t Rank>
AnncTensorView input(T* data, const std::array<int64_t, Rank>& dims,
                     const std::array<int64_t, Rank>& strides,
                     int64_t offset = 0) {
  return {data, data, offset, dims.data(), strides.data(),
          annc::kernels::contextElementType<T>(), static_cast<uint32_t>(Rank),
          ANNC_LAYOUT_STRIDED, ANNC_TENSOR_BOUND | ANNC_TENSOR_READ_ONLY};
}

AnncKernelContext context(const AnncTensorView* inputs, uint32_t inputCount,
                          AnncTensorView* outputs, uint32_t outputCount,
                          const AnncRuntimeServices* services) {
  return {sizeof(AnncKernelContext), ANNC_CONTEXT_ABI_VERSION, inputs,
          inputCount, outputs, outputCount, nullptr, 0, nullptr, 0, services};
}

template <typename T>
std::vector<T> values(const AnncTensorView& view) {
  size_t bytes = 0;
  require(annc::kernels::contextByteCount(view.element_type, view.rank,
                                          view.sizes, &bytes),
          "invalid result shape");
  if (!bytes) return {};
  const T* data = annc::kernels::contextData<T>(view);
  return {data, data + bytes / sizeof(T)};
}

void testHostServices() {
  Host host;
  auto services = host.services();
  AnncTensorView output{};
  auto raw = context(nullptr, 0, &output, 1, &services);
  annc::kernels::KernelContext view(&raw);
  auto* bound = view.allocateOutput<float>(0, {2}, ANNC_ALLOCATION_ZERO);
  require(bound && values<float>(*bound) == std::vector<float>({0, 0}),
          "zero output allocation failed");
  require(view.allocateOutput<float>(0, {2}, ANNC_ALLOCATION_ZERO) == bound &&
              host.outputOrder.size() == 1,
          "bound output was not reused");
  int64_t count = 0;
  require(view.getThreadPool().parallelFor(
              3, [&](int64_t first, int64_t last) { count += last - first; }) ==
              ANNC_STATUS_OK &&
              count == 3,
          "parallel service failed");
}

void testMatMul(AnncKernelFn kernel) {
  float lhs[] = {-99, 1, 2, 3, 4, 5, 6};
  float rhs[] = {7, 8, 9, 10, 11, 12};
  const std::array<int64_t, 2> lhsDims{2, 3}, lhsStrides{3, 1};
  const std::array<int64_t, 2> rhsDims{3, 2}, rhsStrides{2, 1};
  AnncTensorView inputs[] = {input(lhs, lhsDims, lhsStrides, 1),
                             input(rhs, rhsDims, rhsStrides)};
  AnncTensorView output{};
  Host host;
  auto services = host.services();
  auto raw = context(inputs, 2, &output, 1, &services);
  require(kernel(&raw) == ANNC_STATUS_OK &&
              values<float>(output) == std::vector<float>({58, 64, 139, 154}) &&
              host.shardCalls > 0,
          "MatMul result or shard service failed");
  require(kernel(&raw) == ANNC_STATUS_OK && host.outputOrder.size() == 1,
          "MatMul output was not reused");

  raw.abi_version = 2;
  require(kernel(&raw) == ANNC_STATUS_INVALID_ARGUMENT,
          "wrong context ABI was accepted");
}

void testGather(AnncKernelFn kernel) {
  float data[] = {10, 11, 20, 21, 30, 31, 40, 41};
  int64_t keys[] = {99, 2, 99, 0, 99, 2, 99, 1};
  int32_t begin[] = {0, 1};
  const std::array<int64_t, 2> dims{4, 2}, strides{2, 1};
  const std::array<int64_t, 1> beginDims{2}, beginStrides{1};
  AnncTensorView inputs[] = {input(data, dims, strides),
                             input(keys, dims, strides),
                             input(begin, beginDims, beginStrides)};
  AnncTensorView outputs[3]{};
  Host host;
  auto services = host.services();
  auto raw = context(inputs, 3, outputs, 3, &services);
  require(kernel(&raw) == ANNC_STATUS_OK &&
              values<int64_t>(outputs[0]) == std::vector<int64_t>({2, 0, 1}) &&
              values<int32_t>(outputs[1]) == std::vector<int32_t>({0, 1, 0, 2}) &&
              values<float>(outputs[2]) ==
                  std::vector<float>({30, 31, 10, 11, 20, 21}),
          "Gather result failed");

  for (int i = 0; i < 4; ++i) keys[2 * i + 1] = 3;
  Host second;
  services = second.services();
  std::fill(std::begin(outputs), std::end(outputs), AnncTensorView{});
  require(kernel(&raw) == ANNC_STATUS_OK &&
              values<int64_t>(outputs[0]) == std::vector<int64_t>({3}) &&
              values<float>(outputs[2]) == std::vector<float>({40, 41}),
          "Gather dynamic output shape failed");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 4) return 2;
  void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    std::cerr << dlerror() << '\n';
    return 1;
  }
  int result = 0;
  try {
    const char* matmulSymbol =
        argc == 4 ? argv[2] : "_mlir_ciface_ANNCKernel_ContextV3_MatMul_float_auto";
    const char* gatherSymbol =
        argc == 4 ? argv[3]
                  : "_mlir_ciface_ANNCKernel_ContextV3_KPFusedGather_float_"
                    "int64_t_int32_t_int32_t_auto";
    auto matmul = reinterpret_cast<AnncKernelFn>(dlsym(library, matmulSymbol));
    auto gather = reinterpret_cast<AnncKernelFn>(dlsym(library, gatherSymbol));
    require(matmul && gather, "context symbols missing");
    testHostServices();
    testMatMul(matmul);
    testGather(gather);
    std::cout << "PASS: context ABI, host services, MatMul and Gather\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    result = 1;
  }
  dlclose(library);
  return result;
}
