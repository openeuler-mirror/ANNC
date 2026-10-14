#include "annc_ep_execution.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <utility>

namespace annc_ep {

namespace {

// Depth of ANNC parallel regions this thread is currently executing, set
// around every callback dispatched through KernelContext_ParallelFor. Mirrors
// the TensorFlow adapter's in_parallel_region semantics: nested parallel_for
// calls falling out of a region run serially instead of nesting.
thread_local int g_parallel_depth = 0;

class ParallelRegionGuard {
 public:
  ParallelRegionGuard() { ++g_parallel_depth; }
  ~ParallelRegionGuard() { --g_parallel_depth; }
  ParallelRegionGuard(const ParallelRegionGuard&) = delete;
  ParallelRegionGuard& operator=(const ParallelRegionGuard&) = delete;
};

// ANNC_EP_TIMING=1: per-call split of the context path -- "prepare" (read the
// fused node's inputs, build their memref descriptors, copy the contract)
// versus "kernel" (the shared-library entry: allocate the outputs through the
// callbacks, then the arithmetic, including the host-pool dispatch).  The
// residual against the profiler's per-node time is ORT's dispatch.
bool TimingEnabled() {
  static const bool enabled = std::getenv("ANNC_EP_TIMING") != nullptr;
  return enabled;
}

struct ContextTiming {
  std::atomic<int64_t> calls{0};
  std::atomic<int64_t> prepare_ns{0};
  std::atomic<int64_t> kernel_ns{0};
  std::atomic<int64_t> min_prepare_ns{INT64_MAX};
  std::atomic<int64_t> min_kernel_ns{INT64_MAX};

  void Note(int64_t prepare, int64_t kernel) {
    calls.fetch_add(1);
    prepare_ns.fetch_add(prepare);
    kernel_ns.fetch_add(kernel);
    int64_t current = min_prepare_ns.load();
    while (prepare < current &&
           !min_prepare_ns.compare_exchange_weak(current, prepare)) {
    }
    current = min_kernel_ns.load();
    while (kernel < current &&
           !min_kernel_ns.compare_exchange_weak(current, kernel)) {
    }
  }

  ~ContextTiming() {
    const int64_t n = calls.load();
    if (n == 0) return;
    const double per = 1.0 / static_cast<double>(n);
    std::fprintf(stderr,
                 "[annc-ep] context timing over %lld calls (mean/min us): "
                 "prepare=%.3f/%.3f kernel+alloc=%.3f/%.3f\n",
                 static_cast<long long>(n),
                 prepare_ns.load() * per / 1000.0,
                 min_prepare_ns.load() / 1000.0,
                 kernel_ns.load() * per / 1000.0,
                 min_kernel_ns.load() / 1000.0);
  }
};

ContextTiming& ContextTimer() {
  static ContextTiming timing;
  return timing;
}

// User data for one KernelContext_ParallelFor dispatch. The shard is consumed
// synchronously: ORT's parallel-for joins before returning.
//
// ANNC hands over *ranges* (begin, end); KernelContext_ParallelFor hands over
// one unit index at a time.  The shim therefore asks ORT for one batch per
// shard -- total = shard count, so ORT's batch loop calls back once per shard
// -- and each callback converts its batch index into the row range it covers.
// The naive mapping (num_batch = shard count, total = row count) makes ORT
// invoke the callback once per *row*, and for the 30-1000 row sites these
// models have that per-row std::function + TLS-guard cost dominated the
// kernel: 44 us mean against a 12 us floor, i.e. most of the per-node gap to
// ORT's own patched kernels (task 20261013-019, phase 2).
struct Shard {
  const std::function<void(std::int64_t, std::int64_t)>* fn;
  std::int64_t total_units;
  std::int64_t shards;
};

void ORT_API_CALL InvokeShard(void* user_data, size_t index) {
  auto* shard = static_cast<Shard*>(user_data);
  ParallelRegionGuard guard;
  const std::int64_t begin =
      static_cast<std::int64_t>(index) * shard->total_units / shard->shards;
  const std::int64_t end =
      (static_cast<std::int64_t>(index) + 1) * shard->total_units /
      shard->shards;
  (*shard->fn)(begin, end);
}

size_t ElementSize(AnncElementType type) {
  switch (type) {
    case ANNC_ELEMENT_TYPE_F32:
    case ANNC_ELEMENT_TYPE_I32:
      return 4;
    case ANNC_ELEMENT_TYPE_I64:
      return 8;
    default:
      return 0;
  }
}

// Byte count of a tensor description, or 0 when the shape is unusable.
size_t DescribedBytes(const AnncTensorDesc* desc) {
  const size_t element = ElementSize(desc->element_type);
  if (element == 0 || desc->dims == nullptr) return 0;
  size_t elements = 1;
  for (uint32_t i = 0; i < desc->rank; ++i) {
    if (desc->dims[i] < 0) return 0;
    elements *= static_cast<size_t>(desc->dims[i]);
  }
  return elements * element;
}

}  // namespace

// Public: the input resolver binds argument slots by the contract's element
// type as well as its rank, so it needs the same ONNX -> ANNC mapping the
// descriptor builder here uses.
bool AnncElementFromOrt(ONNXTensorElementDataType type, AnncElementType* out) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *out = ANNC_ELEMENT_TYPE_F32;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      *out = ANNC_ELEMENT_TYPE_I32;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      *out = ANNC_ELEMENT_TYPE_I64;
      return true;
    default:
      return false;
  }
}

// ---------------------------------------------------------------------------
// Argument resolution
// ---------------------------------------------------------------------------

ArgumentBinding BindArgumentsToInputs(
    const std::vector<InputShapeKey>& inputs,
    const std::vector<int64_t>& arg_ranks,
    const std::vector<AnncElementType>& arg_types) {
  ArgumentBinding binding;
  binding.indices.assign(arg_ranks.size(), -1);
  for (size_t i = 0; i < inputs.size(); ++i) {
    for (size_t arg = 0; arg < arg_ranks.size(); ++arg) {
      if (inputs[i].rank != arg_ranks[arg]) continue;
      if (arg >= arg_types.size() || inputs[i].type != arg_types[arg]) continue;
      if (binding.indices[arg] < 0) {
        binding.indices[arg] = static_cast<int64_t>(i);
      } else if (binding.ambiguous_arg < 0) {
        binding.ambiguous_arg = static_cast<int64_t>(arg);
        binding.ambiguous_inputs[0] = binding.indices[arg];
        binding.ambiguous_inputs[1] = static_cast<int64_t>(i);
      }
    }
  }
  return binding;
}

// ---------------------------------------------------------------------------
// Thread pool
// ---------------------------------------------------------------------------

int OrtThreadPool::num_threads() const {
  const unsigned hardware = std::thread::hardware_concurrency();
  if (hardware <= 1) return 2;
  return static_cast<int>(std::min<unsigned>(hardware, 256));
}

bool OrtThreadPool::in_parallel_region() const {
  return g_parallel_depth > 0;
}

void OrtThreadPool::parallel_for(
    std::int64_t total, const annc::threadpool::ParallelForOptions& options,
    const std::function<void(std::int64_t, std::int64_t)>& fn) {
  if (total <= 0) return;

  const std::int64_t grain =
      std::max<std::int64_t>(options.grain_size.value_or(1), 1);
  std::int64_t batches = (total + grain - 1) / grain;
  batches = std::min<std::int64_t>(batches, std::max(1, num_threads()));
  // How many shards to ask for is the *kernel's* decision: annc::threadpool::
  // parallel_for only reaches this adapter after Parallel.cpp decided the
  // range is worth parallelising, and the kernels pass a grain_size sized to
  // their own work (Kernel/builtin_kernels/parallel_grain.h).  ANNC_EP_SHARDS
  // overrides the count for tuning.
  if (const char* override_shards = std::getenv("ANNC_EP_SHARDS")) {
    batches = std::min<std::int64_t>(
        batches, std::max<std::int64_t>(1, std::atoll(override_shards)));
  }

  if (std::getenv("ANNC_EP_DEBUG") != nullptr) {
    static std::atomic<int> printed{0};
    if (printed.fetch_add(1) < 12) {
      std::fprintf(stderr,
                   "[annc-ep] pool: units=%lld grain=%lld shards=%lld\n",
                   static_cast<long long>(total),
                   static_cast<long long>(grain),
                   static_cast<long long>(batches));
    }
  }
  // One ORT batch per shard: ORT calls the shim once per batch index, and the
  // shim turns that index into the row range it covers (see InvokeShard).
  Shard shard{&fn, total, batches};
  OrtStatus* status = api_->KernelContext_ParallelFor(
      kernel_context_, &InvokeShard, static_cast<size_t>(batches),
      static_cast<size_t>(batches), &shard);
  if (status != nullptr) {
    // No pool behind the context (or an unusable one): run the whole range on
    // this thread. parallel_for must not fail.
    if (std::getenv("ANNC_EP_DEBUG") != nullptr) {
      std::fprintf(stderr, "[annc-ep] KernelContext_ParallelFor failed: %s\n",
                   api_->GetErrorMessage(status));
    }
    api_->ReleaseStatus(status);
    fn(0, total);
  }
}

// ---------------------------------------------------------------------------
// Execution context
// ---------------------------------------------------------------------------

OnnxExecutionContext::OnnxExecutionContext(
    const OrtApi* api, OrtKernelContext* kernel_context,
    std::vector<OutputContract> outputs, std::vector<uint32_t> output_slots)
    : api_(api),
      kernel_context_(kernel_context),
      outputs_(std::move(outputs)),
      output_slots_(std::move(output_slots)),
      allocated_(outputs_.size(), false),
      pool_(api, kernel_context) {
  handle_payload_.state = this;
  handle_ = reinterpret_cast<AnncExecutionHandle*>(&handle_payload_);
  context_.struct_size = sizeof(AnncExecutionContext);
  context_.abi_version = ANNC_EXECUTION_ABI_VERSION;
  context_.handle = handle_;
  context_.allocate_output = &OnnxExecutionContext::AllocateOutput;
  context_.allocate_temp = &OnnxExecutionContext::AllocateTemp;
  context_.report_error = &OnnxExecutionContext::ReportError;
  context_.thread_pool =
      reinterpret_cast<AnncThreadPoolHandle*>(static_cast<
          annc::threadpool::AnncThreadPool*>(&pool_));
}

OnnxExecutionContext::~OnnxExecutionContext() {
  if (temp_allocator_ != nullptr) {
    for (void* pointer : temps_) {
      if (OrtStatus* status = api_->AllocatorFree(temp_allocator_, pointer)) {
        api_->ReleaseStatus(status);
      }
    }
    api_->ReleaseAllocator(temp_allocator_);
  }
  if (mem_info_ != nullptr) api_->ReleaseMemoryInfo(mem_info_);
  if (error_ != nullptr) api_->ReleaseStatus(error_);
}

OnnxExecutionContext* OnnxExecutionContext::StateOf(
    AnncExecutionHandle* handle) {
  if (handle == nullptr) return nullptr;
  return reinterpret_cast<HandlePayload*>(handle)->state;
}

AnncStatusCode OnnxExecutionContext::Fail(AnncStatusCode status,
                                          const std::string& message) {
  if (error_ == nullptr) {
    error_ = api_->CreateStatus(ORT_FAIL,
                                ("ANNC EP execution context: " + message)
                                    .c_str());
    std::fprintf(stderr, "[annc-ep] execution error: %s\n", message.c_str());
  }
  return status;
}

void OnnxExecutionContext::RecordStatus(OrtStatus* status) {
  if (status == nullptr) return;
  if (error_ == nullptr) {
    error_ = status;
    return;
  }
  api_->ReleaseStatus(status);
}

OrtStatus* OnnxExecutionContext::TakeError() {
  OrtStatus* error = error_;
  error_ = nullptr;
  return error;
}

bool OnnxExecutionContext::AllOutputsAllocated() const {
  return std::all_of(allocated_.begin(), allocated_.end(),
                     [](bool allocated) { return allocated; });
}

AnncStatusCode OnnxExecutionContext::AllocateOutput(AnncExecutionHandle* handle,
                                                    uint32_t slot,
                                                    AnncTensorDesc* desc) {
  auto* state = StateOf(handle);
  if (state == nullptr) return ANNC_STATUS_INVALID_ARGUMENT;
  return state->AllocateOutputImpl(slot, desc);
}

AnncStatusCode OnnxExecutionContext::AllocateOutputImpl(uint32_t slot,
                                                        AnncTensorDesc* desc) {
  if (desc == nullptr || desc->struct_size < sizeof(*desc)) {
    return Fail(ANNC_STATUS_INVALID_ARGUMENT, "output description is invalid");
  }
  if (slot >= outputs_.size()) {
    return Fail(ANNC_STATUS_INVALID_ARGUMENT,
                "output slot " + std::to_string(slot) + " is out of range");
  }
  if (allocated_[slot]) {
    return Fail(ANNC_STATUS_INVALID_ARGUMENT,
                "output slot " + std::to_string(slot) +
                    " was allocated more than once");
  }
  const OutputContract& contract = outputs_[slot];
  if (contract.rank != static_cast<int64_t>(desc->rank)) {
    return Fail(ANNC_STATUS_INVALID_ARGUMENT,
                "output slot " + std::to_string(slot) + " rank mismatch");
  }
  if (contract.element_type != desc->element_type) {
    return Fail(ANNC_STATUS_INVALID_ARGUMENT,
                "output slot " + std::to_string(slot) + " element type mismatch");
  }
  // A zero-size tensor is legal (a batch-0 input produces an empty output):
  // the dims must be non-negative, but the byte count may be zero, and ORT
  // then hands back a tensor whose data pointer can be null. dims itself is
  // required for every rank -- rank 0 describes one element, not an empty
  // tensor.
  if (desc->dims == nullptr) {
    return Fail(ANNC_STATUS_INVALID_ARGUMENT,
                "output slot " + std::to_string(slot) + " has no dimensions");
  }
  for (uint32_t i = 0; i < desc->rank; ++i) {
    if (desc->dims[i] < 0) {
      return Fail(ANNC_STATUS_INVALID_ARGUMENT,
                  "output slot " + std::to_string(slot) + " has invalid dims");
    }
  }

  // The site's slots and the fused node's outputs can be ordered differently
  // (the pattern lists its sites; ORT numbers the partition's escaping
  // values). The EP resolves that mapping by tensor name at Compile time and
  // hands it over here; without one the orders agree.
  const uint32_t physical =
      slot < output_slots_.size() ? output_slots_[slot] : slot;
  OrtValue* value = nullptr;
  if (OrtStatus* status = api_->KernelContext_GetOutput(
          kernel_context_, physical, desc->dims, desc->rank, &value)) {
    RecordStatus(status);  // keeps the ORT message; ownership moves here
    return ANNC_STATUS_ALLOCATION_FAILED;
  }

  // The kernel states the element type it wants; ORT allocates per the node's
  // declared output type. A mismatch would mean the kernel interpreted the
  // buffer wrongly, so it is rejected here instead of silently accepted.
  OrtTensorTypeAndShapeInfo* info = nullptr;
  if (OrtStatus* status = api_->GetTensorTypeAndShape(value, &info)) {
    RecordStatus(status);
    return ANNC_STATUS_INTERNAL;
  }
  ONNXTensorElementDataType ort_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  if (OrtStatus* status = api_->GetTensorElementType(info, &ort_type)) {
    // The handle is one this context just obtained; a failure here would
    // leave the type undefined and trip the mismatch check below. Release the
    // status instead of dropping it.
    api_->ReleaseStatus(status);
  }
  api_->ReleaseTensorTypeAndShapeInfo(info);
  AnncElementType expected = 0;
  if (!AnncElementFromOrt(ort_type, &expected) || expected != desc->element_type) {
    return Fail(ANNC_STATUS_INVALID_ARGUMENT,
                "output slot " + std::to_string(slot) + " (graph output " +
                    std::to_string(physical) +
                    ") does not match the graph's element type");
  }

  void* data = nullptr;
  if (OrtStatus* status = api_->GetTensorMutableData(value, &data)) {
    RecordStatus(status);
    return ANNC_STATUS_INTERNAL;
  }
  desc->data = data;
  allocated_[slot] = true;
  return ANNC_STATUS_OK;
}

AnncStatusCode OnnxExecutionContext::AllocateTemp(AnncExecutionHandle* handle,
                                                  AnncTensorDesc* desc) {
  auto* state = StateOf(handle);
  if (state == nullptr) return ANNC_STATUS_INVALID_ARGUMENT;
  return state->AllocateTempImpl(desc);
}

AnncStatusCode OnnxExecutionContext::AllocateTempImpl(AnncTensorDesc* desc) {
  if (desc == nullptr || desc->struct_size < sizeof(*desc)) {
    return Fail(ANNC_STATUS_INVALID_ARGUMENT,
                "temporary description is invalid");
  }
  if (ElementSize(desc->element_type) == 0) {
    return Fail(ANNC_STATUS_INVALID_ARGUMENT,
                "temporary description has an unsupported element type");
  }
  if (desc->dims == nullptr) {
    return Fail(ANNC_STATUS_INVALID_ARGUMENT,
                "temporary description has no dimensions");
  }
  for (uint32_t i = 0; i < desc->rank; ++i) {
    if (desc->dims[i] < 0) {
      return Fail(ANNC_STATUS_INVALID_ARGUMENT,
                  "temporary description has invalid dims");
    }
  }
  const size_t bytes = DescribedBytes(desc);
  if (bytes == 0) {
    // An empty scratch buffer needs no allocation and nothing to free.
    desc->data = nullptr;
    return ANNC_STATUS_OK;
  }

  if (mem_info_ == nullptr) {
    if (OrtStatus* status = api_->CreateCpuMemoryInfo(
            OrtArenaAllocator, OrtMemTypeDefault, &mem_info_)) {
      RecordStatus(status);
      return ANNC_STATUS_ALLOCATION_FAILED;
    }
  }
  if (temp_allocator_ == nullptr) {
    if (OrtStatus* status = api_->KernelContext_GetAllocator(
            kernel_context_, mem_info_, &temp_allocator_)) {
      RecordStatus(status);
      return ANNC_STATUS_ALLOCATION_FAILED;
    }
  }

  void* data = nullptr;
  if (OrtStatus* status =
          api_->AllocatorAlloc(temp_allocator_, bytes, &data)) {
    RecordStatus(status);
    return ANNC_STATUS_ALLOCATION_FAILED;
  }
  if (desc->flags & ANNC_ALLOCATION_ZERO) std::memset(data, 0, bytes);
  temps_.push_back(data);
  desc->data = data;
  return ANNC_STATUS_OK;
}

void OnnxExecutionContext::ReportError(AnncExecutionHandle* handle,
                                       AnncStatusCode status,
                                       const char* message) {
  auto* state = StateOf(handle);
  if (state == nullptr) return;
  state->Fail(status, message != nullptr ? message : "unspecified kernel error");
}

// ---------------------------------------------------------------------------
// Context-ABI site execution
// ---------------------------------------------------------------------------

namespace {

// The EP accepts rank 0..4 kernel arguments (the site contract rejects
// anything larger). The descriptor is laid out as the MLIR C interface
// expects: { void* allocated; void* aligned; intptr_t offset;
//             intptr_t sizes[rank]; intptr_t strides[rank]; }
constexpr int64_t kMaxRank = 4;

struct MemRefStorage {
  alignas(16) unsigned char
      bytes[2 * sizeof(void*) + sizeof(int64_t) * (1 + 2 * kMaxRank)];
};

void BuildMemRefDescriptor(int64_t rank, const int64_t* dims, void* data,
                           MemRefStorage* storage) {
  unsigned char* base = storage->bytes;
  std::memcpy(base, &data, sizeof(void*));                  // allocated
  std::memcpy(base + sizeof(void*), &data, sizeof(void*));  // aligned
  const int64_t offset = 0;
  std::memcpy(base + 2 * sizeof(void*), &offset, sizeof(offset));
  int64_t* sizes =
      reinterpret_cast<int64_t*>(base + 2 * sizeof(void*) + sizeof(int64_t));
  for (int64_t i = 0; i < rank; ++i) sizes[i] = dims[i];
  int64_t* strides = sizes + rank;
  if (rank > 0) {
    strides[rank - 1] = 1;
    for (int64_t i = rank - 1; i > 0; --i) strides[i - 1] = strides[i] * sizes[i];
  }
}

}  // namespace

OrtStatus* ComputeWithContext(const OrtApi* api,
                              OrtKernelContext* kernel_context,
                              const ContextCall& request) {
  const bool timing = TimingEnabled();
  const auto t_start = timing ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
  OnnxExecutionContext context(api, kernel_context, request.outputs,
                               request.output_slots);

  std::vector<MemRefStorage> storage(request.arg_indices.size());
  std::vector<void*> args;
  args.reserve(request.arg_indices.size() + 1);
  args.push_back(context.abi());

  for (size_t i = 0; i < request.arg_indices.size(); ++i) {
    const OrtValue* value = nullptr;
    if (OrtStatus* status = api->KernelContext_GetInput(
            kernel_context, static_cast<size_t>(request.arg_indices[i]),
            &value)) {
      return status;
    }
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (OrtStatus* status = api->GetTensorTypeAndShape(value, &info)) {
      return status;
    }
    size_t rank = 0;
    if (OrtStatus* status = api->GetDimensionsCount(info, &rank)) {
      api->ReleaseTensorTypeAndShapeInfo(info);
      return status;
    }
    ONNXTensorElementDataType ort_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    if (OrtStatus* status = api->GetTensorElementType(info, &ort_type)) {
      api->ReleaseTensorTypeAndShapeInfo(info);
      return status;
    }
    if (static_cast<int64_t>(rank) != request.arg_ranks[i] ||
        rank > static_cast<size_t>(kMaxRank)) {
      api->ReleaseTensorTypeAndShapeInfo(info);
      return api->CreateStatus(
          ORT_FAIL, "ANNC EP: kernel argument rank does not match the site");
    }
    int64_t dims[kMaxRank] = {0, 0, 0, 0};
    if (rank > 0) {
      if (OrtStatus* status = api->GetDimensions(info, dims, rank)) {
        api->ReleaseTensorTypeAndShapeInfo(info);
        api->ReleaseStatus(status);  // replaced by the message below
        return api->CreateStatus(
            ORT_FAIL, "ANNC EP: cannot read a kernel argument shape");
      }
    }
    api->ReleaseTensorTypeAndShapeInfo(info);
    AnncElementType element = 0;
    if (!AnncElementFromOrt(ort_type, &element)) {
      return api->CreateStatus(
          ORT_FAIL, "ANNC EP: unsupported kernel argument element type");
    }
    if (element != request.arg_types[i]) {
      return api->CreateStatus(
          ORT_FAIL,
          "ANNC EP: kernel argument element type does not match the site");
    }
    const void* raw = nullptr;
    if (OrtStatus* status = api->GetTensorData(value, &raw)) return status;
    BuildMemRefDescriptor(static_cast<int64_t>(rank), dims,
                          const_cast<void*>(raw), &storage[i]);
    args.push_back(storage[i].bytes);
  }

  const auto t_prepared = timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
  const std::int32_t status_code = CallContextKernel(request.symbol, args);
  if (timing) {
    const auto t_end = std::chrono::steady_clock::now();
    ContextTimer().Note(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_prepared - t_start)
            .count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_prepared)
            .count());
  }
  if (OrtStatus* error = context.TakeError()) return error;
  if (status_code != 0) {
    char message[128];
    std::snprintf(message, sizeof(message),
                  "ANNC EP: kernel returned status %d", status_code);
    return api->CreateStatus(ORT_FAIL, message);
  }
  if (!context.AllOutputsAllocated()) {
    return api->CreateStatus(
        ORT_FAIL, "ANNC EP: kernel did not allocate every declared output");
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Kernel entry dispatch
// ---------------------------------------------------------------------------

std::int32_t CallContextKernel(void* symbol, const std::vector<void*>& args) {
  switch (args.size()) {
    case 1:
      return reinterpret_cast<std::int32_t (*)(void*)>(symbol)(args[0]);
    case 2:
      return reinterpret_cast<std::int32_t (*)(void*, void*)>(symbol)(args[0],
                                                                     args[1]);
    case 3:
      return reinterpret_cast<std::int32_t (*)(void*, void*, void*)>(symbol)(
          args[0], args[1], args[2]);
    case 4:
      return reinterpret_cast<std::int32_t (*)(void*, void*, void*, void*)>(
          symbol)(args[0], args[1], args[2], args[3]);
    case 5:
      return reinterpret_cast<std::int32_t (*)(void*, void*, void*, void*,
                                               void*)>(symbol)(
          args[0], args[1], args[2], args[3], args[4]);
    case 6:
      return reinterpret_cast<std::int32_t (*)(void*, void*, void*, void*,
                                               void*, void*)>(symbol)(
          args[0], args[1], args[2], args[3], args[4], args[5]);
    case 7:
      return reinterpret_cast<std::int32_t (*)(void*, void*, void*, void*,
                                               void*, void*, void*)>(symbol)(
          args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
    case 8:
      return reinterpret_cast<std::int32_t (*)(void*, void*, void*, void*,
                                               void*, void*, void*, void*)>(
          symbol)(args[0], args[1], args[2], args[3], args[4], args[5],
                  args[6], args[7]);
    default:
      return -1;
  }
}

}  // namespace annc_ep
