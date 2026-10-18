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
  // The shard-to-range mapping ORT itself uses to split `total` units into
  // `num_batches` batches (TryBatchParallelFor -> PartitionWork): the first
  // `total % shards` shards carry one extra unit.
  const std::int64_t per = shard->total_units / shard->shards;
  const std::int64_t extra = shard->total_units % shard->shards;
  const std::int64_t i = static_cast<std::int64_t>(index);
  const std::int64_t begin =
      i < extra ? i * (per + 1) : extra * (per + 1) + (i - extra) * per;
  const std::int64_t end = begin + per + (i < extra ? 1 : 0);
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

// ANNC_EP_DEBUG=1: the first decisions this process made -- what the kernel
// reported and how many shards that bought.  shards=1 means the range ran on
// the calling thread, which is what ORT's own kernels do below the cost
// model's threshold.
void LogPoolDecision(std::int64_t total,
                     const annc::threadpool::ParallelForOptions& options,
                     std::int64_t shards) {
  if (std::getenv("ANNC_EP_DEBUG") == nullptr) return;
  static std::atomic<int> printed{0};
  if (printed.fetch_add(1) >= 12) return;
  std::fprintf(stderr,
               "[annc-ep] pool: units=%lld cost=%lld grain=%lld shards=%lld\n",
               static_cast<long long>(total),
               static_cast<long long>(options.cost_per_unit.value_or(0)),
               static_cast<long long>(options.grain_size.value_or(0)),
               static_cast<long long>(shards));
}

std::int64_t DivCeil(std::int64_t a, std::int64_t b) {
  return (a + b - 1) / b;
}

// Port of concurrency::ThreadPool::CalculateParallelForBlock
// (onnxruntime/core/common/threadpool.cc), the rule that sizes the blocks
// ThreadPool::ParallelFor hands to the pool: no block below kTaskSize cycles of
// work, no more than four blocks per thread, then grown coarser while doing so
// keeps the block count dividing evenly across the pool.
std::int64_t CalculateParallelForBlock(std::int64_t n, double cost_per_unit,
                                       std::int64_t num_threads) {
  constexpr std::int64_t kTaskSize = 40000;  // Eigen TensorCostModel
  constexpr std::int64_t kMaxOvershardingFactor = 4;
  const auto efficiency = [num_threads](std::int64_t block_count) {
    return static_cast<double>(block_count) /
           static_cast<double>(DivCeil(block_count, num_threads) * num_threads);
  };

  const std::int64_t task_block = std::max<std::int64_t>(
      1, static_cast<std::int64_t>(kTaskSize / cost_per_unit));
  std::int64_t block = std::min<std::int64_t>(
      n, std::max<std::int64_t>(
             DivCeil(n, kMaxOvershardingFactor * num_threads), task_block));
  const std::int64_t max_block = std::min<std::int64_t>(n, 2 * block);

  std::int64_t block_count = DivCeil(n, block);
  double max_efficiency = efficiency(block_count);
  for (std::int64_t prev = block_count; max_efficiency < 1.0 && prev > 1;) {
    const std::int64_t coarser = DivCeil(n, prev - 1);
    if (coarser > max_block) break;  // reached the max block size
    const std::int64_t coarser_count = DivCeil(n, coarser);
    prev = coarser_count;
    const double coarser_efficiency = efficiency(coarser_count);
    if (coarser_efficiency + 0.01 >= max_efficiency) {
      block = coarser;
      if (max_efficiency < coarser_efficiency) {
        max_efficiency = coarser_efficiency;
      }
    }
  }
  return block;
}

int OrtThreadPool::num_threads() const {
#if defined(ANNC_EP_NATIVE_POOL)
  if (native_enabled_) {
    // The real degree of the session's operator pool (its own users' query,
    // so the common layer's serial/parallel gate now sees what ORT's kernels
    // see).  DegreeOfParallelism itself answers 1 for a null pool.
    const int dop =
        onnxruntime::concurrency::ThreadPool::DegreeOfParallelism(native_);
    return dop > 0 ? dop : 1;
  }
#endif
  const unsigned hardware = std::thread::hardware_concurrency();
  if (hardware <= 1) return 2;
  return static_cast<int>(std::min<unsigned>(hardware, 256));
}

OrtThreadPool::OrtThreadPool(const OrtApi* api, OrtKernelContext* kernel_context)
    : api_(api), kernel_context_(kernel_context) {
#if defined(ANNC_EP_NATIVE_POOL)
  native_enabled_ = native::PoolAvailable();
  if (native_enabled_) native_ = native::PoolOf(kernel_context);
#endif
}

void* OrtThreadPool::native_pool() const {
#if defined(ANNC_EP_NATIVE_POOL)
  if (native_enabled_) return native_;
#endif
  return nullptr;
}

// ANNC_EP_DEBUG=1: what the native dispatch actually did.  `shards` counts
// range callbacks -- the shard count ORT chose, serial included (1 = it ran
// the whole range on the caller) -- and is not the number of threads that
// executed them.
void LogNativePoolDecision(
    std::int64_t total, const annc::threadpool::ParallelForOptions& options,
    std::int64_t shards) {
  if (std::getenv("ANNC_EP_DEBUG") == nullptr) return;
  static std::atomic<int> printed{0};
  if (printed.fetch_add(1) >= 12) return;
  std::fprintf(stderr,
               "[annc-ep] pool: native units=%lld cost=%lld grain=%lld "
               "shards=%lld\n",
               static_cast<long long>(total),
               static_cast<long long>(options.cost_per_unit.value_or(0)),
               static_cast<long long>(options.grain_size.value_or(0)),
               static_cast<long long>(shards));
}

bool OrtThreadPool::in_parallel_region() const {
  return g_parallel_depth > 0;
}

void OrtThreadPool::parallel_for(
    std::int64_t total, const annc::threadpool::ParallelForOptions& options,
    const std::function<void(std::int64_t, std::int64_t)>& fn) {
  if (total <= 0) return;

#if defined(ANNC_EP_NATIVE_POOL)
  if (native_enabled_) {
    const std::int64_t grain =
        std::max<std::int64_t>(options.grain_size.value_or(0), 0);
    if (options.cost_per_unit.has_value()) {
      // The framework's own decision: TryParallelFor applies Eigen's cost
      // model against the pool's real degree of parallelism and either runs
      // fn(0, total) on the caller or hands out CalculateParallelForBlock-
      // sized ranges -- the rule ORT's own patched kernels run under, so the
      // shard count is no longer ANNC's to estimate (the estimate was the
      // source of the 8x per-node dispatch gap and of 17-shard dispatches on
      // two-thread pools; task 20261013-019).  ANNC's cost units are already
      // TensorOpCost's compute-cycles scale (Support/ThreadPool/
      // RowParallel.h), so the number passes through.  A null pool (intra-
      // op=1) is serial inside TryParallelFor itself.
      const double cost_per_unit = static_cast<double>(
          std::max<std::int64_t>(*options.cost_per_unit, 1));
      const onnxruntime::TensorOpCost tensor_cost{0.0, 0.0, cost_per_unit};
      std::atomic<std::int64_t> shards{0};
      onnxruntime::concurrency::ThreadPool::TryParallelFor(
          native_, static_cast<std::ptrdiff_t>(total), tensor_cost,
          [&](std::ptrdiff_t first, std::ptrdiff_t last) {
            ParallelRegionGuard guard;
            shards.fetch_add(1, std::memory_order_relaxed);
            fn(static_cast<std::int64_t>(first),
               static_cast<std::int64_t>(last));
          });
      LogNativePoolDecision(total, options, shards.load());
      return;
    }
    if (grain > 0) {
      // A kernel that reports only grain_size asks for fixed-size blocks:
      // ORT's own fixed-block entry, with the grain as the block size.  The
      // serial case (total <= grain) is handled inside; a null pool has no
      // member to call, and serial is the only thing it could mean.
      if (native_ == nullptr) {
        LogNativePoolDecision(total, options, 1);
        fn(0, total);
        return;
      }
      std::atomic<std::int64_t> shards{0};
      native_->ParallelForFixedBlockSizeScheduling(
          static_cast<std::ptrdiff_t>(total),
          static_cast<std::ptrdiff_t>(grain),
          [&](std::ptrdiff_t first, std::ptrdiff_t last) {
            ParallelRegionGuard guard;
            shards.fetch_add(1, std::memory_order_relaxed);
            fn(static_cast<std::int64_t>(first),
               static_cast<std::int64_t>(last));
          });
      LogNativePoolDecision(total, options, shards.load());
      return;
    }
    // No signal at all: fall through to the public path below, whose
    // num_batches default now equals TryBatchParallelFor(..., 0) because
    // num_threads() is the real degree of parallelism.
  }
#endif

  // The scheduling decision belongs to the framework in ORT's own kernels:
  // they call ThreadPool::TryParallelFor(total, TensorOpCost, fn), whose
  // ParallelFor step (core/common/threadpool.cc) consults Eigen's cost model
  // and either runs the whole range on the caller or cuts it into
  // CalculateParallelForBlock-sized blocks for the pool.  A plugin EP only
  // gets the public KernelContext_ParallelFor, whose header puts that sizing on
  // the caller ("Pitfall: Caller should cap num_batches to a reasonable value
  // based on the cost of fn and the value of total":
  // include/onnxruntime/core/platform/threadpool.h).  So this adapter applies
  // the framework's own rule -- the same Eigen model and the same block
  // calculation -- to the cost the kernel reports in ORT's units
  // (include/Support/ThreadPool/RowParallel.h).  Eigen's constants
  // (unsupported/Eigen/CXX11/src/Tensor/TensorCostModel.h): kStartupCycles =
  // kPerThreadCycles = 100000.
  //
  // A kernel that reports only grain_size asks for fixed-size blocks instead:
  // that is the TF adapter's kFixedBlockSize path, honoured verbatim.  With
  // neither signal the range is split like the in-tree pool splits an
  // optionless call (Support/ThreadPool/Parallel.cpp defaults grain_size to 1,
  // one unit per block) and like ORT itself splits it when the caller passes
  // num_batches <= 0 ("it will be replaced to the value of
  // DegreeOfParallelism()", include/onnxruntime/core/platform/threadpool.h) --
  // our num_threads() stands in for that degree, see below.
  const std::int64_t grain =
      std::max<std::int64_t>(options.grain_size.value_or(0), 0);
  std::int64_t batches = 0;
  if (options.cost_per_unit.has_value()) {
    constexpr double kStartupCycles = 100000.0;
    constexpr double kPerThreadCycles = 100000.0;
    const double cost_per_unit =
        static_cast<double>(std::max<std::int64_t>(*options.cost_per_unit, 1));
    const double cost = static_cast<double>(total) * cost_per_unit;
    const double threads = (cost - kStartupCycles) / kPerThreadCycles + 0.9;
    // ORT's own cap is DegreeOfParallelism(pool) = pool threads + the calling
    // thread, which no public API exposes; num_threads() is the advisory bound
    // (the hardware_concurrency fallback an in-tree EP adapter uses too, see
    // providers/dnnl/dnnl_threadpool.h).  It only binds for ranges past ~25M
    // cycles of work, far above any site these kernels see.
    if (threads < 1.0) {
      batches = 1;  // CostModel::numThreads() == 1: run on the caller
    } else {
      const std::int64_t block =
          CalculateParallelForBlock(total, cost_per_unit, num_threads());
      batches = DivCeil(total, block);
    }
  } else if (grain > 0) {
    batches = DivCeil(total, grain);
  } else {
    batches = std::min<std::int64_t>(total, num_threads());
  }
  batches =
      std::min<std::int64_t>(std::max<std::int64_t>(batches, 1), num_threads());
  if (batches <= 1) {
    // Below the framework's parallelisation threshold: run the whole range
    // here, the way ThreadPool::ParallelFor does with f(0, n).
    LogPoolDecision(total, options, /*shards=*/1);
    fn(0, total);
    return;
  }
  LogPoolDecision(total, options, batches);
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
    const std::vector<OutputContract>& outputs,
    const std::vector<uint32_t>* output_slots)
    : api_(api),
      kernel_context_(kernel_context),
      outputs_(&outputs),
      output_slots_(output_slots),
      allocated_(outputs.size(), false),
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
  if (slot >= outputs_->size()) {
    return Fail(ANNC_STATUS_INVALID_ARGUMENT,
                "output slot " + std::to_string(slot) + " is out of range");
  }
  if (allocated_[slot]) {
    return Fail(ANNC_STATUS_INVALID_ARGUMENT,
                "output slot " + std::to_string(slot) +
                    " was allocated more than once");
  }
  const OutputContract& contract = (*outputs_)[slot];
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
      output_slots_ != nullptr && slot < output_slots_->size()
          ? (*output_slots_)[slot]
          : slot;
  OrtValue* value = nullptr;
  if (OrtStatus* status = api_->KernelContext_GetOutput(
          kernel_context_, physical, desc->dims, desc->rank, &value)) {
    RecordStatus(status);  // keeps the ORT message; ownership moves here
    return ANNC_STATUS_ALLOCATION_FAILED;
  }

  // The kernel states the element type it wants; ORT allocates per the node's
  // declared output type. A mismatch would mean the kernel interpreted the
  // buffer wrongly, so it is rejected here instead of silently accepted.
  ONNXTensorElementDataType ort_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  if (api_->GetTensorElementTypeAndShapeDataReference != nullptr) {
    const int64_t* shape = nullptr;
    size_t rank = 0;
    if (OrtStatus* status = api_->GetTensorElementTypeAndShapeDataReference(
            value, &ort_type, &shape, &rank)) {
      RecordStatus(status);
      return ANNC_STATUS_INTERNAL;
    }
  } else {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (OrtStatus* status = api_->GetTensorTypeAndShape(value, &info)) {
      RecordStatus(status);
      return ANNC_STATUS_INTERNAL;
    }
    if (OrtStatus* status = api_->GetTensorElementType(info, &ort_type)) {
      RecordStatus(status);
      api_->ReleaseTensorTypeAndShapeInfo(info);
      return ANNC_STATUS_INTERNAL;
    }
    api_->ReleaseTensorTypeAndShapeInfo(info);
  }
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
                               &request.output_slots);

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
    ONNXTensorElementDataType ort_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    const int64_t* shape = nullptr;
    size_t rank = 0;
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (api->GetTensorElementTypeAndShapeDataReference != nullptr) {
      if (OrtStatus* status = api->GetTensorElementTypeAndShapeDataReference(
              value, &ort_type, &shape, &rank)) return status;
    } else {
      if (OrtStatus* status = api->GetTensorTypeAndShape(value, &info)) return status;
      if (OrtStatus* status = api->GetDimensionsCount(info, &rank)) {
        api->ReleaseTensorTypeAndShapeInfo(info);
        return status;
      }
      if (OrtStatus* status = api->GetTensorElementType(info, &ort_type)) {
        api->ReleaseTensorTypeAndShapeInfo(info);
        return status;
      }
    }
    if (static_cast<int64_t>(rank) != request.arg_ranks[i] ||
        rank > static_cast<size_t>(kMaxRank)) {
      if (info != nullptr) api->ReleaseTensorTypeAndShapeInfo(info);
      return api->CreateStatus(
          ORT_FAIL, "ANNC EP: kernel argument rank does not match the site");
    }
    int64_t dims[kMaxRank] = {0, 0, 0, 0};
    if (rank > 0) {
      if (shape != nullptr) {
        std::copy_n(shape, rank, dims);
      } else {
        if (OrtStatus* status = api->GetDimensions(info, dims, rank)) {
          api->ReleaseTensorTypeAndShapeInfo(info);
          api->ReleaseStatus(status);  // replaced by the message below
          return api->CreateStatus(
              ORT_FAIL, "ANNC EP: cannot read a kernel argument shape");
        }
      }
    }
    if (info != nullptr) api->ReleaseTensorTypeAndShapeInfo(info);
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
    case 9:
      return reinterpret_cast<std::int32_t (*)(void*, void*, void*, void*,
                                               void*, void*, void*, void*,
                                               void*)>(symbol)(
          args[0], args[1], args[2], args[3], args[4], args[5], args[6],
          args[7], args[8]);
    case 10:
      return reinterpret_cast<std::int32_t (*)(void*, void*, void*, void*,
                                               void*, void*, void*, void*,
                                               void*, void*)>(symbol)(
          args[0], args[1], args[2], args[3], args[4], args[5], args[6],
          args[7], args[8], args[9]);
    case 11:
      return reinterpret_cast<std::int32_t (*)(void*, void*, void*, void*,
                                               void*, void*, void*, void*,
                                               void*, void*, void*)>(symbol)(
          args[0], args[1], args[2], args[3], args[4], args[5], args[6],
          args[7], args[8], args[9], args[10]);
    case 12:
      return reinterpret_cast<std::int32_t (*)(
          void*, void*, void*, void*, void*, void*, void*, void*, void*,
          void*, void*, void*)>(symbol)(
          args[0], args[1], args[2], args[3], args[4], args[5], args[6],
          args[7], args[8], args[9], args[10], args[11]);
    case 13:
      return reinterpret_cast<std::int32_t (*)(
          void*, void*, void*, void*, void*, void*, void*, void*, void*,
          void*, void*, void*, void*)>(symbol)(
          args[0], args[1], args[2], args[3], args[4], args[5], args[6],
          args[7], args[8], args[9], args[10], args[11], args[12]);
    case 14:
      return reinterpret_cast<std::int32_t (*)(
          void*, void*, void*, void*, void*, void*, void*, void*, void*,
          void*, void*, void*, void*, void*)>(symbol)(
          args[0], args[1], args[2], args[3], args[4], args[5], args[6],
          args[7], args[8], args[9], args[10], args[11], args[12], args[13]);
    default:
      return -1;
  }
}

}  // namespace annc_ep
