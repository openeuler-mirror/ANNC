#include "annc_ep_execution.h"

#include "Context/KernelContextUtils.h"

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
// fused node's inputs and build borrowed tensor views)
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
  size_t bytes = 0;
  annc::kernels::contextByteCount(desc->element_type, desc->rank, desc->dims,
                                 &bytes);
  return bytes;
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
  const int parallelism = std::max(
      1, std::min(num_threads(), options.max_parallelism.value_or(num_threads())));
  if (parallelism == 1 || in_parallel_region()) {
    fn(0, total);
    return;
  }

#if defined(ANNC_EP_NATIVE_POOL)
  if (native_enabled_ && parallelism == num_threads()) {
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
      // TensorOpCost's compute-cycles scale (Context/RowParallel.h), so the
      // number passes through. A null pool (intra-op=1) is serial inside
      // TryParallelFor itself.
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
  // (Context/RowParallel.h).  Eigen's constants
  // (unsupported/Eigen/CXX11/src/Tensor/TensorCostModel.h): kStartupCycles =
  // kPerThreadCycles = 100000.
  //
  // A kernel that reports only grain_size asks for fixed-size blocks instead:
  // that is the TF adapter's kFixedBlockSize path, honoured verbatim.  With
  // neither signal the range is split like the in-tree pool splits an
  // optionless call (context/lib/Parallel.cpp defaults grain_size to 1,
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
          CalculateParallelForBlock(total, cost_per_unit, parallelism);
      batches = DivCeil(total, block);
    }
  } else if (grain > 0) {
    batches = DivCeil(total, grain);
  } else {
    batches = std::min<std::int64_t>(total, parallelism);
  }
  batches =
      std::min<std::int64_t>(std::max<std::int64_t>(batches, 1), parallelism);
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
  output_views_.resize(outputs.size());
  allocated_output_views_.resize(outputs.size());
  services_ = {sizeof(AnncRuntimeServices),
               ANNC_CONTEXT_ABI_VERSION,
               this,
               AllocateOutput,
               AllocateTemp,
               ReportError,
               GetStatus,
               ParallelFor,
               NumThreads,
               Shard,
               NativePool};
  context_.struct_size = sizeof(context_);
  context_.abi_version = ANNC_CONTEXT_ABI_VERSION;
  context_.outputs = output_views_.data();
  context_.num_outputs = static_cast<uint32_t>(output_views_.size());
  context_.services = &services_;
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

AnncStatusCode OnnxExecutionContext::Fail(AnncStatusCode status,
                                          const std::string& message) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (status_.load() == ANNC_STATUS_OK) status_.store(status);
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
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (status_.load() == ANNC_STATUS_OK) status_.store(ANNC_STATUS_INTERNAL);
  if (error_ == nullptr) {
    error_ = status;
    return;
  }
  api_->ReleaseStatus(status);
}

OrtStatus* OnnxExecutionContext::TakeError() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  OrtStatus* error = error_;
  error_ = nullptr;
  return error;
}

bool OnnxExecutionContext::AllOutputsAllocated() const {
  return std::all_of(allocated_.begin(), allocated_.end(),
                     [](bool allocated) { return allocated; });
}

bool OnnxExecutionContext::AllOutputsBound() const {
  for (size_t i = 0; i < output_views_.size(); ++i) {
    const auto& view = output_views_[i];
    const auto& allocated = allocated_output_views_[i];
    if (!allocated_[i] || !(view.flags & ANNC_TENSOR_BOUND) ||
        (view.flags & ANNC_TENSOR_READ_ONLY) ||
        view.allocated != allocated.allocated ||
        view.aligned != allocated.aligned || view.offset != allocated.offset ||
        view.element_type != allocated.element_type ||
        view.rank != allocated.rank || view.sizes != allocated.sizes ||
        view.strides != allocated.strides || view.layout != allocated.layout)
      return false;
  }
  return true;
}

bool OnnxExecutionContext::BindView(const AnncTensorDesc& request, void* data,
                                    AnncTensorView* result) {
  shapes_.emplace_back();
  auto& shape = shapes_.back();
  int64_t stride = 1;
  for (uint32_t i = request.rank; i > 0; --i) {
    shape.sizes[i - 1] = request.dims[i - 1];
    shape.strides[i - 1] = stride;
    if (request.dims[i - 1] &&
        stride > INT64_MAX / request.dims[i - 1]) return false;
    stride *= request.dims[i - 1];
  }
  *result = {data, data, 0, shape.sizes.data(), shape.strides.data(),
             request.element_type, request.rank, ANNC_LAYOUT_STRIDED,
             ANNC_TENSOR_BOUND};
  return true;
}

AnncStatusCode OnnxExecutionContext::AllocateOutput(
    void* opaque, uint32_t slot, const AnncTensorDesc* request,
    AnncTensorView* result) {
  auto& state = *static_cast<OnnxExecutionContext*>(opaque);
  std::lock_guard<std::recursive_mutex> lock(state.mutex_);
  size_t bytes = 0;
  if (!request || request->struct_size < sizeof(*request) || !result ||
      (request->flags & ~ANNC_ALLOCATION_ZERO) ||
      !annc::kernels::contextByteCount(request->element_type, request->rank,
                                      request->dims, &bytes))
    return state.Fail(ANNC_STATUS_INVALID_ARGUMENT, "invalid output request");
  try {
    AnncTensorDesc desc = *request;
    auto code = state.AllocateOutputImpl(slot, &desc);
    if (code != ANNC_STATUS_OK) return code;
    if (!state.BindView(desc, desc.data, result))
      return state.Fail(ANNC_STATUS_INVALID_ARGUMENT, "output stride overflow");
    state.allocated_output_views_[slot] = *result;
    if (bytes && (desc.flags & ANNC_ALLOCATION_ZERO))
      std::memset(desc.data, 0, bytes);
    return ANNC_STATUS_OK;
  } catch (...) {
    return state.Fail(ANNC_STATUS_ALLOCATION_FAILED, "output allocation failed");
  }
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
  // then hands back a tensor whose data pointer can be null. A rank-zero
  // scalar needs no dims pointer and describes one element.
  if (desc->rank && desc->dims == nullptr) {
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
  if (DescribedBytes(desc) && !data)
    return Fail(ANNC_STATUS_ALLOCATION_FAILED, "output buffer is null");
  desc->data = data;
  allocated_[slot] = true;
  return ANNC_STATUS_OK;
}

AnncStatusCode OnnxExecutionContext::AllocateTemp(
    void* opaque, const AnncTensorDesc* request, AnncTensorView* result) {
  auto& state = *static_cast<OnnxExecutionContext*>(opaque);
  std::lock_guard<std::recursive_mutex> lock(state.mutex_);
  size_t bytes;
  if (!request || request->struct_size < sizeof(*request) || !result ||
      (request->flags & ~ANNC_ALLOCATION_ZERO) ||
      !annc::kernels::contextByteCount(request->element_type, request->rank,
                                      request->dims, &bytes))
    return state.Fail(ANNC_STATUS_INVALID_ARGUMENT, "invalid temporary request");
  try {
    AnncTensorDesc desc = *request;
    auto code = state.AllocateTempImpl(&desc);
    if (code != ANNC_STATUS_OK) return code;
    if (!state.BindView(desc, desc.data, result))
      return state.Fail(ANNC_STATUS_INVALID_ARGUMENT, "temporary stride overflow");
    return ANNC_STATUS_OK;
  } catch (...) {
    return state.Fail(ANNC_STATUS_ALLOCATION_FAILED, "temporary allocation failed");
  }
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
  if (desc->rank && desc->dims == nullptr) {
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
  if (!data) return Fail(ANNC_STATUS_ALLOCATION_FAILED, "allocator returned null");
  if (desc->flags & ANNC_ALLOCATION_ZERO) std::memset(data, 0, bytes);
  try {
    temps_.push_back(data);
  } catch (...) {
    if (auto* error = api_->AllocatorFree(temp_allocator_, data))
      api_->ReleaseStatus(error);
    throw;
  }
  desc->data = data;
  return ANNC_STATUS_OK;
}

void OnnxExecutionContext::ReportError(void* opaque, AnncStatusCode status,
                                       const char* message) {
  if (status != ANNC_STATUS_OK)
    static_cast<OnnxExecutionContext*>(opaque)->Fail(
        status, message ? message : "unspecified kernel error");
}

AnncStatusCode OnnxExecutionContext::GetStatus(void* opaque) {
  return static_cast<OnnxExecutionContext*>(opaque)->status_.load();
}

int32_t OnnxExecutionContext::NumThreads(void* opaque) {
  return static_cast<OnnxExecutionContext*>(opaque)->pool_.num_threads();
}

void* OnnxExecutionContext::NativePool(void* opaque, const char* provider) {
  if (!provider || std::strcmp(provider, "onnxruntime") != 0) return nullptr;
  return static_cast<OnnxExecutionContext*>(opaque)->pool_.native_pool();
}

AnncStatusCode OnnxExecutionContext::Dispatch(
    int64_t total, const annc::threadpool::ParallelForOptions& options,
    AnncContextTask task, void* task_data) {
  if (total < 0 || !task)
    return Fail(ANNC_STATUS_INVALID_ARGUMENT, "invalid parallel request");
  if (!total) return status_.load();
  try {
    if (pool_.in_parallel_region() || options.max_parallelism.value_or(0) == 1)
      task(task_data, 0, total);
    else
      pool_.parallel_for(total, options, [&](int64_t first, int64_t last) {
        try { task(task_data, first, last); }
        catch (...) { Fail(ANNC_STATUS_INTERNAL, "parallel task raised an exception"); }
      });
  } catch (...) {
    return Fail(ANNC_STATUS_INTERNAL, "parallel dispatch raised an exception");
  }
  return status_.load();
}

AnncStatusCode OnnxExecutionContext::ParallelFor(
    void* opaque, int64_t total, int64_t grain, AnncContextTask task,
    void* task_data) {
  auto& state = *static_cast<OnnxExecutionContext*>(opaque);
  if (grain < 0) return state.Fail(ANNC_STATUS_INVALID_ARGUMENT, "negative grain");
  annc::threadpool::ParallelForOptions options;
  options.grain_size = grain;
  return state.Dispatch(total, options, task, task_data);
}

AnncStatusCode OnnxExecutionContext::Shard(
    void* opaque, int32_t max_parallelism, int64_t total, int64_t cost,
    AnncContextTask task, void* task_data) {
  auto& state = *static_cast<OnnxExecutionContext*>(opaque);
  if (max_parallelism < 1 || cost < 0)
    return state.Fail(ANNC_STATUS_INVALID_ARGUMENT, "invalid shard options");
  annc::threadpool::ParallelForOptions options;
  options.max_parallelism = max_parallelism;
  options.cost_per_unit = cost;
  return state.Dispatch(total, options, task, task_data);
}

OrtStatus* ComputeWithContext(const OrtApi* api,
                              OrtKernelContext* kernel_context,
                              const ContextCall& request) {
  const bool timing = TimingEnabled();
  const auto t_start = timing ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
  OnnxExecutionContext context(api, kernel_context, request.outputs,
                               &request.output_slots);

  if (!request.symbol || request.arg_indices.size() != request.arg_ranks.size() ||
      request.arg_types.size() != request.arg_ranks.size())
    return api->CreateStatus(ORT_FAIL, "ANNC EP: invalid context call contract");
  size_t input_count = 0;
  if (OrtStatus* status = api->KernelContext_GetInputCount(kernel_context, &input_count))
    return status;
  constexpr size_t kMaxRank = ANNC_CONTEXT_MAX_RANK;
  struct InputStorage { std::array<int64_t, kMaxRank> sizes{}, strides{}; };
  std::vector<InputStorage> storage(request.arg_indices.size());
  std::vector<AnncTensorView> inputs(storage.size());
  std::vector<AnncAttributeView> attrs;
  for (const auto& value : request.i64_attrs)
    attrs.push_back({ANNC_ATTRIBUTE_I64, 0, &value, 1});
  context.abi()->inputs = inputs.data();
  context.abi()->num_inputs = static_cast<uint32_t>(inputs.size());
  context.abi()->attrs = attrs.data();
  context.abi()->num_attrs = static_cast<uint32_t>(attrs.size());
  context.abi()->resources = request.resources.data();
  context.abi()->num_resources =
      static_cast<uint32_t>(request.resources.size());

  for (size_t i = 0; i < request.arg_indices.size(); ++i) {
    if (request.arg_indices[i] < 0 ||
        static_cast<size_t>(request.arg_indices[i]) >= input_count)
      return api->CreateStatus(ORT_FAIL, "ANNC EP: invalid context input index");
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
    auto* dims = storage[i].sizes.data();
    if (rank > 0) {
      if (shape != nullptr) {
        std::copy_n(shape, rank, dims);
      } else {
        if (!info)
          return api->CreateStatus(
              ORT_FAIL, "ANNC EP: kernel argument has no shape data");
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
    size_t bytes;
    if (!annc::kernels::contextByteCount(element, rank, dims, &bytes) ||
        (bytes && !raw))
      return api->CreateStatus(ORT_FAIL, "ANNC EP: invalid input tensor shape or data");
    int64_t stride = 1;
    for (size_t axis = rank; axis > 0; --axis) {
      storage[i].strides[axis - 1] = stride;
      if (dims[axis - 1] && stride > INT64_MAX / dims[axis - 1])
        return api->CreateStatus(ORT_FAIL, "ANNC EP: input stride overflow");
      stride *= dims[axis - 1];
    }
    inputs[i] = {const_cast<void*>(raw), const_cast<void*>(raw), 0, dims,
                 storage[i].strides.data(), element, static_cast<uint32_t>(rank),
                 ANNC_LAYOUT_STRIDED, ANNC_TENSOR_BOUND | ANNC_TENSOR_READ_ONLY};
  }

  const auto t_prepared = timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
  const AnncStatusCode status_code =
      reinterpret_cast<AnncKernelFn>(request.symbol)(context.abi());
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
                  "ANNC EP: kernel returned status %u", status_code);
    return api->CreateStatus(ORT_FAIL, message);
  }
  if (!context.AllOutputsAllocated()) {
    return api->CreateStatus(
        ORT_FAIL, "ANNC EP: kernel did not allocate every declared output");
  }
  if (!context.AllOutputsBound()) {
    return api->CreateStatus(ORT_FAIL,
                             "ANNC EP: kernel did not bind every output to its "
                             "ONNX Runtime allocation");
  }
  return nullptr;
}

}  // namespace annc_ep
