#include "annc_fused_op.h"
#include "annc_fused_op_jit.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <iostream>
#include <string>

#include "Kernel/ExecutionContext.h"
#include "Kernel/MemRefTypes.h"
#include "Support/ThreadPool/ThreadPool.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/threadpool.h"

struct AnncExecutionHandle {
  void* user;
};

namespace tensorflow {

struct AnncFusedProfileSample {
  double load_library_us = 0.0;
  double backend_dispatch_us = 0.0;
  double threadpool_setup_us = 0.0;
  double threadpool_restore_us = 0.0;
  double input_memref_us = 0.0;
  double output_alloc_us = 0.0;
  double output_init_us = 0.0;
  double output_memref_us = 0.0;
  double arg_order_us = 0.0;
  double execution_setup_us = 0.0;
  double execution_post_us = 0.0;
  double kernel_us = 0.0;
  double execution_callback_us = 0.0;
  double tensorflow_output_alloc_us = 0.0;
  double tensorflow_temp_alloc_us = 0.0;
  int execution_output_callbacks = 0;
  int execution_temp_callbacks = 0;
};

namespace {

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

double ElapsedUs(TimePoint start, TimePoint end = Clock::now()) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}

struct TensorFlowExecutionState;

struct TensorFlowExecutionState {
  OpKernelContext* context;
  const std::vector<DataType>* output_types;
  const std::vector<int>* output_ranks;
  Status error;
  std::vector<Tensor*> outputs;
  std::vector<std::unique_ptr<Tensor>> temps;
  bool profile_enabled;
  AnncFusedProfileSample* profile_sample;
};

TensorFlowExecutionState* GetExecutionState(AnncExecutionHandle* handle) {
  return handle ? static_cast<TensorFlowExecutionState*>(handle->user)
                : nullptr;
}

AnncStatusCode RecordExecutionError(TensorFlowExecutionState* state,
                                    AnncStatusCode status,
                                    const std::string& message) {
  if (state && state->error.ok()) {
    state->error = errors::InvalidArgument(message);
  }
  return status;
}

AnncElementType ToAnncElementType(DataType type) {
  switch (type) {
    case DT_FLOAT:
      return ANNC_ELEMENT_TYPE_F32;
    case DT_INT32:
      return ANNC_ELEMENT_TYPE_I32;
    case DT_INT64:
      return ANNC_ELEMENT_TYPE_I64;
    default:
      return static_cast<AnncElementType>(0);
  }
}

DataType ToTensorFlowDataType(uint32_t type) {
  switch (type) {
    case ANNC_ELEMENT_TYPE_F32:
      return DT_FLOAT;
    case ANNC_ELEMENT_TYPE_I32:
      return DT_INT32;
    case ANNC_ELEMENT_TYPE_I64:
      return DT_INT64;
    default:
      return DT_INVALID;
  }
}

bool FillTensorShape(const AnncTensorDesc* desc, TensorShape* shape) {
  if (!desc || !shape || desc->rank > 8 || (desc->rank && !desc->dims)) {
    return false;
  }
  shape->Clear();
  for (uint32_t i = 0; i < desc->rank; ++i) {
    if (desc->dims[i] < 0) return false;
    shape->AddDim(desc->dims[i]);
  }
  return true;
}

AnncStatusCode AllocateExecutionOutput(AnncExecutionHandle* handle,
                                       uint32_t slot, AnncTensorDesc* desc) {
  auto* state = GetExecutionState(handle);
  if (!state || !desc || desc->struct_size < sizeof(*desc))
    return ANNC_STATUS_INVALID_ARGUMENT;
  const bool profile_enabled = state->profile_enabled && state->profile_sample;
  auto callback_start = profile_enabled ? Clock::now() : TimePoint{};
  if (slot >= state->output_types->size() ||
      slot >= state->output_ranks->size()) {
    return RecordExecutionError(
        state, ANNC_STATUS_INVALID_ARGUMENT,
        "ANNC execution output slot " + std::to_string(slot) +
            " is out of range");
  }
  if (ToAnncElementType((*state->output_types)[slot]) != desc->element_type) {
    return RecordExecutionError(
        state, ANNC_STATUS_INVALID_ARGUMENT,
        "ANNC execution output slot " + std::to_string(slot) +
            " dtype mismatch");
  }
  if ((*state->output_ranks)[slot] != static_cast<int>(desc->rank)) {
    return RecordExecutionError(
        state, ANNC_STATUS_INVALID_ARGUMENT,
        "ANNC execution output slot " + std::to_string(slot) +
            " rank mismatch");
  }
  if (slot < state->outputs.size() && state->outputs[slot] != nullptr) {
    return RecordExecutionError(
        state, ANNC_STATUS_INVALID_ARGUMENT,
        "ANNC execution output slot " + std::to_string(slot) +
            " was allocated more than once");
  }
  TensorShape shape;
  if (!FillTensorShape(desc, &shape)) {
    return RecordExecutionError(
        state, ANNC_STATUS_INVALID_ARGUMENT,
        "ANNC execution output slot " + std::to_string(slot) +
            " has invalid dimensions");
  }
  Tensor* output = nullptr;
  auto allocation_start = profile_enabled ? Clock::now() : TimePoint{};
  Status status = state->context->allocate_output(slot, shape, &output);
  if (profile_enabled) {
    state->profile_sample->tensorflow_output_alloc_us +=
        ElapsedUs(allocation_start);
  }
  if (!status.ok()) {
    state->error = status;
    return ANNC_STATUS_ALLOCATION_FAILED;
  }
  if (state->outputs.size() <= slot) state->outputs.resize(slot + 1, nullptr);
  state->outputs[slot] = output;
  desc->data = const_cast<char*>(output->tensor_data().data());
  if (profile_enabled) {
    ++state->profile_sample->execution_output_callbacks;
    state->profile_sample->execution_callback_us +=
        ElapsedUs(callback_start);
  }
  return ANNC_STATUS_OK;
}

AnncStatusCode AllocateExecutionTemp(AnncExecutionHandle* handle,
                                     AnncTensorDesc* desc) {
  auto* state = GetExecutionState(handle);
  if (!state || !desc || desc->struct_size < sizeof(*desc))
    return ANNC_STATUS_INVALID_ARGUMENT;
  const bool profile_enabled = state->profile_enabled && state->profile_sample;
  auto callback_start = profile_enabled ? Clock::now() : TimePoint{};
  TensorShape shape;
  if (!FillTensorShape(desc, &shape)) return ANNC_STATUS_INVALID_ARGUMENT;
  DataType dtype = ToTensorFlowDataType(desc->element_type);
  if (dtype == DT_INVALID) return ANNC_STATUS_INVALID_ARGUMENT;
  auto temp = std::make_unique<Tensor>();
  auto allocation_start = profile_enabled ? Clock::now() : TimePoint{};
  Status status = state->context->allocate_temp(dtype, shape, temp.get());
  if (profile_enabled) {
    state->profile_sample->tensorflow_temp_alloc_us +=
        ElapsedUs(allocation_start);
  }
  if (!status.ok()) {
    state->error = status;
    return ANNC_STATUS_ALLOCATION_FAILED;
  }
  desc->data = const_cast<char*>(temp->tensor_data().data());
  state->temps.push_back(std::move(temp));
  if (profile_enabled) {
    ++state->profile_sample->execution_temp_callbacks;
    state->profile_sample->execution_callback_us +=
        ElapsedUs(callback_start);
  }
  return ANNC_STATUS_OK;
}

void ReportExecutionError(AnncExecutionHandle* handle, AnncStatusCode status,
                          const char* message) {
  auto* state = GetExecutionState(handle);
  if (!state) return;
  if (state->error.ok()) {
    state->error =
        errors::Internal("ANNC execution error ", static_cast<int>(status),
                         ": ", message ? message : "");
  }
}

size_t JitCacheCapacity() {
  constexpr size_t kDefaultCapacity = 64;
  const char* configured = std::getenv("ANNC_JIT_CACHE_MAX_ENTRIES");
  if (!configured || configured[0] == '\0') return kDefaultCapacity;
  errno = 0;
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(configured, &end, 10);
  if (errno != 0 || end == configured || *end != '\0' ||
      parsed > std::numeric_limits<size_t>::max()) {
    LOG(WARNING) << "[ANNC-JIT-CACHE] invalid ANNC_JIT_CACHE_MAX_ENTRIES="
                 << configured << ", using " << kDefaultCapacity;
    return kDefaultCapacity;
  }
  return static_cast<size_t>(parsed);
}

annc::jit::JitCompilationCache& GlobalJitCompilationCache() {
  static auto* cache = new annc::jit::JitCompilationCache(JitCacheCapacity());
  return *cache;
}

const char* CacheEventName(annc::jit::CacheEvent event) {
  switch (event) {
    case annc::jit::CacheEvent::kMiss:
      return "miss";
    case annc::jit::CacheEvent::kWait:
      return "wait";
    case annc::jit::CacheEvent::kHit:
      return "hit";
  }
  return "unknown";
}

bool IsAnncFusedProfilingEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("ANNC_FUSED_PROFILE");
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

int AnncFusedProfileInterval() {
  static const int interval = [] {
    const char* value = std::getenv("ANNC_FUSED_PROFILE_INTERVAL");
    if (!value || value[0] == '\0') return 100000;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) return 100000;
    return static_cast<int>(parsed);
  }();
  return interval;
}

struct ProfileStats {
  int count = 0;
  double load_library_us = 0.0;
  double backend_dispatch_us = 0.0;
  double threadpool_setup_us = 0.0;
  double threadpool_restore_us = 0.0;
  double input_memref_us = 0.0;
  double output_alloc_us = 0.0;
  double output_init_us = 0.0;
  double output_memref_us = 0.0;
  double arg_order_us = 0.0;
  double execution_setup_us = 0.0;
  double execution_post_us = 0.0;
  double kernel_us = 0.0;
  double execution_callback_us = 0.0;
  double tensorflow_output_alloc_us = 0.0;
  double tensorflow_temp_alloc_us = 0.0;
  int execution_output_callbacks = 0;
  int execution_temp_callbacks = 0;
  double total_us = 0.0;
};

mutex profile_stats_mu_(LINKER_INITIALIZED);

template <typename T>
std::string FormatVector(const std::vector<T>& values) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) os << ",";
    os << values[i];
  }
  os << "]";
  return os.str();
}

void LogProfileStats(const char* tag, int total,
                     const std::map<std::string, ProfileStats>& stats) {
  LOG(INFO) << "[" << tag << "] ====== report (calls=" << total << ") ======";
  for (const auto& [key, s] : stats) {
    const double count = static_cast<double>(s.count);
    LOG(INFO) << "[" << tag << "] key=" << key << " count=" << s.count
              << " avg_load_library=" << (s.load_library_us / count) << " us"
              << " avg_backend_dispatch="
              << (s.backend_dispatch_us / count) << " us"
              << " avg_threadpool_setup=" << (s.threadpool_setup_us / count)
              << " us"
              << " avg_threadpool_restore="
              << (s.threadpool_restore_us / count) << " us"
              << " avg_input_memref=" << (s.input_memref_us / count) << " us"
              << " avg_output_alloc=" << (s.output_alloc_us / count) << " us"
              << " avg_output_init=" << (s.output_init_us / count) << " us"
              << " avg_output_memref=" << (s.output_memref_us / count) << " us"
              << " avg_arg_order=" << (s.arg_order_us / count) << " us"
              << " avg_execution_setup=" << (s.execution_setup_us / count)
              << " us"
              << " avg_execution_post=" << (s.execution_post_us / count)
              << " us"
              << " avg_kernel=" << (s.kernel_us / count) << " us"
              << " avg_execution_callbacks="
              << (s.execution_callback_us / count) << " us"
              << " avg_tf_output_alloc="
              << (s.tensorflow_output_alloc_us / count) << " us"
              << " avg_tf_temp_alloc="
              << (s.tensorflow_temp_alloc_us / count) << " us"
              << " avg_callback_framework="
              << ((s.execution_callback_us -
                   s.tensorflow_output_alloc_us -
                   s.tensorflow_temp_alloc_us) /
                  count)
              << " us"
              << " avg_kernel_excluding_callbacks="
              << ((s.kernel_us - s.execution_callback_us) / count) << " us"
              << " avg_output_callbacks="
              << (static_cast<double>(s.execution_output_callbacks) / count)
              << " avg_temp_callbacks="
              << (static_cast<double>(s.execution_temp_callbacks) / count)
              << " avg_total=" << (s.total_us / count) << " us"
              << " avg_overhead=" << ((s.total_us - s.kernel_us) / count)
              << " us";
  }
}

std::string BuildProfileKey(OpKernelContext* context, int total_inputs,
                            const std::vector<int>& input_ranks,
                            const std::vector<std::string>& output_shapes,
                            const std::string& kernel_name,
                            const std::string& fusion_pattern) {
  auto append_tensor_shape = [](std::ostringstream& os, const Tensor& tensor) {
    os << tensor.dims() << "D[";
    for (int d = 0; d < tensor.dims(); ++d) {
      if (d > 0) os << "x";
      os << tensor.dim_size(d);
    }
    os << "]";
  };

  auto append_output_shapes = [&output_shapes](std::ostringstream& os) {
    if (output_shapes.empty()) return;
    os << "_out=";
    for (size_t i = 0; i < output_shapes.size(); ++i) {
      if (i > 0) os << ",";
      os << output_shapes[i];
    }
  };

  if (fusion_pattern == "dnn_embedding_hash_bucket") {
    std::ostringstream key;
    key << kernel_name << "_pattern=" << fusion_pattern;
    if (total_inputs > 0) {
      key << "_weight=";
      append_tensor_shape(key, context->input(0));
    }
    if (total_inputs > 1) {
      key << "_input=";
      append_tensor_shape(key, context->input(1));
    }
    append_output_shapes(key);
    return key.str();
  }

  int lhs = -1;
  int rhs = -1;
  for (int i = 0; i < total_inputs; ++i) {
    int rank = input_ranks.empty() ? context->input(i).dims() : input_ranks[i];
    if (rank == 2) {
      if (lhs < 0) {
        lhs = i;
      } else if (rhs < 0) {
        rhs = i;
      }
    }
  }

  std::string key = kernel_name;
  const bool use_matmul_profile =
      fusion_pattern.empty() ||
      fusion_pattern.find("matmul") != std::string::npos ||
      kernel_name.find("matmul") != std::string::npos;
  if (use_matmul_profile && lhs >= 0 && rhs >= 0) {
    const Tensor& t0 = context->input(lhs);
    const Tensor& t1 = context->input(rhs);
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    if (t0.dim_size(1) == t1.dim_size(0)) {
      m = t0.dim_size(0);
      k = t0.dim_size(1);
      n = t1.dim_size(1);
    } else {
      m = t1.dim_size(0);
      k = t1.dim_size(1);
      n = t0.dim_size(1);
    }
    key += "_" + std::to_string(m) + "x" + std::to_string(k) + "x" +
           std::to_string(n);
    if (!fusion_pattern.empty()) {
      key += "_pattern=" + fusion_pattern;
    }
    return key;
  }

  std::ostringstream generic_key;
  generic_key << kernel_name;
  if (!fusion_pattern.empty()) {
    generic_key << "_pattern=" << fusion_pattern;
  }
  generic_key << "_inputs=";
  for (int i = 0; i < total_inputs; ++i) {
    if (i > 0) generic_key << ",";
    append_tensor_shape(generic_key, context->input(i));
  }
  append_output_shapes(generic_key);
  return generic_key.str();
}

class TensorFlowAnncThreadPool final : public annc::threadpool::AnncThreadPool {
 public:
  explicit TensorFlowAnncThreadPool(thread::ThreadPool* workers)
      : workers_(workers) {}

  int num_threads() const override {
    return workers_ ? workers_->NumThreads() : 1;
  }

  bool in_parallel_region() const override {
    return workers_ && workers_->CurrentThreadId() >= 0;
  }

  void parallel_for(int64_t total,
                    const annc::threadpool::ParallelForOptions& options,
                    const std::function<void(int64_t, int64_t)>& fn) override {
    if (!workers_ || total <= 0) {
      if (total > 0) fn(0, total);
      return;
    }

    if (options.grain_size.has_value()) {
      workers_->ParallelFor(
          total,
          thread::ThreadPool::SchedulingParams(
              thread::ThreadPool::SchedulingStrategy::kFixedBlockSize,
              std::nullopt, std::max<int64_t>(*options.grain_size, 1)),
          fn);
      return;
    }

    const int64_t cost_per_unit =
        std::max<int64_t>(options.cost_per_unit.value_or(1), 1);
    workers_->ParallelFor(total,
                          thread::ThreadPool::SchedulingParams(
                              thread::ThreadPool::SchedulingStrategy::kAdaptive,
                              cost_per_unit, std::nullopt),
                          fn);
  }

 private:
  thread::ThreadPool* workers_;
};

class ScopedAnncThreadPool final {
 public:
  ScopedAnncThreadPool(
      annc::threadpool::AnncThreadPool* thread_pool,
      void (*set_current_threadpool)(annc::threadpool::AnncThreadPool*),
      annc::threadpool::AnncThreadPool* (*get_current_threadpool)())
      : set_current_threadpool_(set_current_threadpool),
        previous_(get_current_threadpool ? get_current_threadpool() : nullptr),
        active_(set_current_threadpool != nullptr) {
    if (set_current_threadpool_) {
      set_current_threadpool_(thread_pool);
    }
  }

  ~ScopedAnncThreadPool() { Restore(); }

  void Restore() {
    if (active_) {
      set_current_threadpool_(previous_);
      active_ = false;
    }
  }

  ScopedAnncThreadPool(const ScopedAnncThreadPool&) = delete;
  ScopedAnncThreadPool& operator=(const ScopedAnncThreadPool&) = delete;

 private:
  void (*set_current_threadpool_)(annc::threadpool::AnncThreadPool*);
  annc::threadpool::AnncThreadPool* previous_;
  bool active_;
};

thread::ThreadPool* GetTensorFlowCpuThreadPool(OpKernelContext* context) {
  if (!context || !context->device()) {
    return nullptr;
  }

  const DeviceBase::CpuWorkerThreads* worker_threads = nullptr;
  // DeviceBase::tensorflow_cpu_worker_threads() CHECKs when no CPU worker
  // threads were installed, so only call it for CPU devices where ANNCFused is
  // registered today.
  worker_threads = context->device()->tensorflow_cpu_worker_threads();
  return worker_threads ? worker_threads->workers : nullptr;
}

template <int Rank>
struct RankedMemRefDescriptor {
  void* allocated;
  void* aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];
};

template <>
struct RankedMemRefDescriptor<0> {
  void* allocated;
  void* aligned;
  int64_t offset;
};

template <int Rank>
Status CreateRankedMemRef(
    const Tensor& tensor, RankedMemRefDescriptor<Rank>* ref,
    std::vector<AnncStringRef>* string_storage = nullptr) {
  if (tensor.dims() != Rank) {
    return errors::InvalidArgument("Expected rank ", Rank, ", got ",
                                   tensor.dims());
  }

  if (tensor.dtype() == DT_STRING) {
    if (!string_storage) {
      return errors::InvalidArgument("String tensor memref requires storage");
    }
    auto flat = tensor.flat<tstring>();
    string_storage->resize(flat.size());
    for (int64_t i = 0; i < flat.size(); ++i) {
      const tstring& value = flat(i);
      (*string_storage)[i] =
          AnncStringRef{value.data(), static_cast<int64_t>(value.size())};
    }
    ref->allocated = string_storage->data();
  } else {
    ref->allocated = const_cast<void*>(
        static_cast<const void*>(tensor.tensor_data().data()));
  }
  ref->aligned = ref->allocated;
  ref->offset = 0;

  int64_t stride = 1;
  for (int i = Rank - 1; i >= 0; --i) {
    ref->sizes[i] = tensor.dim_size(i);
    ref->strides[i] = stride;
    stride *= ref->sizes[i];
  }

  return OkStatus();
}

Status CreateRankedMemRef(
    const Tensor& tensor, RankedMemRefDescriptor<0>* ref,
    std::vector<AnncStringRef>* string_storage = nullptr) {
  if (tensor.dims() != 0) {
    return errors::InvalidArgument("Expected rank 0, got ", tensor.dims());
  }
  if (tensor.dtype() == DT_STRING) {
    if (!string_storage) {
      return errors::InvalidArgument("String tensor memref requires storage");
    }
    auto scalar = tensor.scalar<tstring>();
    string_storage->resize(1);
    const tstring& value = scalar();
    (*string_storage)[0] =
        AnncStringRef{value.data(), static_cast<int64_t>(value.size())};
    ref->allocated = string_storage->data();
    ref->aligned = ref->allocated;
    ref->offset = 0;
    return OkStatus();
  }

  ref->allocated =
      const_cast<void*>(static_cast<const void*>(tensor.tensor_data().data()));
  ref->aligned = ref->allocated;
  ref->offset = 0;
  return OkStatus();
}

Status BuildRankedMemRefArg(
    const Tensor& tensor, int expected_rank,
    std::array<RankedMemRefDescriptor<0>, 1>* rank0,
    std::array<RankedMemRefDescriptor<1>, 1>* rank1,
    std::array<RankedMemRefDescriptor<2>, 1>* rank2,
    std::array<RankedMemRefDescriptor<3>, 1>* rank3,
    std::array<RankedMemRefDescriptor<4>, 1>* rank4,
    std::array<RankedMemRefDescriptor<5>, 1>* rank5,
    std::array<RankedMemRefDescriptor<6>, 1>* rank6, void** arg,
    std::vector<AnncStringRef>* string_storage = nullptr) {
  switch (expected_rank) {
    case 0:
      TF_RETURN_IF_ERROR(
          CreateRankedMemRef(tensor, &(*rank0)[0], string_storage));
      *arg = &(*rank0)[0];
      return OkStatus();
    case 1:
      TF_RETURN_IF_ERROR(
          CreateRankedMemRef(tensor, &(*rank1)[0], string_storage));
      *arg = &(*rank1)[0];
      return OkStatus();
    case 2:
      TF_RETURN_IF_ERROR(
          CreateRankedMemRef(tensor, &(*rank2)[0], string_storage));
      *arg = &(*rank2)[0];
      return OkStatus();
    case 3:
      TF_RETURN_IF_ERROR(
          CreateRankedMemRef(tensor, &(*rank3)[0], string_storage));
      *arg = &(*rank3)[0];
      return OkStatus();
    case 4:
      TF_RETURN_IF_ERROR(
          CreateRankedMemRef(tensor, &(*rank4)[0], string_storage));
      *arg = &(*rank4)[0];
      return OkStatus();
    case 5:
      TF_RETURN_IF_ERROR(
          CreateRankedMemRef(tensor, &(*rank5)[0], string_storage));
      *arg = &(*rank5)[0];
      return OkStatus();
    case 6:
      TF_RETURN_IF_ERROR(
          CreateRankedMemRef(tensor, &(*rank6)[0], string_storage));
      *arg = &(*rank6)[0];
      return OkStatus();
    default:
      return errors::InvalidArgument("mlir_ciface supports rank 0-6, got ",
                                     expected_rank);
  }
}

std::vector<int64_t> ParseShapeSpec(const std::string& spec) {
  std::vector<int64_t> dims;
  std::stringstream ss(spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty() || item == "?") {
      dims.push_back(-1);
    } else {
      dims.push_back(std::stoll(item));
    }
  }
  return dims;
}

TensorShape InferOutputShape(OpKernelContext* context, int output_index,
                             int output_rank,
                             const std::vector<std::string>& output_shapes,
                             int num_constants, int num_fixed) {
  if (output_index < static_cast<int>(output_shapes.size()) &&
      !output_shapes[output_index].empty()) {
    std::vector<int64_t> dims = ParseShapeSpec(output_shapes[output_index]);
    TensorShape shape;
    for (int i = 0; i < static_cast<int>(dims.size()); ++i) {
      int64_t dim = dims[i];
      if (dim < 0 && context->num_inputs() > num_constants + num_fixed) {
        const Tensor& dynamic_input = context->input(num_constants + num_fixed);
        if (i < dynamic_input.dims()) {
          dim = dynamic_input.dim_size(i);
        }
      }
      shape.AddDim(dim < 0 ? 1 : dim);
    }
    if (shape.dims() == output_rank) {
      return shape;
    }
  }

  if (context->num_inputs() > num_constants + num_fixed) {
    TensorShape shape = context->input(num_constants + num_fixed).shape();
    if (shape.dims() == output_rank) {
      if (num_fixed > 0 && output_rank > 0) {
        const Tensor& fixed = context->input(num_constants);
        if (fixed.dims() > 0) {
          shape.set_dim(output_rank - 1, fixed.dim_size(fixed.dims() - 1));
        }
      }
      return shape;
    }
  }

  if (context->num_inputs() > num_constants && num_fixed > 0) {
    TensorShape shape = context->input(num_constants).shape();
    if (shape.dims() == output_rank) {
      return shape;
    }
  }

  TensorShape shape;
  for (int i = 0; i < output_rank; ++i) {
    shape.AddDim(1);
  }
  return shape;
}

std::vector<int> BuildDefaultKernelArgOrder(int num_inputs, int num_outputs) {
  std::vector<int> order;
  order.reserve(num_inputs + num_outputs);
  for (int i = 0; i < num_inputs; ++i) {
    order.push_back(i);
  }
  for (int i = 0; i < num_outputs; ++i) {
    order.push_back(num_inputs + i);
  }
  return order;
}

std::vector<int> ResolveKernelArgOrder(const std::vector<int>& configured_order,
                                       int num_inputs, int num_outputs,
                                       bool has_input_ranks) {
  if (!configured_order.empty()) return configured_order;
  if (num_inputs == 3 && num_outputs == 1 && !has_input_ranks) {
    return {2, 0, 3, 1};
  }
  return BuildDefaultKernelArgOrder(num_inputs, num_outputs);
}

std::vector<int64_t> ShapeVector(const TensorShape& shape) {
  std::vector<int64_t> dims;
  dims.reserve(shape.dims());
  for (int i = 0; i < shape.dims(); ++i) dims.push_back(shape.dim_size(i));
  return dims;
}

Status BuildRuntimeArgumentShapes(
    OpKernelContext* context, int num_constants, int num_fixed, int num_dynamic,
    int num_outputs, const std::vector<int>& input_ranks,
    const std::vector<int>& output_ranks,
    const std::vector<std::string>& output_shapes,
    const std::vector<int>& configured_order,
    std::vector<annc::jit::JitArgumentSignature>* arguments) {
  const int num_inputs = num_constants + num_fixed + num_dynamic;
  if (context->num_inputs() != num_inputs) {
    return errors::InvalidArgument("ANNCFused expected ", num_inputs,
                                   " inputs, got ", context->num_inputs());
  }

  std::vector<annc::jit::JitArgumentSignature> tensors;
  tensors.reserve(num_inputs + num_outputs);
  for (int i = 0; i < num_inputs; ++i) {
    const Tensor& tensor = context->input(i);
    if (!input_ranks.empty() && tensor.dims() != input_ranks[i]) {
      return errors::InvalidArgument("ANNCFused input ", i, " expected rank ",
                                     input_ranks[i], ", got ", tensor.dims());
    }
    tensors.push_back({ShapeVector(tensor.shape())});
  }

  for (int i = 0; i < num_outputs; ++i) {
    TensorShape shape = InferOutputShape(
        context, i, output_ranks[i], output_shapes, num_constants, num_fixed);
    tensors.push_back({ShapeVector(shape)});
  }

  std::vector<int> order = ResolveKernelArgOrder(
      configured_order, num_inputs, num_outputs, !input_ranks.empty());
  if (order.size() != tensors.size()) {
    return errors::InvalidArgument("kernel_arg_order has ", order.size(),
                                   " entries, expected ", tensors.size());
  }

  arguments->clear();
  arguments->reserve(order.size());
  for (int index : order) {
    if (index < 0 || index >= static_cast<int>(tensors.size())) {
      return errors::InvalidArgument("kernel_arg_order index ", index,
                                     " is out of range [0, ", tensors.size(),
                                     ")");
    }
    arguments->push_back(tensors[index]);
  }
  return OkStatus();
}

Status CallMlirCiface(void* func, const std::vector<void*>& args) {
  switch (args.size()) {
    case 0:
      reinterpret_cast<void (*)()>(func)();
      break;
    case 1:
      reinterpret_cast<void (*)(void*)>(func)(args[0]);
      break;
    case 2:
      reinterpret_cast<void (*)(void*, void*)>(func)(args[0], args[1]);
      break;
    case 3:
      reinterpret_cast<void (*)(void*, void*, void*)>(func)(args[0], args[1],
                                                             args[2]);
      break;
    case 4:
      reinterpret_cast<void (*)(void*, void*, void*, void*)>(func)(
          args[0], args[1], args[2], args[3]);
      break;
    case 5:
      reinterpret_cast<void (*)(void*, void*, void*, void*, void*)>(func)(
          args[0], args[1], args[2], args[3], args[4]);
      break;
    case 6:
      reinterpret_cast<void (*)(void*, void*, void*, void*, void*, void*)>(
          func)(args[0], args[1], args[2], args[3], args[4], args[5]);
      break;
    case 7:
      reinterpret_cast<void (*)(void*, void*, void*, void*, void*, void*,
                               void*)>(func)(args[0], args[1], args[2], args[3],
                                             args[4], args[5], args[6]);
      break;
    case 8:
      reinterpret_cast<void (*)(void*, void*, void*, void*, void*, void*,
                               void*, void*)>(func)(args[0], args[1], args[2],
                                                     args[3], args[4], args[5],
                                                     args[6], args[7]);
      break;
    default:
      return errors::Unimplemented(
          "mlir_ciface supports up to 8 memref arguments, got ", args.size());
  }
  return OkStatus();
}

}  // namespace

// ============================================================================
// ANNCFusedOp Implementation
// ============================================================================

mutex ANNCFusedOp::lib_cache_mu_(LINKER_INITIALIZED);
std::unordered_map<std::string, void*> ANNCFusedOp::lib_cache_;

ANNCFusedOp::ANNCFusedOp(OpKernelConstruction* context)
    : OpKernel(context),
      num_constants_(0),
      num_fixed_(0),
      num_dynamic_(0),
      zero_initialize_outputs_(true),
      handle_(nullptr),
      mlir_ciface_func_(nullptr),
      annc_set_current_threadpool_(nullptr),
      annc_get_current_threadpool_(nullptr),
      loaded_(false) {
  OP_REQUIRES_OK(context, context->GetAttr("Nconstants", &num_constants_));
  OP_REQUIRES_OK(context, context->GetAttr("Nfixed", &num_fixed_));
  OP_REQUIRES_OK(context, context->GetAttr("Ndynamic", &num_dynamic_));

  OP_REQUIRES_OK(context, context->GetAttr("kernel_name", &kernel_name_));
  if (context->HasAttr("template_fingerprint")) {
    OP_REQUIRES_OK(context, context->GetAttr("template_fingerprint",
                                             &template_fingerprint_));
  }
  OP_REQUIRES_OK(context, context->GetAttr("num_outputs", &num_outputs_));
  OP_REQUIRES_OK(context, context->GetAttr("output_ranks", &output_ranks_));
  if (context->HasAttr("input_ranks")) {
    OP_REQUIRES_OK(context, context->GetAttr("input_ranks", &input_ranks_));
  }
  if (context->HasAttr("output_shapes")) {
    OP_REQUIRES_OK(context, context->GetAttr("output_shapes", &output_shapes_));
  }
  if (context->HasAttr("kernel_arg_order")) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("kernel_arg_order", &kernel_arg_order_));
  }
  OP_REQUIRES_OK(context, context->GetAttr("dynamic_dims", &dynamic_dims_));
  OP_REQUIRES_OK(context, context->GetAttr("symbolic_signature",
                                           &symbolic_signature_str_));
  if (context->HasAttr("fusion_pattern")) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("fusion_pattern", &fusion_pattern_));
  }
  OP_REQUIRES_OK(context, context->GetAttr("T", &dtype_));
  if (context->HasAttr("Toutputs")) {
    OP_REQUIRES_OK(context, context->GetAttr("Toutputs", &output_types_));
  }

  if (context->HasAttr("shared_lib_path")) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("shared_lib_path", &shared_lib_path_));
  }
  if (context->HasAttr("atir_module_path")) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("atir_module_path", &atir_module_path_));
  }
  if (context->HasAttr("abi")) {
    OP_REQUIRES_OK(context, context->GetAttr("abi", &abi_));
  } else {
    abi_ = "mlir_ciface";
  }
  if (context->HasAttr("zero_initialize_outputs")) {
    OP_REQUIRES_OK(context, context->GetAttr("zero_initialize_outputs",
                                             &zero_initialize_outputs_));
  }
  if (fusion_pattern_ == "dnn_embedding_hash_bucket") {
    zero_initialize_outputs_ = false;
  }

  OP_REQUIRES(context, !kernel_name_.empty(),
              errors::InvalidArgument("kernel_name cannot be empty"));
  OP_REQUIRES(
      context, !shared_lib_path_.empty() || !template_fingerprint_.empty(),
      errors::InvalidArgument("ANNCFused JIT requires template_fingerprint"));
  OP_REQUIRES(
      context, !shared_lib_path_.empty() || !atir_module_path_.empty(),
      errors::InvalidArgument("ANNCFused requires shared_lib_path for AOT or "
                              "atir_module_path for JIT"));
  OP_REQUIRES(context, abi_ == "mlir_ciface" || abi_ == "annc_execution_v2",
              errors::InvalidArgument("Unsupported ANNCFused abi: ", abi_));
  OP_REQUIRES(context, num_outputs_ == static_cast<int>(output_ranks_.size()),
              errors::InvalidArgument("num_outputs must match output_ranks"));
  OP_REQUIRES(
      context,
      output_types_.empty() ||
          output_types_.size() == static_cast<size_t>(num_outputs_),
      errors::InvalidArgument("Toutputs must be empty or match num_outputs"));
  OP_REQUIRES(
      context,
      input_ranks_.empty() ||
          input_ranks_.size() ==
              static_cast<size_t>(num_constants_ + num_fixed_ + num_dynamic_),
      errors::InvalidArgument(
          "input_ranks must be empty or match total input count"));
  OP_REQUIRES(context,
              output_shapes_.empty() ||
                  output_shapes_.size() == static_cast<size_t>(num_outputs_),
              errors::InvalidArgument(
                  "output_shapes must be empty or match num_outputs"));
}

ANNCFusedOp::~ANNCFusedOp() {}

annc::jit::JitCompileResult ANNCFusedOp::CompileJitKernel(
    const std::vector<annc::jit::JitArgumentSignature>& arguments) {
  AnncJitCompileRequest request;
  request.atir_module_path = atir_module_path_;
  request.kernel_name = kernel_name_;
  request.argument_shapes.reserve(arguments.size());
  for (const auto& argument : arguments) {
    request.argument_shapes.push_back(argument.dims);
  }
  std::shared_ptr<annc::jit::JitExecutable> executable;
  Status status = CompileAnncJitKernel(request, &executable);
  return status.ok()
             ? annc::jit::JitCompileResult{std::move(executable), ""}
             : annc::jit::JitCompileResult{nullptr, status.ToString()};
}

void ANNCFusedOp::Compute(OpKernelContext* context) {
  const bool profile_enabled = IsAnncFusedProfilingEnabled();
  auto t_compute_start = profile_enabled ? Clock::now() : TimePoint{};
  AnncFusedProfileSample profile_sample;

  // ── Direct OpenBLAS MatMul fast path ──
  auto t_backend_dispatch_start =
      profile_enabled ? Clock::now() : TimePoint{};
  {
    static const char* kAnncBackend = getenv("ANNC_BACKEND");
    auto lower = kernel_name_;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (kAnncBackend && strcmp(kAnncBackend, "openblas") == 0 &&
        lower.find("matmul") != std::string::npos) {
      OP_REQUIRES(
          context, false,
          errors::Unimplemented(
              "ANNC_BACKEND=openblas direct MatMul path is not implemented"));
    }
  }
  if (profile_enabled) {
    profile_sample.backend_dispatch_us =
        ElapsedUs(t_backend_dispatch_start);
  }

  const bool jit_mode = shared_lib_path_.empty();
  thread::ThreadPool* tf_thread_pool = GetTensorFlowCpuThreadPool(context);

  std::shared_ptr<annc::jit::JitExecutable> jit_executable;
  void* kernel_function = mlir_ciface_func_;
  auto set_thread_pool = annc_set_current_threadpool_;
  auto get_thread_pool = annc_get_current_threadpool_;
  bool library_loaded_this_call = false;
  auto t_load_start = profile_enabled ? Clock::now() : TimePoint{};

  if (jit_mode) {
    OP_REQUIRES(
        context, !atir_module_path_.empty(),
        errors::InvalidArgument("ANNCFused JIT requires atir_module_path"));
    std::vector<annc::jit::JitArgumentSignature> arguments;
    OP_REQUIRES_OK(
        context, BuildRuntimeArgumentShapes(
                     context, num_constants_, num_fixed_, num_dynamic_,
                     num_outputs_, input_ranks_, output_ranks_, output_shapes_,
                     kernel_arg_order_, &arguments));

    annc::jit::JitCacheKey cache_key;
    std::string cache_key_error;
    OP_REQUIRES(
        context,
        annc::jit::BuildJitCacheKey(template_fingerprint_, arguments,
                                    &cache_key, &cache_key_error),
        errors::Internal("Cannot build ANNC JIT cache key: ",
                         cache_key_error));

    const std::string key_summary = cache_key.digest.substr(0, 16);
    annc::jit::JitCompilationCache& cache = GlobalJitCompilationCache();
    annc::jit::JitCacheLookup lookup = cache.GetOrCompile(cache_key, [&] {
      auto compile_start = Clock::now();
      annc::jit::JitCompileResult compiled = CompileJitKernel(arguments);
      if (compiled.ok()) {
        if (profile_enabled) {
          LOG(INFO) << "[ANNC-JIT-CACHE] compiled key=" << key_summary
                    << " kernel=" << kernel_name_
                    << " compile_ms="
                    << (ElapsedUs(compile_start) / 1000.0);
        }
      } else {
        LOG(ERROR) << "[ANNC-JIT-CACHE] compile_failed key=" << key_summary
                   << " kernel=" << kernel_name_
                   << " compile_ms=" << (ElapsedUs(compile_start) / 1000.0)
                   << " error=" << compiled.error;
      }
      return compiled;
    });
    if (profile_enabled) {
      annc::jit::JitCacheStats stats_after = cache.stats();
      LOG(INFO) << "[ANNC-JIT-CACHE] " << CacheEventName(lookup.event)
                << " key=" << key_summary << " kernel=" << kernel_name_
                << " entries=" << stats_after.entries;
      for (const std::string& evicted_key : lookup.evictedKeys) {
        LOG(INFO) << "[ANNC-JIT-CACHE] evicted key="
                  << evicted_key.substr(0, 16)
                  << " entries=" << stats_after.entries;
      }
    }
    OP_REQUIRES(context, lookup.ok(), errors::Internal(lookup.error));
    jit_executable = std::move(lookup.executable);
    kernel_function = jit_executable->kernelFunction();
    set_thread_pool =
        reinterpret_cast<void (*)(annc::threadpool::AnncThreadPool*)>(
            jit_executable->setThreadPoolFunction());
    get_thread_pool =
        reinterpret_cast<annc::threadpool::AnncThreadPool* (*)()>(
            jit_executable->getThreadPoolFunction());
    library_loaded_this_call = lookup.event != annc::jit::CacheEvent::kHit;
  } else if (!loaded_ || current_so_path_ != shared_lib_path_) {
    OP_REQUIRES_OK(context, LoadLibrary(shared_lib_path_));
    current_so_path_ = shared_lib_path_;
    library_loaded_this_call = true;
  }
  // kernel_function was captured before the AOT branch above; refresh it from
  // the member LoadLibrary just resolved, or the first invocation of a kernel
  // always sees a null pointer.
  if (!jit_mode) kernel_function = mlir_ciface_func_;
  if (profile_enabled && library_loaded_this_call) {
    profile_sample.load_library_us = ElapsedUs(t_load_start);
  }

  Status execute_status;
  auto t_threadpool_start = profile_enabled ? Clock::now() : TimePoint{};
  TensorFlowAnncThreadPool adapter(tf_thread_pool);
  ScopedAnncThreadPool scoped(tf_thread_pool ? &adapter : nullptr,
                              set_thread_pool, get_thread_pool);
  if (profile_enabled) {
    profile_sample.threadpool_setup_us = ElapsedUs(t_threadpool_start);
  }

  if (abi_ == "annc_execution_v2") {
    execute_status = ExecuteExecutionV2Kernel(
        context, kernel_function, profile_enabled, &profile_sample);
  } else {
    execute_status = ExecuteMlirCifaceKernel(
        context, kernel_function, profile_enabled, &profile_sample);
  }
  OP_REQUIRES_OK(context, execute_status);
  auto t_threadpool_restore_start =
      profile_enabled ? Clock::now() : TimePoint{};
  scoped.Restore();
  if (profile_enabled) {
    profile_sample.threadpool_restore_us =
        ElapsedUs(t_threadpool_restore_start);
  }
  if (!profile_enabled) {
    return;
  }
  double total_us = ElapsedUs(t_compute_start);

  static std::map<std::string, ProfileStats> stats;
  static int total_calls = 0;
  const int n = num_constants_ + num_fixed_ + num_dynamic_;
  auto key = BuildProfileKey(context, n, input_ranks_, output_shapes_,
                             kernel_name_, fusion_pattern_);
  mutex_lock lock(profile_stats_mu_);
  auto& s = stats[key];
  s.count++;
  s.load_library_us += profile_sample.load_library_us;
  s.backend_dispatch_us += profile_sample.backend_dispatch_us;
  s.threadpool_setup_us += profile_sample.threadpool_setup_us;
  s.threadpool_restore_us += profile_sample.threadpool_restore_us;
  s.input_memref_us += profile_sample.input_memref_us;
  s.output_alloc_us += profile_sample.output_alloc_us;
  s.output_init_us += profile_sample.output_init_us;
  s.output_memref_us += profile_sample.output_memref_us;
  s.arg_order_us += profile_sample.arg_order_us;
  s.execution_setup_us += profile_sample.execution_setup_us;
  s.execution_post_us += profile_sample.execution_post_us;
  s.kernel_us += profile_sample.kernel_us;
  s.execution_callback_us += profile_sample.execution_callback_us;
  s.tensorflow_output_alloc_us +=
      profile_sample.tensorflow_output_alloc_us;
  s.tensorflow_temp_alloc_us += profile_sample.tensorflow_temp_alloc_us;
  s.execution_output_callbacks +=
      profile_sample.execution_output_callbacks;
  s.execution_temp_callbacks += profile_sample.execution_temp_callbacks;
  s.total_us += total_us;
  total_calls++;
  if (total_calls % AnncFusedProfileInterval() == 0) {
    LogProfileStats("ANNC-FUSED-PROFILE", total_calls, stats);
  }
}

Status ANNCFusedOp::ExecuteExecutionV2Kernel(
    OpKernelContext* context, void* kernel_function, bool profile_enabled,
    AnncFusedProfileSample* profile_sample) {
  if (!kernel_function) {
    return errors::FailedPrecondition(
        "Execution V2 kernel symbol is not loaded");
  }
  const int expected_inputs = num_constants_ + num_fixed_ + num_dynamic_;
  if (context->num_inputs() != expected_inputs) {
    return errors::InvalidArgument("ANNCFused expected ", expected_inputs,
                                   " inputs, got ", context->num_inputs());
  }
  if (expected_inputs > 7) {
    return errors::Unimplemented(
        "annc_execution_v2 supports up to 7 inputs, got ", expected_inputs);
  }
  auto t_execution_setup_start =
      profile_enabled ? Clock::now() : TimePoint{};
  TensorFlowExecutionState state{context,
                                 &output_types_,
                                 &output_ranks_,
                                 OkStatus(),
                                 {},
                                 {},
                                 profile_enabled,
                                 profile_sample};
  AnncExecutionHandle handle{&state};
  AnncExecutionContext execution{sizeof(AnncExecutionContext),
                                 ANNC_EXECUTION_ABI_VERSION,
                                 &handle,
                                 &AllocateExecutionOutput,
                                 &AllocateExecutionTemp,
                                 &ReportExecutionError};
  if (output_types_.size() != static_cast<size_t>(num_outputs_)) {
    return errors::InvalidArgument("annc_execution_v2 requires Toutputs");
  }

  std::vector<void*> memrefs(expected_inputs, nullptr);
  std::vector<std::array<RankedMemRefDescriptor<0>, 1>> rank0(expected_inputs);
  std::vector<std::array<RankedMemRefDescriptor<1>, 1>> rank1(expected_inputs);
  std::vector<std::array<RankedMemRefDescriptor<2>, 1>> rank2(expected_inputs);
  std::vector<std::array<RankedMemRefDescriptor<3>, 1>> rank3(expected_inputs);
  std::vector<std::array<RankedMemRefDescriptor<4>, 1>> rank4(expected_inputs);
  std::vector<std::array<RankedMemRefDescriptor<5>, 1>> rank5(expected_inputs);
  std::vector<std::array<RankedMemRefDescriptor<6>, 1>> rank6(expected_inputs);
  std::vector<std::vector<AnncStringRef>> string_storage(expected_inputs);
  if (profile_enabled && profile_sample) {
    profile_sample->execution_setup_us =
        ElapsedUs(t_execution_setup_start);
  }

  auto t_input_memref_start = profile_enabled ? Clock::now() : TimePoint{};
  for (int i = 0; i < expected_inputs; ++i) {
    const Tensor& tensor = context->input(i);
    int rank = input_ranks_.empty() ? tensor.dims() : input_ranks_[i];
    TF_RETURN_IF_ERROR(BuildRankedMemRefArg(
        tensor, rank, &rank0[i], &rank1[i], &rank2[i], &rank3[i], &rank4[i],
        &rank5[i], &rank6[i], &memrefs[i], &string_storage[i]));
  }
  if (profile_enabled && profile_sample) {
    profile_sample->input_memref_us = ElapsedUs(t_input_memref_start);
  }

  auto t_arg_order_start = profile_enabled ? Clock::now() : TimePoint{};
  std::vector<void*> args;
  args.reserve(expected_inputs + 1);
  args.push_back(&execution);
  args.insert(args.end(), memrefs.begin(), memrefs.end());
  if (profile_enabled && profile_sample) {
    profile_sample->arg_order_us = ElapsedUs(t_arg_order_start);
  }

  auto t_kernel_start = profile_enabled ? Clock::now() : TimePoint{};
  Status status = CallMlirCiface(kernel_function, args);
  if (profile_enabled && profile_sample) {
    profile_sample->kernel_us = ElapsedUs(t_kernel_start);
  }
  auto t_execution_post_start =
      profile_enabled ? Clock::now() : TimePoint{};
  if (!state.error.ok()) return state.error;
  if (!status.ok()) return status;
  if (state.outputs.size() != static_cast<size_t>(num_outputs_) ||
      std::any_of(state.outputs.begin(), state.outputs.end(),
                  [](const Tensor* output) { return output == nullptr; })) {
    return errors::Internal("annc_execution_v2 did not allocate all outputs");
  }
  if (profile_enabled && profile_sample) {
    profile_sample->execution_post_us =
        ElapsedUs(t_execution_post_start);
  }
  return OkStatus();
}

Status ANNCFusedOp::LoadLibrary(const std::string& so_path) {
  mutex_lock lock(lib_cache_mu_);

  auto it = lib_cache_.find(so_path);
  if (it != lib_cache_.end()) {
    handle_ = it->second;
  } else {
    handle_ = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) {
      const char* error = dlerror();
      LOG(ERROR) << "[ANNC-FUSED-LOAD] dlopen failed so_path=" << so_path
                 << " error=" << (error ? error : "<null>");
      return errors::NotFound("Cannot load library ", so_path, ": ",
                              error ? error : "<null>");
    }
    lib_cache_[so_path] = handle_;
  }

  return ResolveLibrarySymbols(so_path);
}

Status ANNCFusedOp::ResolveLibrarySymbols(const std::string& so_path) {
  std::string symbol_name = "_mlir_ciface_" + kernel_name_;
  void* symbol = dlsym(handle_, symbol_name.c_str());
  if (!symbol) {
    const char* error = dlerror();
    LOG(ERROR) << "[ANNC-FUSED-LOAD] dlsym failed symbol=" << symbol_name
               << " so_path=" << so_path
               << " error=" << (error ? error : "<null>");
    return errors::NotFound("Cannot find symbol ", symbol_name, " in ",
                            so_path);
  }
  mlir_ciface_func_ = symbol;

  annc_set_current_threadpool_ =
      reinterpret_cast<void (*)(annc::threadpool::AnncThreadPool*)>(
          dlsym(handle_, "annc_set_current_threadpool"));
  annc_get_current_threadpool_ =
      reinterpret_cast<annc::threadpool::AnncThreadPool* (*)()>(
          dlsym(handle_, "annc_get_current_threadpool"));

  if (!annc_set_current_threadpool_ || !annc_get_current_threadpool_) {
    annc_set_current_threadpool_ = nullptr;
    annc_get_current_threadpool_ = nullptr;
  }

  loaded_ = true;
  return OkStatus();
}

Status ANNCFusedOp::ExecuteMlirCifaceKernel(OpKernelContext* context,
                                            void* kernel_function,
                                            bool profile_enabled,
                                            AnncFusedProfileSample* profile_sample) {
  if (!kernel_function) {
    LOG(ERROR) << "[ANNC-FUSED-EXEC] kernel symbol is null kernel="
               << kernel_name_;
    return errors::FailedPrecondition("MLIR C interface kernel is not loaded");
  }

  const int expected_inputs = num_constants_ + num_fixed_ + num_dynamic_;
  if (context->num_inputs() != expected_inputs) {
    LOG(ERROR) << "[ANNC-FUSED-EXEC] input count mismatch kernel="
               << kernel_name_ << " expected=" << expected_inputs
               << " actual=" << context->num_inputs();
    return errors::InvalidArgument("ANNCFused expected ", expected_inputs,
                                   " inputs, got ", context->num_inputs());
  }

  const int total_memrefs = expected_inputs + num_outputs_;
  std::vector<void*> memrefs(total_memrefs, nullptr);
  std::vector<std::array<RankedMemRefDescriptor<0>, 1>> rank0(total_memrefs);
  std::vector<std::array<RankedMemRefDescriptor<1>, 1>> rank1(total_memrefs);
  std::vector<std::array<RankedMemRefDescriptor<2>, 1>> rank2(total_memrefs);
  std::vector<std::array<RankedMemRefDescriptor<3>, 1>> rank3(total_memrefs);
  std::vector<std::array<RankedMemRefDescriptor<4>, 1>> rank4(total_memrefs);
  std::vector<std::array<RankedMemRefDescriptor<5>, 1>> rank5(total_memrefs);
  std::vector<std::array<RankedMemRefDescriptor<6>, 1>> rank6(total_memrefs);
  std::vector<std::vector<AnncStringRef>> string_storage(total_memrefs);

  auto t_input_memref_start = profile_enabled ? Clock::now() : TimePoint{};
  for (int i = 0; i < expected_inputs; ++i) {
    const Tensor& tensor = context->input(i);
    int rank = input_ranks_.empty() ? tensor.dims() : input_ranks_[i];
    Status status = BuildRankedMemRefArg(
        tensor, rank, &rank0[i], &rank1[i], &rank2[i], &rank3[i], &rank4[i],
        &rank5[i], &rank6[i], &memrefs[i], &string_storage[i]);
    if (!status.ok()) {
      LOG(ERROR) << "[ANNC-FUSED-EXEC] failed to build input memref kernel="
                 << kernel_name_ << " input_index=" << i
                 << " expected_rank=" << rank
                 << " status=" << status.ToString();
      return status;
    }
  }
  if (profile_enabled && profile_sample) {
    profile_sample->input_memref_us = ElapsedUs(t_input_memref_start);
  }

  for (int i = 0; i < num_outputs_; ++i) {
    Tensor* output = nullptr;

    TensorShape out_shape =
        InferOutputShape(context, i, output_ranks_[i], output_shapes_,
                         num_constants_, num_fixed_);
    auto t_output_alloc_start = profile_enabled ? Clock::now() : TimePoint{};
    Status alloc_status = context->allocate_output(i, out_shape, &output);
    if (!alloc_status.ok()) {
      LOG(ERROR) << "[ANNC-FUSED-EXEC] allocate_output failed kernel="
                 << kernel_name_ << " output_index=" << i
                 << " shape=" << out_shape.DebugString()
                 << " status=" << alloc_status.ToString();
      return alloc_status;
    }
    if (profile_enabled && profile_sample) {
      profile_sample->output_alloc_us += ElapsedUs(t_output_alloc_start);
    }

    if (zero_initialize_outputs_) {
      auto t_output_init_start = profile_enabled ? Clock::now() : TimePoint{};
      output->flat<float>().setZero();
      if (profile_enabled && profile_sample) {
        profile_sample->output_init_us += ElapsedUs(t_output_init_start);
      }
    }

    int memref_index = expected_inputs + i;
    auto t_output_memref_start = profile_enabled ? Clock::now() : TimePoint{};
    Status status = BuildRankedMemRefArg(
        *output, output_ranks_[i], &rank0[memref_index], &rank1[memref_index],
        &rank2[memref_index], &rank3[memref_index], &rank4[memref_index],
        &rank5[memref_index], &rank6[memref_index], &memrefs[memref_index],
        &string_storage[memref_index]);
    if (!status.ok()) {
      LOG(ERROR) << "[ANNC-FUSED-EXEC] failed to build output memref kernel="
                 << kernel_name_ << " output_index=" << i
                 << " rank=" << output_ranks_[i]
                 << " status=" << status.ToString();
      return status;
    }
    if (profile_enabled && profile_sample) {
      profile_sample->output_memref_us += ElapsedUs(t_output_memref_start);
    }
  }

  auto t_arg_order_start = profile_enabled ? Clock::now() : TimePoint{};
  std::vector<int> order = ResolveKernelArgOrder(
      kernel_arg_order_, expected_inputs, num_outputs_, !input_ranks_.empty());

  std::vector<void*> args;
  args.reserve(order.size());
  for (int index : order) {
    if (index < 0 || index >= total_memrefs) {
      LOG(ERROR) << "[ANNC-FUSED-EXEC] kernel_arg_order out of range kernel="
                 << kernel_name_ << " index=" << index
                 << " total_memrefs=" << total_memrefs
                 << " order=" << FormatVector(order);
      return errors::InvalidArgument("kernel_arg_order index ", index,
                                     " is out of range [0, ", total_memrefs,
                                     ")");
    }
    args.push_back(memrefs[index]);
  }
  if (profile_enabled && profile_sample) {
    profile_sample->arg_order_us = ElapsedUs(t_arg_order_start);
  }

  auto t_kernel_start = profile_enabled ? Clock::now() : TimePoint{};
  Status status = CallMlirCiface(kernel_function, args);
  if (!status.ok()) {
    LOG(ERROR) << "[ANNC-FUSED-EXEC] kernel call failed kernel=" << kernel_name_
               << " status=" << status.ToString();
  }
  if (profile_enabled && profile_sample) {
    profile_sample->kernel_us = ElapsedUs(t_kernel_start);
  }
  return status;
}

}  // namespace tensorflow
