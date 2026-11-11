#include "fused_op.h"
#include "ciface_dispatch.h"
#include "fused_op_jit.h"
#include "threadpool.h"
#include "threadpool_callbacks.h"

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
#include <sstream>
#include <iostream>
#include <string>
#include <utility>

#include "Kernel/MemRefTypes.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

struct AnncFusedProfileSample {
  double load_library_us = 0.0;
  double jit_arg_shapes_us = 0.0;
  double jit_cache_key_us = 0.0;
  double jit_cache_lookup_us = 0.0;
  double jit_cache_lookup_hit_us = 0.0;
  double jit_compile_us = 0.0;
  int jit_hit_count = 0;
  int jit_compile_count = 0;
  int jit_wait_count = 0;
  double backend_dispatch_us = 0.0;
  double threadpool_setup_us = 0.0;
  double threadpool_restore_us = 0.0;
  double input_memref_us = 0.0;
  double output_alloc_us = 0.0;
  double output_init_us = 0.0;
  double output_memref_us = 0.0;
  double arg_order_us = 0.0;
  double kernel_us = 0.0;
};

namespace {

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

double ElapsedUs(TimePoint start, TimePoint end = Clock::now()) {
  return std::chrono::duration<double, std::micro>(end - start).count();
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

bool IsJitShapeFastPathEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("ANNC_ENABLE_JIT_SHAPE_FASTPATH");
    return !value || value[0] == '\0' || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

bool SameJitArguments(
    const std::vector<annc::jit::JitArgumentSignature>& lhs,
    const std::vector<annc::jit::JitArgumentSignature>& rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].dims != rhs[i].dims) return false;
  }
  return true;
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
  double jit_arg_shapes_us = 0.0;
  double jit_cache_key_us = 0.0;
  double jit_cache_lookup_us = 0.0;
  double jit_cache_lookup_hit_us = 0.0;
  double jit_compile_us = 0.0;
  int jit_hits = 0;
  int jit_compiles = 0;
  int jit_waits = 0;
  // 稳态热路径统计：仅统计 JIT cache 命中（或 AOT 已加载）的调用，
  // 不含首次编译与 wait 等冷启动成本，无需靠减法推导。
  int steady_calls = 0;
  double steady_total_us = 0.0;
  double steady_kernel_us = 0.0;
  double steady_overhead_us = 0.0;
  double backend_dispatch_us = 0.0;
  double threadpool_setup_us = 0.0;
  double threadpool_restore_us = 0.0;
  double input_memref_us = 0.0;
  double output_alloc_us = 0.0;
  double output_init_us = 0.0;
  double output_memref_us = 0.0;
  double arg_order_us = 0.0;
  double kernel_us = 0.0;
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
  for (const auto& entry : stats) {
    const auto& key = entry.first;
    const auto& s = entry.second;
    const double count = static_cast<double>(s.count);
    const double jit_hits = static_cast<double>(s.jit_hits);
    const double steady_calls = static_cast<double>(s.steady_calls);
    LOG(INFO) << "[" << tag << "] key=" << key << " count=" << s.count
              << " jit_hits=" << s.jit_hits
              << " jit_compiles=" << s.jit_compiles
              << " jit_waits=" << s.jit_waits
              << " steady_calls=" << s.steady_calls
              << " avg_steady_total="
              << (steady_calls > 0 ? (s.steady_total_us / steady_calls) : 0.0)
              << " us"
              << " avg_steady_kernel="
              << (steady_calls > 0 ? (s.steady_kernel_us / steady_calls) : 0.0)
              << " us"
              << " avg_steady_overhead="
              << (steady_calls > 0 ? (s.steady_overhead_us / steady_calls)
                                   : 0.0)
              << " us"
              << " avg_jit_arg_shapes=" << (s.jit_arg_shapes_us / count)
              << " us"
              << " avg_jit_cache_key=" << (s.jit_cache_key_us / count)
              << " us"
              << " avg_jit_cache_lookup="
              << (s.jit_cache_lookup_us / count) << " us"
              << " avg_jit_hit_lookup="
              << (jit_hits > 0 ? (s.jit_cache_lookup_hit_us / jit_hits) : 0.0)
              << " us"
              << " avg_jit_compile="
              << (s.jit_compiles > 0
                      ? (s.jit_compile_us / static_cast<double>(s.jit_compiles))
                      : 0.0)
              << " us"
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
              << " avg_kernel=" << (s.kernel_us / count) << " us"
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

class TensorFlowAnncThreadPool final {
 public:
  explicit TensorFlowAnncThreadPool(thread::ThreadPool* workers) {
    if (workers) {
      const auto callbacks =
          annc::tensorflow::MakeThreadPoolCallbacks(*workers);
      adapter_.reset(anncCreateThreadPoolAdapter(&callbacks));
    }
  }

  annc::threadpool::AnncThreadPool* get() const {
    // The kernel's existing C hooks take an opaque C++ pool pointer. Only
    // support and the generated kernel inspect that interface's layout.
    return reinterpret_cast<annc::threadpool::AnncThreadPool*>(adapter_.get());
  }

 private:
  annc::tensorflow::ThreadPoolAdapterPtr adapter_;
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
                             const std::vector<std::vector<int64_t>>& output_specs,
                             int num_constants, int num_fixed) {
  if (output_index < static_cast<int>(output_specs.size()) &&
      !output_specs[output_index].empty()) {
    const std::vector<int64_t>& dims = output_specs[output_index];
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
    const std::vector<std::vector<int64_t>>& output_specs,
    const std::vector<int>& configured_order,
    const std::vector<int>& context_input_order, bool context_v3,
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

  // V3 specializes canonical semantic inputs only. The adapter transports
  // TensorFlow inputs in grouped order, so apply the same permutation here.
  if (context_v3) {
    if (!configured_order.empty()) {
      return errors::InvalidArgument(
          "annc_context_v3 does not accept kernel_arg_order");
    }
    arguments->clear();
    arguments->reserve(tensors.size());
    if (context_input_order.empty()) {
      *arguments = std::move(tensors);
      return OkStatus();
    }
    if (context_input_order.size() != tensors.size()) {
      return errors::InvalidArgument(
          "context_input_order must match total input count");
    }
    for (int transport_index : context_input_order) {
      if (transport_index < 0 ||
          transport_index >= static_cast<int>(tensors.size())) {
        return errors::InvalidArgument("context_input_order index ",
                                       transport_index, " is out of range");
      }
      arguments->push_back(tensors[transport_index]);
    }
    return OkStatus();
  }

  for (int i = 0; i < num_outputs; ++i) {
    TensorShape shape = InferOutputShape(
        context, i, output_ranks[i], output_specs, num_constants, num_fixed);
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
  if (!annc::ciface::Call(func, args)) {
    return errors::Unimplemented("mlir_ciface supports up to ",
                                 annc::ciface::kMaxArgs,
                                 " memref arguments, got ", args.size());
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
  OP_REQUIRES(
      context,
      num_constants_ >= 0 && num_fixed_ >= 0 && num_dynamic_ >= 0 &&
          static_cast<int64_t>(num_constants_) + num_fixed_ + num_dynamic_ <=
              std::numeric_limits<int>::max(),
      errors::InvalidArgument("ANNCFused input counts are out of range"));

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
  if (context->HasAttr("context_input_order")) {
    OP_REQUIRES_OK(context, context->GetAttr("context_input_order",
                                             &context_input_order_));
  }
  if (context->HasAttr("output_shapes")) {
    OP_REQUIRES_OK(context, context->GetAttr("output_shapes", &output_shapes_));
  }
  // Parse once here; Compute() runs on the hot path and must not re-parse the
  // attribute strings (stringstream + getline + stoll) on every call.
  output_shape_specs_.reserve(output_shapes_.size());
  for (const std::string& spec : output_shapes_) {
    output_shape_specs_.push_back(ParseShapeSpec(spec));
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
  OP_REQUIRES(context,
              abi_ == "mlir_ciface" || abi_ == ANNC_CONTEXT_ABI_NAME,
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

  if (abi_ == ANNC_CONTEXT_ABI_NAME) {
    if (context->HasAttr("context_i64_attrs")) {
      std::vector<int64> values;
      OP_REQUIRES_OK(context, context->GetAttr("context_i64_attrs", &values));
      context_i64_attrs_.assign(values.begin(), values.end());
    }
    const int num_inputs = num_constants_ + num_fixed_ + num_dynamic_;
    OP_REQUIRES(context, kernel_arg_order_.empty(),
                errors::InvalidArgument(
                    "annc_context_v3 does not accept kernel_arg_order"));
    OP_REQUIRES(
        context,
        context_input_order_.empty() ||
            context_input_order_.size() == static_cast<size_t>(num_inputs),
        errors::InvalidArgument(
            "context_input_order must match total input count"));
    std::vector<bool> seen(num_inputs, false);
    for (int index : context_input_order_) {
      OP_REQUIRES(context, index >= 0 && index < num_inputs && !seen[index],
                  errors::InvalidArgument(
                      "context_input_order must be a complete permutation"));
      seen[index] = true;
    }
    for (const auto* ranks : {&input_ranks_, &output_ranks_}) {
      for (int rank : *ranks) {
        OP_REQUIRES(context, rank >= 0 && rank <= ANNC_CONTEXT_MAX_RANK,
                    errors::InvalidArgument(
                        "annc_context_v3 requires ranks in [0, 8]"));
      }
    }
    for (const char* attr : {"Tconstants", "Tfixed", "Tdynamic"}) {
      std::vector<DataType> types;
      OP_REQUIRES_OK(context, context->GetAttr(attr, &types));
      const int expected = std::strcmp(attr, "Tconstants") == 0 ? num_constants_
                           : std::strcmp(attr, "Tfixed") == 0   ? num_fixed_
                                                                : num_dynamic_;
      OP_REQUIRES(context, types.size() == static_cast<size_t>(expected),
                  errors::InvalidArgument(attr, " must match its input count"));
      input_types_.insert(input_types_.end(), types.begin(), types.end());
    }
    OP_REQUIRES(
        context, input_types_.size() == static_cast<size_t>(num_inputs),
        errors::InvalidArgument(
            "annc_context_v3 input types must match total input count"));
    OP_REQUIRES(context,
                output_types_.size() == static_cast<size_t>(num_outputs_),
                errors::InvalidArgument("annc_context_v3 requires Toutputs"));
    for (const auto* types : {&input_types_, &output_types_}) {
      for (DataType type : *types) {
        OP_REQUIRES(
            context,
            ToAnncElementType(type) != 0 ||
                (types == &input_types_ && type == DT_STRING),
            errors::InvalidArgument(
                "annc_context_v3 requires numeric tensors or string inputs"));
      }
    }
  }

  // fusion.metadata is intentionally excluded from the compiler's template
  // fingerprint. Include its positional runtime contract here. Serialize
  // canonical types/ranks too; TensorFlow transport order is not code identity.
  jit_template_identity_ = template_fingerprint_ + ":" + abi_;
  if (abi_ == ANNC_CONTEXT_ABI_NAME) {
    auto append = [&](const std::string& value) {
      jit_template_identity_ +=
          ":" + std::to_string(value.size()) + ":" + value;
    };
    append("annc-context-slots-v2");
    append(std::to_string(context_i64_attrs_.size()));
    for (int64_t value : context_i64_attrs_) append(std::to_string(value));
    append(std::to_string(input_types_.size()));
    for (size_t i = 0; i < input_types_.size(); ++i) {
      const size_t transport =
          context_input_order_.empty() ? i : context_input_order_[i];
      append(std::to_string(static_cast<int>(input_types_[transport])));
      append(input_ranks_.empty() ? "runtime"
                                  : std::to_string(input_ranks_[transport]));
    }
    append(std::to_string(output_types_.size()));
    for (size_t i = 0; i < output_types_.size(); ++i) {
      append(std::to_string(static_cast<int>(output_types_[i])));
      append(std::to_string(output_ranks_[i]));
    }
  }
}

ANNCFusedOp::~ANNCFusedOp() {}

annc::jit::JitCompileResult ANNCFusedOp::CompileJitKernel(
    const std::vector<annc::jit::JitArgumentSignature>& arguments,
    int64_t intra_thread_count) {
  AnncJitCompileRequest request;
  request.atir_module_path = atir_module_path_;
  request.kernel_name = kernel_name_;
  request.intra_thread_count = intra_thread_count;
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
  // 稳态判定：JIT 模式下仅 cache 命中算稳态；AOT 模式下未触发库加载即稳态。
  bool steady_call = !jit_mode;
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
    auto t_jit_arg_shapes_start = profile_enabled ? Clock::now() : TimePoint{};
    OP_REQUIRES_OK(
        context,
        BuildRuntimeArgumentShapes(
            context, num_constants_, num_fixed_, num_dynamic_, num_outputs_,
            input_ranks_, output_ranks_, output_shape_specs_, kernel_arg_order_,
            context_input_order_, abi_ == ANNC_CONTEXT_ABI_NAME, &arguments));
    if (profile_enabled) {
      profile_sample.jit_arg_shapes_us = ElapsedUs(t_jit_arg_shapes_start);
    }

    auto t_jit_cache_lookup_start =
        profile_enabled ? Clock::now() : TimePoint{};
    if (IsJitShapeFastPathEnabled()) {
      mutex_lock lock(jit_fastpath_mu_);
      if (last_jit_executable_ &&
          SameJitArguments(last_jit_arguments_, arguments)) {
        jit_executable = last_jit_executable_;
      }
    }
    if (jit_executable) {
      if (profile_enabled) {
        profile_sample.jit_cache_lookup_us =
            ElapsedUs(t_jit_cache_lookup_start);
        profile_sample.jit_cache_lookup_hit_us =
            profile_sample.jit_cache_lookup_us;
        profile_sample.jit_hit_count = 1;
      }
      steady_call = true;
    } else {
      annc::jit::JitCacheKey cache_key;
      std::string cache_key_error;
      auto t_jit_cache_key_start =
          profile_enabled ? Clock::now() : TimePoint{};
      OP_REQUIRES(context,
                  annc::jit::BuildJitCacheKey(jit_template_identity_, arguments,
                                              &cache_key, &cache_key_error),
                  errors::Internal("Cannot build ANNC JIT cache key: ",
                                   cache_key_error));
      if (profile_enabled) {
        profile_sample.jit_cache_key_us = ElapsedUs(t_jit_cache_key_start);
      }

      const std::string key_summary = cache_key.digest.substr(0, 16);
      annc::jit::JitCompilationCache& cache = GlobalJitCompilationCache();
      annc::jit::JitCacheLookup lookup = cache.GetOrCompile(cache_key, [&] {
        auto compile_start = Clock::now();
        // Read the pool size only on the compile (miss) path: the GEMM plan is
        // thread-count specialized and must match the intra-op pool that will
        // execute it. The pool is fixed at session creation, so this value is
        // stable for the cache lifetime and intentionally stays out of the key.
        const int64_t intra_thread_count =
            tf_thread_pool ? tf_thread_pool->NumThreads() : 1;
        annc::jit::JitCompileResult compiled =
            CompileJitKernel(arguments, intra_thread_count);
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
        if (profile_enabled) {
          profile_sample.jit_compile_us = ElapsedUs(compile_start);
        }
        return compiled;
      });
      if (profile_enabled) {
        profile_sample.jit_cache_lookup_us =
            ElapsedUs(t_jit_cache_lookup_start);
        if (lookup.event == annc::jit::CacheEvent::kHit) {
          profile_sample.jit_cache_lookup_hit_us =
              profile_sample.jit_cache_lookup_us;
          profile_sample.jit_hit_count = 1;
        } else if (lookup.event == annc::jit::CacheEvent::kMiss) {
          profile_sample.jit_compile_count = 1;
        } else if (lookup.event == annc::jit::CacheEvent::kWait) {
          profile_sample.jit_wait_count = 1;
        }
        // stats() 与 GetOrCompile 共用同一把全局 mutex，逐 hit 调用会在
        // 并发推理时造成二次锁争用，叠加 LOG 锁进一步放大延迟。
        if (lookup.event != annc::jit::CacheEvent::kHit) {
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
      }
      OP_REQUIRES(context, lookup.ok(), errors::Internal(lookup.error));
      jit_executable = std::move(lookup.executable);
      library_loaded_this_call =
          lookup.event != annc::jit::CacheEvent::kHit;
      steady_call = (lookup.event == annc::jit::CacheEvent::kHit);
      if (IsJitShapeFastPathEnabled()) {
        mutex_lock lock(jit_fastpath_mu_);
        last_jit_arguments_ = arguments;
        last_jit_executable_ = jit_executable;
      }
    }
    kernel_function = jit_executable->kernelFunction();
    set_thread_pool =
        reinterpret_cast<void (*)(annc::threadpool::AnncThreadPool*)>(
            jit_executable->setThreadPoolFunction());
    get_thread_pool =
        reinterpret_cast<annc::threadpool::AnncThreadPool* (*)()>(
            jit_executable->getThreadPoolFunction());
  } else if (!loaded_ || current_so_path_ != shared_lib_path_) {
    OP_REQUIRES_OK(context, LoadLibrary(shared_lib_path_));
    current_so_path_ = shared_lib_path_;
    library_loaded_this_call = true;
    steady_call = false;
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
  OP_REQUIRES(
      context, !tf_thread_pool || adapter.get(),
      errors::ResourceExhausted("Cannot create ANNC thread-pool adapter"));
  ScopedAnncThreadPool scoped(adapter.get(), set_thread_pool, get_thread_pool);
  if (profile_enabled) {
    profile_sample.threadpool_setup_us = ElapsedUs(t_threadpool_start);
  }

  if (abi_ == ANNC_CONTEXT_ABI_NAME) {
    auto t_kernel_start = profile_enabled ? Clock::now() : TimePoint{};
    execute_status = ExecuteContextV3Kernel(context, kernel_function);
    if (profile_enabled) {
      profile_sample.kernel_us = ElapsedUs(t_kernel_start);
    }
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
  s.jit_arg_shapes_us += profile_sample.jit_arg_shapes_us;
  s.jit_cache_key_us += profile_sample.jit_cache_key_us;
  s.jit_cache_lookup_us += profile_sample.jit_cache_lookup_us;
  s.jit_cache_lookup_hit_us += profile_sample.jit_cache_lookup_hit_us;
  s.jit_compile_us += profile_sample.jit_compile_us;
  s.jit_hits += profile_sample.jit_hit_count;
  s.jit_compiles += profile_sample.jit_compile_count;
  s.jit_waits += profile_sample.jit_wait_count;
  if (steady_call) {
    s.steady_calls++;
    s.steady_total_us += total_us;
    s.steady_kernel_us += profile_sample.kernel_us;
    s.steady_overhead_us += total_us - profile_sample.kernel_us;
  }
  s.backend_dispatch_us += profile_sample.backend_dispatch_us;
  s.threadpool_setup_us += profile_sample.threadpool_setup_us;
  s.threadpool_restore_us += profile_sample.threadpool_restore_us;
  s.input_memref_us += profile_sample.input_memref_us;
  s.output_alloc_us += profile_sample.output_alloc_us;
  s.output_init_us += profile_sample.output_init_us;
  s.output_memref_us += profile_sample.output_memref_us;
  s.arg_order_us += profile_sample.arg_order_us;
  s.kernel_us += profile_sample.kernel_us;
  s.total_us += total_us;
  total_calls++;
  if (total_calls % AnncFusedProfileInterval() == 0) {
    LogProfileStats("ANNC-FUSED-PROFILE", total_calls, stats);
  }
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
        InferOutputShape(context, i, output_ranks_[i], output_shape_specs_,
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
