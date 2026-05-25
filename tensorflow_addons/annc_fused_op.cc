#include "annc_fused_op.h"

#include <algorithm>
#include <array>
#include <optional>
#include <sstream>

#include "Kernel/threadpool/ThreadPool.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {

namespace {

class TensorFlowAnncThreadPool final
    : public annc::threadpool::AnncThreadPool {
 public:
  explicit TensorFlowAnncThreadPool(thread::ThreadPool* workers)
      : workers_(workers) {}

  int num_threads() const override {
    return workers_ ? workers_->NumThreads() : 1;
  }

  bool in_parallel_region() const override {
    return workers_ && workers_->CurrentThreadId() >= 0;
  }

  void parallel_for(
      int64_t total,
      const annc::threadpool::ParallelForOptions& options,
      const std::function<void(int64_t, int64_t)>& fn) override {
    if (!workers_ || total <= 0) {
      if (total > 0) fn(0, total);
      return;
    }

    if (options.grain_size.has_value()) {
      workers_->ParallelFor(
          total, thread::ThreadPool::SchedulingParams(
                     thread::ThreadPool::SchedulingStrategy::kFixedBlockSize,
                     std::nullopt,
                     std::max<int64_t>(*options.grain_size, 1)),
          fn);
      return;
    }

    const int64_t cost_per_unit =
        std::max<int64_t>(options.cost_per_unit.value_or(1), 1);
    workers_->ParallelFor(
        total, thread::ThreadPool::SchedulingParams(
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
        previous_(get_current_threadpool ? get_current_threadpool() : nullptr) {
    if (set_current_threadpool_) {
      set_current_threadpool_(thread_pool);
    }
  }

  ~ScopedAnncThreadPool() {
    if (set_current_threadpool_) {
      set_current_threadpool_(previous_);
    }
  }

  ScopedAnncThreadPool(const ScopedAnncThreadPool&) = delete;
  ScopedAnncThreadPool& operator=(const ScopedAnncThreadPool&) = delete;

 private:
  void (*set_current_threadpool_)(annc::threadpool::AnncThreadPool*);
  annc::threadpool::AnncThreadPool* previous_;
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
Status CreateRankedMemRef(const Tensor& tensor,
                          RankedMemRefDescriptor<Rank>* ref) {
  if (tensor.dims() != Rank) {
    return errors::InvalidArgument("Expected rank ", Rank, ", got ",
                                   tensor.dims());
  }

  ref->allocated =
      const_cast<void*>(static_cast<const void*>(tensor.tensor_data().data()));
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

Status CreateRankedMemRef(const Tensor& tensor,
                          RankedMemRefDescriptor<0>* ref) {
  if (tensor.dims() != 0) {
    return errors::InvalidArgument("Expected rank 0, got ", tensor.dims());
  }

  ref->allocated =
      const_cast<void*>(static_cast<const void*>(tensor.tensor_data().data()));
  ref->aligned = ref->allocated;
  ref->offset = 0;
  return OkStatus();
}

Status BuildRankedMemRefArg(const Tensor& tensor, int expected_rank,
                            std::array<RankedMemRefDescriptor<0>, 1>* rank0,
                            std::array<RankedMemRefDescriptor<1>, 1>* rank1,
                            std::array<RankedMemRefDescriptor<2>, 1>* rank2,
                            std::array<RankedMemRefDescriptor<3>, 1>* rank3,
                            std::array<RankedMemRefDescriptor<4>, 1>* rank4,
                            std::array<RankedMemRefDescriptor<5>, 1>* rank5,
                            std::array<RankedMemRefDescriptor<6>, 1>* rank6,
                            void** arg) {
  switch (expected_rank) {
    case 0:
      TF_RETURN_IF_ERROR(CreateRankedMemRef(tensor, &(*rank0)[0]));
      *arg = &(*rank0)[0];
      return OkStatus();
    case 1:
      TF_RETURN_IF_ERROR(CreateRankedMemRef(tensor, &(*rank1)[0]));
      *arg = &(*rank1)[0];
      return OkStatus();
    case 2:
      TF_RETURN_IF_ERROR(CreateRankedMemRef(tensor, &(*rank2)[0]));
      *arg = &(*rank2)[0];
      return OkStatus();
    case 3:
      TF_RETURN_IF_ERROR(CreateRankedMemRef(tensor, &(*rank3)[0]));
      *arg = &(*rank3)[0];
      return OkStatus();
    case 4:
      TF_RETURN_IF_ERROR(CreateRankedMemRef(tensor, &(*rank4)[0]));
      *arg = &(*rank4)[0];
      return OkStatus();
    case 5:
      TF_RETURN_IF_ERROR(CreateRankedMemRef(tensor, &(*rank5)[0]));
      *arg = &(*rank5)[0];
      return OkStatus();
    case 6:
      TF_RETURN_IF_ERROR(CreateRankedMemRef(tensor, &(*rank6)[0]));
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

Status CallMlirCiface(void* func, const std::vector<void*>& args) {
  switch (args.size()) {
    case 0:
      reinterpret_cast<void (*)()>(func)();
      return OkStatus();
    case 1:
      reinterpret_cast<void (*)(void*)>(func)(args[0]);
      return OkStatus();
    case 2:
      reinterpret_cast<void (*)(void*, void*)>(func)(args[0], args[1]);
      return OkStatus();
    case 3:
      reinterpret_cast<void (*)(void*, void*, void*)>(func)(
          args[0], args[1], args[2]);
      return OkStatus();
    case 4:
      reinterpret_cast<void (*)(void*, void*, void*, void*)>(func)(
          args[0], args[1], args[2], args[3]);
      return OkStatus();
    case 5:
      reinterpret_cast<void (*)(void*, void*, void*, void*, void*)>(func)(
          args[0], args[1], args[2], args[3], args[4]);
      return OkStatus();
    case 6:
      reinterpret_cast<void (*)(void*, void*, void*, void*, void*, void*)>(
          func)(args[0], args[1], args[2], args[3], args[4], args[5]);
      return OkStatus();
    case 7:
      reinterpret_cast<void (*)(
          void*, void*, void*, void*, void*, void*, void*)>(func)(
          args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
      return OkStatus();
    case 8:
      reinterpret_cast<void (*)(
          void*, void*, void*, void*, void*, void*, void*, void*)>(func)(
          args[0], args[1], args[2], args[3], args[4], args[5], args[6],
          args[7]);
      return OkStatus();
    default:
      return errors::Unimplemented(
          "mlir_ciface supports up to 8 memref arguments, got ", args.size());
  }
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
      handle_(nullptr),
      mlir_ciface_func_(nullptr),
      annc_set_current_threadpool_(nullptr),
      annc_get_current_threadpool_(nullptr),
      loaded_(false) {
  OP_REQUIRES_OK(context, context->GetAttr("Nconstants", &num_constants_));
  OP_REQUIRES_OK(context, context->GetAttr("Nfixed", &num_fixed_));
  OP_REQUIRES_OK(context, context->GetAttr("Ndynamic", &num_dynamic_));

  OP_REQUIRES_OK(context, context->GetAttr("kernel_name", &kernel_name_));
  OP_REQUIRES_OK(context, context->GetAttr("num_outputs", &num_outputs_));
  OP_REQUIRES_OK(context, context->GetAttr("output_ranks", &output_ranks_));
  if (context->HasAttr("input_ranks")) {
    OP_REQUIRES_OK(context, context->GetAttr("input_ranks", &input_ranks_));
  }
  if (context->HasAttr("output_shapes")) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("output_shapes", &output_shapes_));
  }
  if (context->HasAttr("kernel_arg_order")) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("kernel_arg_order", &kernel_arg_order_));
  }
  OP_REQUIRES_OK(context, context->GetAttr("dynamic_dims", &dynamic_dims_));
  OP_REQUIRES_OK(context,
                 context->GetAttr("symbolic_signature", &symbolic_signature_str_));
  OP_REQUIRES_OK(context, context->GetAttr("T", &dtype_));

  if (context->HasAttr("shared_lib_path")) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("shared_lib_path", &shared_lib_path_));
  }
  if (context->HasAttr("abi")) {
    OP_REQUIRES_OK(context, context->GetAttr("abi", &abi_));
  } else {
    abi_ = "mlir_ciface";
  }

  OP_REQUIRES(context, !kernel_name_.empty(),
              errors::InvalidArgument("kernel_name cannot be empty"));
  OP_REQUIRES(context, abi_ == "mlir_ciface",
              errors::InvalidArgument("Unsupported ANNCFused abi: ", abi_));
  OP_REQUIRES(context, num_outputs_ == static_cast<int>(output_ranks_.size()),
              errors::InvalidArgument("num_outputs must match output_ranks"));
  OP_REQUIRES(context,
              input_ranks_.empty() ||
                  input_ranks_.size() ==
                      static_cast<size_t>(num_constants_ + num_fixed_ +
                                          num_dynamic_),
              errors::InvalidArgument(
                  "input_ranks must be empty or match total input count"));
  OP_REQUIRES(context,
              output_shapes_.empty() ||
                  output_shapes_.size() == static_cast<size_t>(num_outputs_),
              errors::InvalidArgument(
                  "output_shapes must be empty or match num_outputs"));
}

ANNCFusedOp::~ANNCFusedOp() {}

void ANNCFusedOp::Compute(OpKernelContext* context) {
  OP_REQUIRES(context, !shared_lib_path_.empty(),
              errors::InvalidArgument(
                  "ANNCFused requires shared_lib_path produced by "
                  "ANNCOptimizerPass"));

  if (!loaded_ || current_so_path_ != shared_lib_path_) {
    OP_REQUIRES_OK(context, LoadLibrary(shared_lib_path_));
    current_so_path_ = shared_lib_path_;
  }

  thread::ThreadPool* tf_thread_pool = GetTensorFlowCpuThreadPool(context);
  TensorFlowAnncThreadPool adapter(tf_thread_pool);
  ScopedAnncThreadPool scoped(tf_thread_pool ? &adapter : nullptr,
                              annc_set_current_threadpool_,
                              annc_get_current_threadpool_);
  VLOG(2) << "ANNCFused "
          << (tf_thread_pool && annc_set_current_threadpool_
                  ? "installed TensorFlow CPU threadpool"
                  : "running without TensorFlow CPU threadpool")
          << " for kernel " << kernel_name_;

  OP_REQUIRES_OK(context, ExecuteMlirCifaceKernel(context));
}

Status ANNCFusedOp::LoadLibrary(const std::string& so_path) {
  mutex_lock lock(lib_cache_mu_);

  auto it = lib_cache_.find(so_path);
  if (it != lib_cache_.end()) {
    handle_ = it->second;
    VLOG(2) << "Using cached library handle for " << so_path;
  } else {
    handle_ = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) {
      return errors::NotFound("Cannot load library ", so_path, ": ", dlerror());
    }
    lib_cache_[so_path] = handle_;
    VLOG(1) << "Loaded and cached new library: " << so_path;
  }

  std::string symbol_name = "_mlir_ciface_" + kernel_name_;
  void* symbol = dlsym(handle_, symbol_name.c_str());
  if (!symbol) {
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
    VLOG(1) << "ANNC threadpool TLS symbols were not found in " << so_path
            << "; generated kernel will run without TensorFlow threadpool";
    annc_set_current_threadpool_ = nullptr;
    annc_get_current_threadpool_ = nullptr;
  }

  loaded_ = true;
  return OkStatus();
}

Status ANNCFusedOp::ExecuteMlirCifaceKernel(OpKernelContext* context) {
  if (!mlir_ciface_func_) {
    return errors::FailedPrecondition("MLIR C interface kernel is not loaded");
  }

  const int expected_inputs = num_constants_ + num_fixed_ + num_dynamic_;
  if (context->num_inputs() != expected_inputs) {
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

  for (int i = 0; i < expected_inputs; ++i) {
    const Tensor& tensor = context->input(i);
    int rank = input_ranks_.empty() ? tensor.dims() : input_ranks_[i];
    TF_RETURN_IF_ERROR(BuildRankedMemRefArg(
        tensor, rank, &rank0[i], &rank1[i], &rank2[i], &rank3[i], &rank4[i],
        &rank5[i], &rank6[i], &memrefs[i]));
  }

  for (int i = 0; i < num_outputs_; ++i) {
    Tensor* output = nullptr;

    TensorShape out_shape = InferOutputShape(context, i, output_ranks_[i],
                                             output_shapes_, num_constants_,
                                             num_fixed_);
    TF_RETURN_IF_ERROR(context->allocate_output(i, out_shape, &output));
    output->flat<float>().setZero();

    int memref_index = expected_inputs + i;
    TF_RETURN_IF_ERROR(BuildRankedMemRefArg(
        *output, output_ranks_[i], &rank0[memref_index], &rank1[memref_index],
        &rank2[memref_index], &rank3[memref_index], &rank4[memref_index],
        &rank5[memref_index], &rank6[memref_index], &memrefs[memref_index]));
  }

  std::vector<int> order = kernel_arg_order_;
  if (order.empty()) {
    // Legacy compatibility: the original ANNC pipeline produced kernels with
    // 3 inputs + 1 output (total 4 memrefs) but expected the argument order
    // [dynamic, constant, output, fixed] i.e. {2, 0, 3, 1}.  When
    // input_ranks is not specified and the shape matches this pattern, apply
    // the legacy reordering so that old compiled .so files still work.
    if (expected_inputs == 3 && num_outputs_ == 1 && input_ranks_.empty()) {
      order = {2, 0, 3, 1};
    } else {
      order = BuildDefaultKernelArgOrder(expected_inputs, num_outputs_);
    }
  }

  std::vector<void*> args;
  args.reserve(order.size());
  for (int index : order) {
    if (index < 0 || index >= total_memrefs) {
      return errors::InvalidArgument("kernel_arg_order index ", index,
                                     " is out of range [0, ", total_memrefs,
                                     ")");
    }
    args.push_back(memrefs[index]);
  }

  return CallMlirCiface(mlir_ciface_func_, args);
}

}  // namespace tensorflow
