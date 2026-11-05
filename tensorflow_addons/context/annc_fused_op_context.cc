#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

#include "Context/KernelContext.h"
#include "Context/ThreadPool.h"
#include "annc_fused_op.h"
#include "ThreadPool.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {

AnncElementType ContextElementType(DataType type) {
  switch (type) {
    case DT_FLOAT:
      return ANNC_ELEMENT_TYPE_F32;
    case DT_STRING:
      return ANNC_ELEMENT_TYPE_STRING;
    case DT_INT32:
      return ANNC_ELEMENT_TYPE_I32;
    case DT_INT64:
      return ANNC_ELEMENT_TYPE_I64;
    default:
      return 0;
  }
}

DataType ContextDataType(AnncElementType type) {
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

// Each owner has a stable address even when the owner vector grows. Neither
// allocation requests nor kernel stack metadata are retained by the adapter.
struct ContextTensorOwner {
  Tensor temp;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  AnncTensorView view{};
  std::vector<AnncStringRef> strings;

  Status Bind(const Tensor& value, bool read_only) {
    const AnncElementType type = ContextElementType(value.dtype());
    if (!type || value.dims() > ANNC_CONTEXT_MAX_RANK) {
      return errors::InvalidArgument(
          "annc_context_v3 requires numeric or string tensors of rank 0-8");
    }
    sizes.resize(value.dims());
    strides.resize(value.dims());
    int64_t stride = 1;
    for (int i = value.dims() - 1; i >= 0; --i) {
      sizes[i] = value.dim_size(i);
      strides[i] = stride;
      if (sizes[i] && stride > std::numeric_limits<int64_t>::max() / sizes[i]) {
        return errors::InvalidArgument("ANNC tensor stride overflow");
      }
      stride *= sizes[i];
    }
    void* data = nullptr;
    if (value.dtype() == DT_STRING) {
      auto values = value.flat<tstring>();
      strings.resize(values.size());
      for (int64_t i = 0; i < values.size(); ++i)
        strings[i] = {values(i).data(), static_cast<int64_t>(values(i).size())};
      data = strings.data();
    } else {
      data = const_cast<char*>(value.tensor_data().data());
    }
    view = {data, data, 0, sizes.data(), strides.data(), type,
            static_cast<uint32_t>(sizes.size()), ANNC_LAYOUT_STRIDED,
            ANNC_TENSOR_BOUND | (read_only ? ANNC_TENSOR_READ_ONLY : 0u)};
    return OkStatus();
  }
};

struct ContextExecutionState {
  OpKernelContext* context;
  const std::vector<DataType>& output_types;
  const std::vector<int>& output_ranks;
  annc::threadpool::AnncThreadPool* thread_pool;
  std::vector<std::unique_ptr<ContextTensorOwner>> outputs;
  std::vector<std::unique_ptr<ContextTensorOwner>> temps;
  std::mutex error_mu;
  std::mutex allocation_mu;
  Status error;
  AnncStatusCode code = ANNC_STATUS_OK;

  AnncStatusCode Fail(AnncStatusCode status, const Status& failure) {
    std::lock_guard<std::mutex> lock(error_mu);
    if (code == ANNC_STATUS_OK) {
      code = status;
      error = failure;
    }
    return code;
  }

  AnncStatusCode GetStatus() {
    std::lock_guard<std::mutex> lock(error_mu);
    return code;
  }

  Status Error() {
    std::lock_guard<std::mutex> lock(error_mu);
    return error;
  }
};

Status AllocationShape(const AnncTensorDesc* request, TensorShape* shape,
                       DataType* dtype) {
  if (!request || request->struct_size < sizeof(*request) ||
      request->rank > ANNC_CONTEXT_MAX_RANK ||
      (request->rank && !request->dims) ||
      (request->flags & ~ANNC_ALLOCATION_ZERO)) {
    return errors::InvalidArgument("Invalid ANNC allocation request");
  }
  *dtype = ContextDataType(request->element_type);
  if (*dtype == DT_INVALID) {
    return errors::InvalidArgument("Unsupported ANNC allocation dtype");
  }
  // Check before AddDim: TensorShape's unchecked mutator can CHECK on overflow.
  int64_t elements = 1;
  for (uint32_t i = 0; i < request->rank; ++i) {
    const int64_t dim = request->dims[i];
    if (dim < 0 ||
        (dim && elements > std::numeric_limits<int64_t>::max() / dim)) {
      return errors::InvalidArgument("Invalid ANNC allocation dimensions");
    }
    elements *= dim;
  }
  const int64_t element_bytes = *dtype == DT_INT64 ? 8 : 4;
  if (elements > std::numeric_limits<int64_t>::max() / element_bytes) {
    return errors::InvalidArgument("ANNC allocation byte size overflow");
  }
  for (uint32_t i = 0; i < request->rank; ++i) {
    shape->AddDim(request->dims[i]);
  }
  return OkStatus();
}

AnncStatusCode AllocateContextTensor(void* opaque, uint32_t slot,
                                     const AnncTensorDesc* request,
                                     AnncTensorView* result, bool output) {
  auto* state = static_cast<ContextExecutionState*>(opaque);
  if (!state) return ANNC_STATUS_INVALID_ARGUMENT;
  if (state->GetStatus() != ANNC_STATUS_OK) return state->GetStatus();
  try {
    std::lock_guard<std::mutex> allocation_lock(state->allocation_mu);
    if (!result) {
      return state->Fail(ANNC_STATUS_INVALID_ARGUMENT,
                         errors::InvalidArgument("Missing allocation result"));
    }
    TensorShape shape;
    DataType dtype;
    Status status = AllocationShape(request, &shape, &dtype);
    if (!status.ok()) return state->Fail(ANNC_STATUS_INVALID_ARGUMENT, status);
    if (output &&
        (slot >= state->outputs.size() || state->outputs[slot] ||
         state->output_types[slot] != dtype ||
         state->output_ranks[slot] != static_cast<int>(request->rank))) {
      return state->Fail(
          ANNC_STATUS_INVALID_ARGUMENT,
          errors::InvalidArgument("ANNC output slot ", slot,
                                  " is invalid, already allocated, or has a "
                                  "dtype/rank mismatch"));
    }
    auto owner = std::make_unique<ContextTensorOwner>();
    Tensor* tensor = nullptr;
    if (output) {
      status = state->context->allocate_output(slot, shape, &tensor);
    } else {
      status = state->context->allocate_temp(dtype, shape, &owner->temp);
      tensor = &owner->temp;
    }
    if (!status.ok()) return state->Fail(ANNC_STATUS_ALLOCATION_FAILED, status);
    status = owner->Bind(*tensor, false);
    if (!status.ok()) return state->Fail(ANNC_STATUS_INVALID_ARGUMENT, status);
    if (request->flags & ANNC_ALLOCATION_ZERO) {
      const auto bytes = tensor->tensor_data().size();
      if (bytes) std::memset(owner->view.aligned, 0, bytes);
    }
    *result = owner->view;
    if (output) {
      state->outputs[slot] = std::move(owner);
    } else {
      state->temps.push_back(std::move(owner));
    }
    return ANNC_STATUS_OK;
  } catch (const std::bad_alloc&) {
    return state->Fail(ANNC_STATUS_ALLOCATION_FAILED,
                       errors::ResourceExhausted("ANNC allocation failed"));
  } catch (const std::exception& error) {
    return state->Fail(ANNC_STATUS_INTERNAL, errors::Internal(error.what()));
  } catch (...) {
    return state->Fail(ANNC_STATUS_INTERNAL,
                       errors::Internal("ANNC allocation callback failed"));
  }
}

AnncStatusCode AllocateContextOutput(void* opaque, uint32_t slot,
                                     const AnncTensorDesc* request,
                                     AnncTensorView* result) {
  return AllocateContextTensor(opaque, slot, request, result, true);
}

AnncStatusCode AllocateContextTemp(void* opaque, const AnncTensorDesc* request,
                                   AnncTensorView* result) {
  return AllocateContextTensor(opaque, 0, request, result, false);
}

void ReportContextError(void* opaque, AnncStatusCode status,
                        const char* message) {
  auto* state = static_cast<ContextExecutionState*>(opaque);
  if (!state || status == ANNC_STATUS_OK) return;
  // Status owns its message, including messages supplied on a kernel stack.
  if (status == ANNC_STATUS_INVALID_ARGUMENT) {
    state->Fail(status, errors::InvalidArgument(message ? message : ""));
  } else if (status == ANNC_STATUS_ALLOCATION_FAILED) {
    state->Fail(status, errors::ResourceExhausted(message ? message : ""));
  } else {
    state->Fail(status, errors::Internal(message ? message : ""));
  }
}

AnncStatusCode ContextGetStatus(void* opaque) {
  auto* state = static_cast<ContextExecutionState*>(opaque);
  return state ? state->GetStatus() : ANNC_STATUS_INVALID_ARGUMENT;
}

AnncStatusCode RunContextParallel(
    void* opaque, int64_t total,
    const annc::threadpool::ParallelForOptions& options,
    AnncContextTask callback, void* task) {
  auto* state = static_cast<ContextExecutionState*>(opaque);
  if (!state) return ANNC_STATUS_INVALID_ARGUMENT;
  if (state->GetStatus() != ANNC_STATUS_OK) return state->GetStatus();
  if (!callback || total < 0) {
    return state->Fail(ANNC_STATUS_INVALID_ARGUMENT,
                       errors::InvalidArgument("Invalid ANNC parallel range"));
  }
  try {
    if (total == 0) return ANNC_STATUS_OK;
    auto invoke = [&](int64_t begin, int64_t end) {
      try {
        callback(task, begin, end);
      } catch (const std::exception& error) {
        state->Fail(ANNC_STATUS_INTERNAL, errors::Internal(error.what()));
      } catch (...) {
        state->Fail(ANNC_STATUS_INTERNAL,
                    errors::Internal("ANNC parallel task failed"));
      }
    };
    if (!state->thread_pool || state->thread_pool->in_parallel_region()) {
      invoke(0, total);
    } else {
      state->thread_pool->parallel_for(total, options, invoke);
    }
    return state->GetStatus();
  } catch (const std::exception& error) {
    return state->Fail(ANNC_STATUS_INTERNAL, errors::Internal(error.what()));
  } catch (...) {
    return state->Fail(ANNC_STATUS_INTERNAL,
                       errors::Internal("ANNC parallel service failed"));
  }
}

AnncStatusCode ContextParallelFor(void* opaque, int64_t total,
                                  int64_t grain_size, AnncContextTask callback,
                                  void* task) {
  if (grain_size < 0) {
    ReportContextError(opaque, ANNC_STATUS_INVALID_ARGUMENT,
                       "Invalid ANNC parallel grain size");
    return ANNC_STATUS_INVALID_ARGUMENT;
  }
  annc::threadpool::ParallelForOptions options;
  if (grain_size > 0) options.grain_size = grain_size;
  return RunContextParallel(opaque, total, options, callback, task);
}

AnncStatusCode ContextShard(void* opaque, int32_t max_parallelism,
                            int64_t total, int64_t cost_per_unit,
                            AnncContextTask callback, void* task) {
  if (max_parallelism < 1 || cost_per_unit < 0) {
    ReportContextError(opaque, ANNC_STATUS_INVALID_ARGUMENT,
                       "Invalid ANNC shard scheduling options");
    return ANNC_STATUS_INVALID_ARGUMENT;
  }
  annc::threadpool::ParallelForOptions options;
  options.max_parallelism = max_parallelism;
  options.cost_per_unit = cost_per_unit;
  return RunContextParallel(opaque, total, options, callback, task);
}

int32_t ContextNumThreads(void* opaque) {
  auto* state = static_cast<ContextExecutionState*>(opaque);
  return state && state->thread_pool
             ? std::max(state->thread_pool->num_threads(), 1)
             : 1;
}

}  // namespace

Status ANNCFusedOp::ExecuteContextV3Kernel(OpKernelContext* context,
                                           void* kernel_function) {
  if (!kernel_function) {
    return errors::FailedPrecondition("Context V3 kernel symbol is not loaded");
  }
  const int num_inputs = num_constants_ + num_fixed_ + num_dynamic_;
  if (context->num_inputs() != num_inputs ||
      input_types_.size() != static_cast<size_t>(num_inputs) ||
      context->num_outputs() != num_outputs_ ||
      output_types_.size() != static_cast<size_t>(num_outputs_)) {
    return errors::InvalidArgument("annc_context_v3 input/output count mismatch");
  }
  try {
    // The adapter borrows TensorFlow's pool for this synchronous invocation.
    annc_context::TensorFlowAnncThreadPool thread_pool(
        annc_context::GetTensorFlowCpuThreadPool(context));
    std::vector<ContextTensorOwner> inputs(num_inputs);
    std::vector<AnncTensorView> input_views(num_inputs);
    std::vector<AnncTensorView> output_views(num_outputs_);
    // Validate input contracts at the TF boundary. Output dtype/rank contracts
    // are checked by AllocateContextOutput; kernels receive only tensor views.
    for (int i = 0; i < num_inputs; ++i) {
      const int transport =
          context_input_order_.empty() ? i : context_input_order_[i];
      const Tensor& tensor = context->input(transport);
      if ((!input_ranks_.empty() &&
           input_ranks_[transport] != tensor.dims()) ||
          input_types_[transport] != tensor.dtype()) {
        return errors::InvalidArgument("ANNC input ", transport,
                                       " dtype/rank mismatch");
      }
      TF_RETURN_IF_ERROR(inputs[i].Bind(tensor, true));
      input_views[i] = inputs[i].view;
    }
    ContextExecutionState state{
        context, output_types_, output_ranks_, &thread_pool, {}, {}, {},
        {},      OkStatus()};
    state.outputs.resize(num_outputs_);
    AnncRuntimeServices services{sizeof(AnncRuntimeServices),
                                 ANNC_CONTEXT_ABI_VERSION,
                                 &state,
                                 &AllocateContextOutput,
                                 &AllocateContextTemp,
                                 &ReportContextError,
                                 &ContextGetStatus,
                                 &ContextParallelFor,
                                 &ContextNumThreads,
                                 &ContextShard,
                                 nullptr};
    std::vector<AnncAttributeView> attributes;
    attributes.reserve(context_i64_attrs_.size());
    for (const int64_t& value : context_i64_attrs_)
      attributes.push_back({ANNC_ATTRIBUTE_I64, 0, &value, 1});
    AnncKernelContext execution{sizeof(AnncKernelContext),
                                ANNC_CONTEXT_ABI_VERSION,
                                input_views.data(),
                                static_cast<uint32_t>(num_inputs),
                                output_views.data(),
                                static_cast<uint32_t>(num_outputs_),
                                attributes.data(),
                                static_cast<uint32_t>(attributes.size()),
                                nullptr,
                                0,
                                &services};
    const AnncStatusCode status =
        reinterpret_cast<AnncKernelFn>(kernel_function)(&execution);
    TF_RETURN_IF_ERROR(state.Error());
    if (status != ANNC_STATUS_OK) {
      const std::string message = "annc_context_v3 kernel returned status " +
                                  std::to_string(status);
      if (status == ANNC_STATUS_INVALID_ARGUMENT) {
        return errors::InvalidArgument(message);
      }
      if (status == ANNC_STATUS_ALLOCATION_FAILED) {
        return errors::ResourceExhausted(message);
      }
      return errors::Internal(message);
    }
    for (int i = 0; i < num_outputs_; ++i) {
      if (!state.outputs[i] ||
          !(output_views[i].flags & ANNC_TENSOR_BOUND) ||
          output_views[i].allocated != state.outputs[i]->view.allocated ||
          output_views[i].aligned != state.outputs[i]->view.aligned ||
          output_views[i].offset != 0 ||
          output_views[i].element_type != state.outputs[i]->view.element_type ||
          output_views[i].rank != state.outputs[i]->view.rank ||
          output_views[i].sizes != state.outputs[i]->view.sizes ||
          output_views[i].strides != state.outputs[i]->view.strides ||
          output_views[i].layout != ANNC_LAYOUT_STRIDED ||
          (output_views[i].flags & ANNC_TENSOR_READ_ONLY)) {
        return errors::Internal("annc_context_v3 did not bind output ", i,
                                " to its TensorFlow allocation");
      }
    }
    return OkStatus();
  } catch (const std::bad_alloc&) {
    return errors::ResourceExhausted("ANNC context setup allocation failed");
  } catch (const std::exception& error) {
    return errors::Internal("ANNC context execution failed: ", error.what());
  } catch (...) {
    return errors::Internal("ANNC context execution failed");
  }
}

}  // namespace tensorflow
