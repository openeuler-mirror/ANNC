// ONNX-side execution context for ANNC kernels.
//
// Mirrors what tensorflow_addons/annc_fused_op.cc does for TensorFlow --
// output/temp allocation and error reporting through the ANNC execution ABI --
// with one deliberate difference: the host thread pool travels *inside* the
// context (ABI v3) instead of being injected into the kernel library's TLS.
// The kernel specs read it through annc::kernels::execution::resolveThreadPool.
//
// One OnnxExecutionContext is constructed per Compute call and must not
// outlive it: the ORT kernel context, the pool and every allocation belong to
// that single synchronous call.
#ifndef ANNC_EP_EXECUTION_H
#define ANNC_EP_EXECUTION_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <onnxruntime_c_api.h>

#include "Kernel/ExecutionContext.h"
#include "Support/ThreadPool/ThreadPool.h"

namespace annc_ep {

// Declared shape of one site output, in slot order. The sizes are runtime
// values, so only rank and element type can be checked when a kernel allocates.
struct OutputContract {
  int64_t rank = -1;
  AnncElementType element_type = 0;
};

// Carries ORT's thread pool into ANNC's parallel regions. One per Compute
// call, handed to the kernel through the execution context.
class OrtThreadPool final : public annc::threadpool::AnncThreadPool {
 public:
  OrtThreadPool(const OrtApi* api, OrtKernelContext* kernel_context)
      : api_(api), kernel_context_(kernel_context) {}

  // ORT's C API exposes no pool-size query, so this is an advisory upper bound
  // that only gates ANNC's serial/parallel decision
  // (Support/ThreadPool/Parallel.cpp). The actual degree is ORT's scheduling
  // decision: with intra-op=1 KernelContext_ParallelFor runs the work inline.
  int num_threads() const override;

  // True while this thread is inside a callback this pool dispatched.
  bool in_parallel_region() const override;

  // Maps ANNC's range callback onto KernelContext_ParallelFor's per-index
  // callback.  cost_per_unit is fed through ORT's own rule -- Eigen's cost
  // model, the one ThreadPool::TryParallelFor applies inside ORT's kernels --
  // to size the batch count, with grain_size as the fixed-block alternative
  // for kernels that report no cost.
  void parallel_for(
      std::int64_t total, const annc::threadpool::ParallelForOptions& options,
      const std::function<void(std::int64_t, std::int64_t)>& fn) override;

 private:
  const OrtApi* api_;
  OrtKernelContext* kernel_context_;
};

// The per-call context plus its state.
class OnnxExecutionContext {
 public:
  // `output_slots` maps the site's output order (the slot a kernel asks for)
  // onto the fused node's output order (what ORT numbers slots by); empty means
  // the two agree. The site's order is the pattern's, ORT's is the partition's,
  // so they only agree by luck -- the EP resolves the mapping by tensor name at
  // Compile time.
  OnnxExecutionContext(const OrtApi* api, OrtKernelContext* kernel_context,
                       std::vector<OutputContract> outputs,
                       std::vector<uint32_t> output_slots = {});
  ~OnnxExecutionContext();

  OnnxExecutionContext(const OnnxExecutionContext&) = delete;
  OnnxExecutionContext& operator=(const OnnxExecutionContext&) = delete;

  // The ANNC ABI struct handed to the kernel as its first argument.
  AnncExecutionContext* abi() { return &context_; }

  // Ownership of the returned status (the first error the callbacks recorded)
  // transfers to the caller; nullptr when the call reported no error.
  OrtStatus* TakeError();

  // True when the kernel allocated every declared output slot.
  bool AllOutputsAllocated() const;

 private:
  // The ABI's execution handle is opaque; this is the host payload behind it,
  // with a stable address for the lifetime of the context.
  struct HandlePayload {
    OnnxExecutionContext* state = nullptr;
  };

  static OnnxExecutionContext* StateOf(AnncExecutionHandle* handle);
  static AnncStatusCode AllocateOutput(AnncExecutionHandle* handle,
                                       uint32_t slot, AnncTensorDesc* desc);
  static AnncStatusCode AllocateTemp(AnncExecutionHandle* handle,
                                     AnncTensorDesc* desc);
  static void ReportError(AnncExecutionHandle* handle, AnncStatusCode status,
                          const char* message);

  AnncStatusCode AllocateOutputImpl(uint32_t slot, AnncTensorDesc* desc);
  AnncStatusCode AllocateTempImpl(AnncTensorDesc* desc);
  // Stores the first error and reports it to stderr once.
  AnncStatusCode Fail(AnncStatusCode status, const std::string& message);
  // Keeps `status` as the run's error when none was recorded yet; later
  // statuses are released instead of overwriting (and leaking) the first one.
  void RecordStatus(OrtStatus* status);

  const OrtApi* api_;
  OrtKernelContext* kernel_context_;
  std::vector<OutputContract> outputs_;
  std::vector<uint32_t> output_slots_;
  std::vector<bool> allocated_;
  OrtStatus* error_ = nullptr;
  OrtMemoryInfo* mem_info_ = nullptr;       // lazy, released in the destructor
  OrtAllocator* temp_allocator_ = nullptr;  // lazy, released once
  std::vector<void*> temps_;                // freed before the allocator
  OrtThreadPool pool_;
  HandlePayload handle_payload_;
  AnncExecutionHandle* handle_ = nullptr;
  AnncExecutionContext context_{};
};

// Calls a context-ABI kernel entry (`_mlir_ciface_<name>`, returning an i32
// status) with args[0] = the execution context and one memref descriptor per
// following entry. Returns the kernel's status code, or -1 for an unsupported
// argument count.
std::int32_t CallContextKernel(void* symbol, const std::vector<void*>& args);

// Maps an ORT tensor element type onto the ANNC element type the ABI uses;
// false when the type has no ANNC counterpart (such a tensor can never fill a
// declared argument slot).
bool AnncElementFromOrt(ONNXTensorElementDataType type, AnncElementType* out);

// Rank and element type of one fused-node input, as the argument resolver
// sees it.
struct InputShapeKey {
  int64_t rank = 0;
  AnncElementType type = 0;
};

// What a site's arguments resolve to among the fused node's inputs.
struct ArgumentBinding {
  // One entry per argument, in contract order: the input index it binds to,
  // or -1 when no input fits it.
  std::vector<int64_t> indices;
  // The first argument that two inputs fit, and the two inputs involved
  // (-1 = none). Such a binding must be refused: the contract cannot say
  // which tensor the kernel should read, and first-match would pick silently.
  int64_t ambiguous_arg = -1;
  int64_t ambiguous_inputs[2] = {-1, -1};
};

// Binds each declared argument to the input whose rank and element type match
// it. Fused nodes carry more inputs than the kernel takes (the region's shape
// chain exposes its constants too), so inputs fitting no argument are
// ignored; an argument left unfilled stays -1 for the caller to report.
ArgumentBinding BindArgumentsToInputs(
    const std::vector<InputShapeKey>& inputs,
    const std::vector<int64_t>& arg_ranks,
    const std::vector<AnncElementType>& arg_types);

// One context-ABI site execution request, resolved against the fused node's
// inputs.
struct ContextCall {
  void* symbol = nullptr;
  // Kernel call order: the rank and element type of each site argument (args
  // in the report).
  std::vector<int64_t> arg_ranks;
  std::vector<AnncElementType> arg_types;
  // ORT input index each argument resolves to; same order as arg_ranks.
  std::vector<int64_t> arg_indices;
  std::vector<OutputContract> outputs;
  // ORT output index each declared output slot maps to; empty = identity.
  std::vector<uint32_t> output_slots;
};

// Runs one context-ABI site: builds one ranked-memref descriptor per argument,
// invokes the kernel and returns an OrtStatus when the kernel, a callback or
// the output contract failed (ownership transfers to the caller). Nullptr on
// success.
OrtStatus* ComputeWithContext(const OrtApi* api,
                              OrtKernelContext* kernel_context,
                              const ContextCall& request);

}  // namespace annc_ep

#endif  // ANNC_EP_EXECUTION_H
