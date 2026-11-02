// ONNX-side execution context for ANNC kernels.
//
// Reuses the framework-neutral AnncKernelContext ABI. ORT owns tensor buffers,
// allocation and scheduling; all borrowed views live for one synchronous call.
#ifndef ANNC_EP_EXECUTION_H
#define ANNC_EP_EXECUTION_H

#include <onnxruntime_c_api.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "Context/KernelContext.h"
#include "Kernel/ExecutionContext.h"
#include "Kernel/PackedB.h"
#include "Context/ThreadPool.h"
#include "annc_ep_native_pool.h"

namespace annc_ep {

// Declared shape of one site output, in slot order. The sizes are runtime
// values, so only rank and element type can be checked when a kernel allocates.
struct OutputContract {
  int64_t rank = -1;
  AnncElementType element_type = 0;
};

// Carries ORT's thread pool into ANNC's parallel regions. One per Compute
// call, handed to the kernel through the execution context.  The plugin
// always builds the native pool; when the host is the ORT release it was
// verified against, the region is backed by ORT's own operator pool and the
// scheduling decision is the framework's.
class OrtThreadPool final : public annc::threadpool::AnncThreadPool {
 public:
  OrtThreadPool(const OrtApi* api, OrtKernelContext* kernel_context);

  // ORT's C API exposes no pool-size query, so without the native pool this
  // is an advisory upper bound that only gates ANNC's serial/parallel
  // decision (context/lib/Parallel.cpp). The actual degree is ORT's
  // scheduling decision: with intra-op=1 KernelContext_ParallelFor runs the
  // work inline.  With the native pool it is the real degree of parallelism
  // (ThreadPool::DegreeOfParallelism; 1 for a null pool).
  int num_threads() const override;

  // True while this thread is inside a callback this pool dispatched.
  bool in_parallel_region() const override;

  // Maps ANNC's range callback onto the host's parallel dispatch.  With the
  // native pool, a kernel reporting cost_per_unit goes to
  // ThreadPool::TryParallelFor: ORT's own cost model decides serial versus
  // parallel and the block size, exactly as it does for the patched kernels
  // this EP mirrors.  grain_size without a cost maps to ORT's fixed-block
  // entry.  Without the native pool the range is mapped onto
  // KernelContext_ParallelFor's per-index callback, with the batch count
  // sized by ANNC's port of ORT's rules.
  void parallel_for(
      std::int64_t total, const annc::threadpool::ParallelForOptions& options,
      const std::function<void(std::int64_t, std::int64_t)>& fn) override;

  // ORT's operator pool object itself, for kernels that call a host-native
  // library entry with it (MLAS's MlasGemmBatch takes this very object --
  // mlas.h's MLAS_THREADPOOL is a type alias of
  // onnxruntime::concurrency::ThreadPool).  Null when the native path is not
  // active: the pool would still be reachable through the context, but only
  // the version-gated native path vouches for the layout the kernel would
  // touch.
  void* native_pool() const override;

 private:
  const OrtApi* api_;
  OrtKernelContext* kernel_context_;
#if defined(ANNC_EP_NATIVE_POOL)
  // ORT's own operator pool behind kernel_context_, read once per Compute
  // call when the native path is available (annc_ep_native_pool.h).  Null is
  // legal and not a validity flag -- ORT leaves the slot null for an
  // intra-op=1 session, and its native entries treat null as serial --
  // native_enabled_ says whether the path is taken at all.
  onnxruntime::concurrency::ThreadPool* native_ = nullptr;
  bool native_enabled_ = false;
#endif
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
                       const std::vector<OutputContract>& outputs,
                       const std::vector<uint32_t>* output_slots = nullptr);
  ~OnnxExecutionContext();

  OnnxExecutionContext(const OnnxExecutionContext&) = delete;
  OnnxExecutionContext& operator=(const OnnxExecutionContext&) = delete;

  // The ANNC ABI struct handed to the kernel as its only argument.
  AnncKernelContext* kernelAbi() { return &context_; }
  AnncExecutionContext* abi() { return &legacy_; }
  void SetPackedB(const AnncPackedB* packed) { legacy_.packed_b = packed; }


  // Ownership of the returned status (the first error the callbacks recorded)
  // transfers to the caller; nullptr when the call reported no error.
  OrtStatus* TakeError();

  // True when the kernel allocated every declared output slot.
  bool AllOutputsAllocated() const;

 private:
  static AnncStatusCode AllocateOutput(void* opaque, uint32_t slot,
                                       const AnncTensorDesc* request,
                                       AnncTensorView* result);
  static AnncStatusCode AllocateTemp(void* opaque, const AnncTensorDesc* request,
                                     AnncTensorView* result);
  static void ReportError(void* opaque, AnncStatusCode status,
                          const char* message);
  static AnncStatusCode GetStatus(void* opaque);
  static int32_t NumThreads(void* opaque);
  static void* NativePool(void* opaque, const char* provider);
  static AnncStatusCode ParallelFor(void* opaque, int64_t total, int64_t grain,
                                    AnncContextTask task, void* task_data);
  static AnncStatusCode Shard(void* opaque, int32_t max_parallelism,
                              int64_t total, int64_t cost,
                              AnncContextTask task, void* task_data);
  AnncStatusCode Dispatch(int64_t total,
                           const annc::threadpool::ParallelForOptions& options,
                           AnncContextTask task, void* task_data);
  bool BindView(const AnncTensorDesc& request, void* data,
                AnncTensorView* result);

  AnncStatusCode AllocateOutputImpl(uint32_t slot, AnncTensorDesc* desc, bool legacy = false);
  AnncStatusCode AllocateTempImpl(AnncTensorDesc* desc);
  // Stores the first error and reports it to stderr once.
  AnncStatusCode Fail(AnncStatusCode status, const std::string& message);
  // Keeps `status` as the run's error when none was recorded yet; later
  // statuses are released instead of overwriting (and leaking) the first one.
  void RecordStatus(OrtStatus* status);

  const OrtApi* api_;
  OrtKernelContext* kernel_context_;
  // The request is immutable and outlives this synchronous context. Borrowing
  // it avoids copying the small contract vectors once per fused site.
  const std::vector<OutputContract>* outputs_;
  const std::vector<uint32_t>* output_slots_;
  std::vector<bool> allocated_;
  OrtStatus* error_ = nullptr;
  OrtMemoryInfo* mem_info_ = nullptr;       // lazy, released in the destructor
  OrtAllocator* temp_allocator_ = nullptr;  // lazy, released once
  std::vector<void*> temps_;                // freed before the allocator
  OrtThreadPool pool_;
  struct ShapeStorage {
    std::array<int64_t, ANNC_CONTEXT_MAX_RANK> sizes{}, strides{};
  };
  std::deque<ShapeStorage> shapes_;
  std::vector<AnncTensorView> output_views_;
  std::recursive_mutex mutex_;
  std::atomic<AnncStatusCode> status_{ANNC_STATUS_OK};
  AnncRuntimeServices services_{};
  AnncKernelContext context_{};
  AnncExecutionContext legacy_{};
};

std::int32_t CallContextKernel(void*, const std::vector<void*>&);

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
  std::string abi = "annc_execution_v2";
  const AnncPackedB* packed_b = nullptr;
  // Kernel call order: the rank and element type of each site argument (args
  // in the report).
  std::vector<int64_t> arg_ranks;
  std::vector<AnncElementType> arg_types;
  // ORT input index each argument resolves to; same order as arg_ranks.
  std::vector<int64_t> arg_indices;
  std::vector<OutputContract> outputs;
  std::vector<int64_t> i64_attrs;
  // Runtime fusion metadata: process-local resources borrowed from this
  // node's compute state. Never written back to the serialized Site report.
  std::vector<AnncResourceView> resources;
  // ORT output index each declared output slot maps to; empty = identity.
  std::vector<uint32_t> output_slots;
};

// Runs one context-ABI site: builds borrowed tensor/attribute views,
// invokes the kernel and returns an OrtStatus when the kernel, a callback or
// the output contract failed (ownership transfers to the caller). Nullptr on
// success.
OrtStatus* ComputeWithContext(const OrtApi* api,
                              OrtKernelContext* kernel_context,
                              const ContextCall& request);

}  // namespace annc_ep

#endif  // ANNC_EP_EXECUTION_H
