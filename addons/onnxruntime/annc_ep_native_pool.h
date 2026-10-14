// ANNC EP -- ORT's native operator thread pool.
//
// The plugin C API exposes no pool: OrtKernelContext is opaque, and the only
// parallel entry, KernelContext_ParallelFor, hands the *caller* the sizing
// decision (see include/onnxruntime/core/platform/threadpool.h's Pitfall
// note) while hiding the pool itself.  ORT's own kernels call
// ThreadPool::TryParallelFor instead: Eigen's cost model decides
// serial-versus-parallel and the block size from the pool's real degree of
// parallelism.  Backing AnncThreadPool with that pool moves the scheduling
// decision back to where ORT's kernels make it; the alternative --
// re-implementing Eigen's model against an estimated pool size -- was
// measured to mis-size the dispatch whenever the estimate is wrong (17
// shards submitted to a two-thread pool, and 17 callbacks even when the pool
// is null: task 20261013-019, analysis 20261014, native-pool-20261014
// probe).
//
// This header is a locked-version private ABI, not a public contract:
//
//  * OrtKernelContext* is ORT's internal OpKernelContext* -- the plugin EP
//    bridge (onnxruntime/core/session/plugin_ep/
//    ep_plugin_provider_interfaces.cc) hands it over with exactly that
//    reinterpret_cast, and OrtApis::KernelContext_ParallelFor reverses it.
//  * threadpool_ sits at slot 3 (offset 24): OpKernelContext is
//    single-inheritance, so vptr at 0, then execution_frame_, kernel_,
//    threadpool_ (include/onnxruntime/core/framework/op_kernel_context.h;
//    no #if guards on those members).  Verified against the 1.26.0 build by
//    disassembling KernelContext_ParallelFor (native-pool-20261014).
//  * The members declared below are the non-inline ones: each resolves to a
//    symbol in ORT's build-tree libonnxruntime_common.a (whose CPUIDInfo
//    dependency needs libcpuinfo.a).  Those archives must come from the same
//    build as the hosting libonnxruntime.so -- an install/SDK tree does not
//    carry them, and mixing builds would duplicate ORT's thread-pool state.
//
// PoolAvailable() refuses the path when the host is not the expected ORT
// release (or the operator set ANNC_EP_NATIVE_POOL=0), and the adapter falls
// back to the public KernelContext_ParallelFor path.  When the ORT version
// moves, re-verify the slot and the archive closure and bump
// ANNC_EP_NATIVE_POOL_ORT_VERSION together with the build tree.
#ifndef ANNC_EP_NATIVE_POOL_H
#define ANNC_EP_NATIVE_POOL_H

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>

#include <onnxruntime_c_api.h>

#ifndef ANNC_EP_NATIVE_POOL_ORT_VERSION
#define ANNC_EP_NATIVE_POOL_ORT_VERSION "1.26.0"
#endif

// Mirror of ORT's declarations (include/onnxruntime/core/platform/
// threadpool.h), in ORT's own namespace so the calls bind to ORT's symbols
// instead of compiling a copy.  Only non-inline members appear: inline
// members would instantiate ANNC's own implementation, which is exactly
// what this header exists to avoid.
namespace onnxruntime {
struct TensorOpCost {
  double bytes_loaded;
  double bytes_stored;
  double compute_cycles;
};
namespace concurrency {
class ThreadPool {
 public:
  // Degree of parallelism the pool's own users see (NumThreads() + 1, or
  // that times the hybrid-core factor); 1 for a null pool.
  static int DegreeOfParallelism(const ThreadPool* tp);
  // ORT's cost-model dispatch: runs fn(0, total) on the caller below the
  // parallelisation threshold, otherwise hands out
  // CalculateParallelForBlock-sized ranges.  A null pool is serial by
  // definition.
  static void TryParallelFor(
      ThreadPool* tp, std::ptrdiff_t total, const TensorOpCost& cost_per_unit,
      const std::function<void(std::ptrdiff_t, std::ptrdiff_t)>& fn);
  // Fixed blocks of `block_size` units; total <= block_size runs on the
  // caller.
  void ParallelForFixedBlockSizeScheduling(
      std::ptrdiff_t total, std::ptrdiff_t block_size,
      const std::function<void(std::ptrdiff_t, std::ptrdiff_t)>& fn);
};
}  // namespace concurrency
}  // namespace onnxruntime

namespace annc_ep {
namespace native {

// OpKernelContext::threadpool_ slot, in pointer units (see the header
// comment for the layout).
constexpr int kThreadPoolSlot = 3;

inline onnxruntime::concurrency::ThreadPool* PoolOf(
    const OrtKernelContext* kernel_context) {
  if (kernel_context == nullptr) return nullptr;
  return static_cast<onnxruntime::concurrency::ThreadPool*>(const_cast<void*>(
      reinterpret_cast<void* const*>(kernel_context)[kThreadPoolSlot]));
}

// Whether the host identified itself as the expected ORT release.  Set once
// at plugin load by CheckHostVersion(), which runs in CreateEpFactories --
// the only place the host's OrtApiBase (whose version query OrtApi itself
// lacks) is in hand -- and therefore always before the first PoolAvailable().
inline bool& HostVersionMatches() {
  static bool matches = false;
  return matches;
}

inline void CheckHostVersion(const OrtApiBase* api_base) {
  bool matches = false;
  if (api_base != nullptr && api_base->GetVersionString != nullptr) {
    const char* version = api_base->GetVersionString();
    matches = version != nullptr &&
              std::strcmp(version, ANNC_EP_NATIVE_POOL_ORT_VERSION) == 0;
  }
  HostVersionMatches() = matches;
}

// Whether the native path may run in this host process: the host version
// check passed and the operator did not disable it with
// ANNC_EP_NATIVE_POOL=0.  Evaluated once per process.
inline bool PoolAvailable() {
  static const bool available = [] {
    if (!HostVersionMatches()) return false;
    const char* disabled = std::getenv("ANNC_EP_NATIVE_POOL");
    if (disabled != nullptr && std::strcmp(disabled, "0") == 0) return false;
    return true;
  }();
  return available;
}

}  // namespace native
}  // namespace annc_ep

#endif  // ANNC_EP_NATIVE_POOL_H
