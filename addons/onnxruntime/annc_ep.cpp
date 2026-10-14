// annc_ep.cpp — ANNC execution-provider plugin for ONNX Runtime
//
// Runtime-compile EP (the only mode): at session init the EP exports the graph
// ORT hands it, asks the ANNC toolchain (`annc-onnx-pipeline`) which sites it
// can fuse there, claims exactly those nodes, and runs them with the kernels
// that same compile produced. No per-model offline artefacts: the toolchain
// produces `sites.json` + its kernel library into the EP cache on demand.
//
// Loaded with RegisterExecutionProviderLibrary + SessionOptionsAppendExecutionProvider_V2;
// exports CreateEpFactories and ReleaseEpFactory (onnxruntime_ep_c_api.h:2558,2582).
//
// Environment:
//   ANNC_EP_PIPELINE    abs path to annc-onnx-pipeline (default: found next to
//                       this library, then ../bin, then PATH)
//   ANNC_EP_CACHE_DIR   compile-cache root (default: $XDG_CACHE_HOME/annc-ep,
//                       then $HOME/.cache/annc-ep)
//   ANNC_EP_DEBUG=1     compile diagnostics + per-node operand shapes on stderr
//   ANNC_EP_TIMING=1    per-node cost split (input / output / kernel) at exit
//
// Scope:
//   * the ANNC toolchain decides the fusion boundaries (no matcher is replayed
//     inside the EP); a site's nodes come from the report's member list, and
//     AddNodesToFuse is handed the whole member set, never a partial block.
//   * kernels are compiled on first use and cached on disk by model signature;
//     one dlopen at first use, one dlsym per fused node, both cached. No
//     EPContext, no offline manifest.
//
// Predicate comments in the ANNC fusion passes name the ORT patch function
// whose rule was ported; the matcher itself lives in ANNC (OpFusion.cpp).
#include <onnxruntime_c_api.h>
// onnxruntime_ep_c_api.h is included by onnxruntime_c_api.h and has no include
// guard of its own -- including it a second time redefines every struct in it.

#include "annc_ep_compile.h"
#include "annc_ep_execution.h"

#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

const OrtApi* g_api = nullptr;
const OrtEpApi* g_ep_api = nullptr;
bool g_debug = false;  // ANNC_EP_DEBUG=1

// ---------------------------------------------------------------------------
// Kernel resolution.
//
// The ANNC link step puts every site of one model into ONE shared library,
// each site exporting its own `_mlir_ciface_<kernel_name>` (measured: 27 sites
// -> 27 symbols). A site's kernel name is derived on the ANNC side from the
// ONNX node that anchors the fusion (sha of the anchor + pattern), so the EP
// cannot guess it from shapes -- it reads it from the compile report
// (`sites.json`) that the runtime compile just produced for this very model.
// ---------------------------------------------------------------------------
constexpr const char* kKernelPrefix = "_mlir_ciface_";

std::string CacheToolchainFingerprint(annc_ep::CompileDriver* driver) {
  std::string fingerprint = driver->ToolchainInfo();
  if (fingerprint.empty()) return {};

  // The pipeline curates its child environment. These variables select
  // matchers, compiler/linker inputs, headers, and runtime search paths, so a
  // changed value must not reuse a previously compiled cache entry.
  constexpr const char* config_names[] = {
      "ANNC_EP_PIPELINE",       "ANNC_GEMM_CONFIG",    "ANNC_GEMM",
      "ANNC_KERNEL_LIB_PATH",   "ANNC_CLANG",          "LIBRARY_PATH",
      "CPATH",                  "C_INCLUDE_PATH",      "CPLUS_INCLUDE_PATH",
  };
  fingerprint += "\nannc-ep-config-v1";
  for (const char* name : config_names) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
      fingerprint += "\n";
      fingerprint += name;
      fingerprint += "=unset";
      continue;
    }
    fingerprint += "\n";
    fingerprint += name;
    fingerprint += "=" + std::to_string(std::strlen(value)) + ":";
    fingerprint += value;
  }

  // The kernel archives themselves are *inputs* to the link step, and the
  // pipeline reads them from ANNC_KERNEL_LIB_PATH -- whose value above is only
  // the directory name.  Without their identity in the key, rebuilding a
  // kernel leaves every cached entry valid and the EP keeps running the old
  // library (hit while porting masked_mul: the kernel-side grain change was
  // invisible until the caches were wiped).  Size+mtime of every archive in
  // that directory is enough to catch a rebuild; hashing contents would mean
  // reading hundreds of megabytes per session.
  if (const char* lib_dir = std::getenv("ANNC_KERNEL_LIB_PATH")) {
    std::vector<std::string> entries;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(lib_dir, error)) {
      if (error) break;
      if (entry.path().extension() != ".a") continue;
      std::error_code stat_error;
      const auto size = std::filesystem::file_size(entry.path(), stat_error);
      if (stat_error) continue;
      const auto stamp = std::filesystem::last_write_time(entry.path(), stat_error);
      if (stat_error) continue;
      entries.push_back(entry.path().filename().string() + ":" +
                        std::to_string(size) + ":" +
                        std::to_string(
                            static_cast<int64_t>(stamp.time_since_epoch().count())));
    }
    std::sort(entries.begin(), entries.end());
    fingerprint += "\nannc-ep-kernel-libs-v1";
    for (const std::string& entry : entries) {
      fingerprint += "\n";
      fingerprint += entry;
    }
  }
  return fingerprint;
}

struct KernelHandle {
  void* handle = nullptr;
  // Raw symbol, used by the context-ABI call path, which dispatches on the
  // site's argument count.
  void* symbol = nullptr;
  // The legacy mlir_ciface signature, used by the preallocated-output path.
  void (*fn)(void*, void*, void*) = nullptr;
};

// Keyed by (library path, kernel name) rather than by kernel name alone: the
// runtime-compile mode hands every model its own library, and two models in one
// process must not resolve into each other's symbols (the previous process-wide
// table keyed only by name could not tell them apart).
class KernelTable {
 public:
  // Returns the kernel for `kernel_name` (the ANNC-side function name, without
  // the ciface prefix) in `library`, or nullptr if either is missing.
  KernelHandle* Get(const std::string& library, const std::string& kernel_name) {
    if (library.empty() || kernel_name.empty()) return nullptr;
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string key = library + "\n" + kernel_name;
    auto it = table_.find(key);
    if (it != table_.end()) return it->second.handle ? &it->second : nullptr;
    KernelHandle handle;
    if (!Load(library, kernel_name, &handle)) {
      table_[key] = KernelHandle{};  // negative cache
      return nullptr;
    }
    return &table_.emplace(key, handle).first->second;
  }

 private:
  bool Load(const std::string& library, const std::string& kernel_name,
            KernelHandle* out) {
    // One dlopen per library: each holds every site's kernel of one model.
    auto it = libraries_.find(library);
    void* shared = nullptr;
    if (it == libraries_.end()) {
      shared = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (shared == nullptr) {
        std::fprintf(stderr, "[annc-ep] dlopen(%s) failed: %s\n", library.c_str(), dlerror());
      }
      libraries_[library] = shared;  // nullptr memoized: no retry per site
    } else {
      shared = it->second;
    }
    if (shared == nullptr) return false;
    const std::string symbol = kKernelPrefix + kernel_name;
    out->symbol = dlsym(shared, symbol.c_str());
    if (!out->symbol) {
      std::fprintf(stderr, "[annc-ep] no symbol %s in %s\n", symbol.c_str(), library.c_str());
      return false;
    }
    out->fn = reinterpret_cast<void (*)(void*, void*, void*)>(out->symbol);
    out->handle = shared;
    return true;
  }

  std::mutex mutex_;
  std::unordered_map<std::string, void*> libraries_;  // path -> dlopen handle
  std::unordered_map<std::string, KernelHandle> table_;
};

KernelTable& Kernels() {
  static KernelTable table;
  return table;
}


// ---------------------------------------------------------------------------
// Graph helpers
// ---------------------------------------------------------------------------
std::string NodeNameOf(const OrtNode* node) {
  const char* name = nullptr;
  (void)g_api->Node_GetName(node, &name);
  return name ? name : "<unnamed>";
}


std::vector<const OrtValueInfo*> NodeOutputs(const OrtNode* node) {
  size_t count = 0;
  if (g_api->Node_GetNumOutputs(node, &count) != nullptr || count == 0) return {};
  std::vector<const OrtValueInfo*> outputs(count, nullptr);
  if (g_api->Node_GetOutputs(node, outputs.data(), count) != nullptr) return {};
  return outputs;
}

// The graph's own name for a value (empty when ORT has none). This is the
// original model's tensor name -- exactly what the site report's names are
// translated back to at claim time.
std::string ValueNameOf(const OrtValueInfo* value) {
  if (value == nullptr) return {};
  const char* name = nullptr;
  (void)g_api->GetValueInfoName(value, &name);
  return name ? name : "";
}

std::vector<std::string> NodeInputNames(const OrtNode* node) {
  size_t count = 0;
  if (node == nullptr ||
      g_api->Node_GetNumInputs(node, &count) != nullptr) {
    return {};
  }
  std::vector<const OrtValueInfo*> inputs(count, nullptr);
  if (count != 0 &&
      g_api->Node_GetInputs(node, inputs.data(), count) != nullptr) {
    return {};
  }
  std::vector<std::string> names;
  names.reserve(count);
  for (const OrtValueInfo* value : inputs) names.push_back(ValueNameOf(value));
  return names;
}

std::vector<std::string> NodeOutputNames(const OrtNode* node) {
  std::vector<const OrtValueInfo*> outputs = NodeOutputs(node);
  std::vector<std::string> names;
  names.reserve(outputs.size());
  for (const OrtValueInfo* value : outputs) names.push_back(ValueNameOf(value));
  return names;
}

// The report names the exported model's tensors ("t<i>_<p>" / "in_<k>" /
// "w_<k>"); Compile() sees the original graph. Translating at claim time (the
// only place with the rename table) lets Compile() bind kernel arguments and
// outputs by the graph's own names -- the only way to tell apart arguments the
// rank/type rule cannot separate. Outputs additionally carry the result index
// the report appends ("t178_0:0") and the exported names never contain ':', so
// that suffix is dropped when the full name is not in the table.
// Returns the translated copy, or std::nullopt when a name cannot be resolved.
std::optional<annc_ep::Site> TranslatedSite(const annc_ep::Site& site,
                                            const annc_ep::NameMaps& maps) {
  auto lookup = [&](const std::string& name) -> const std::string* {
    auto it = maps.renamed_to_tensor.find(name);
    if (it != maps.renamed_to_tensor.end()) return &it->second;
    const size_t colon = name.rfind(':');
    if (colon == std::string::npos || colon + 1 == name.size()) return nullptr;
    for (size_t i = colon + 1; i < name.size(); ++i) {
      if (!std::isdigit(static_cast<unsigned char>(name[i]))) return nullptr;
    }
    it = maps.renamed_to_tensor.find(name.substr(0, colon));
    return it == maps.renamed_to_tensor.end() ? nullptr : &it->second;
  };
  annc_ep::Site translated = site;
  for (annc_ep::Site::Arg& arg : translated.args) {
    const std::string* original = lookup(arg.name);
    if (original == nullptr) return std::nullopt;
    arg.name = *original;
  }
  for (annc_ep::Site::Arg& output : translated.outputs) {
    const std::string* original = lookup(output.name);
    if (original == nullptr) return std::nullopt;
    output.name = *original;
  }
  return translated;
}



// ---------------------------------------------------------------------------
// Compute state
// ---------------------------------------------------------------------------
struct ComputeState {
  void (*fn)(void*, void*, void*) = nullptr;  // resolved once in CreateState
  // Context-ABI sites (annc_execution_v2): the raw kernel entry and the site's
  // contract -- argument ranks/types in kernel call order and the declared
  // output slots the callbacks validate against.
  void* symbol = nullptr;
  std::string abi;
  std::vector<int64_t> arg_ranks;
  std::vector<AnncElementType> arg_types;
  std::vector<annc_ep::OutputContract> outputs;
  // Input index each context argument resolves to. Name-resolved sites carry
  // theirs from Compile; the rest are filled on the first call, guarded by
  // `context_mutex` because ORT may run the same node concurrently.
  std::mutex context_mutex;
  std::vector<int64_t> arg_indices;
  // Declared output slot -> the fused node's output index (empty = identity).
  std::vector<uint32_t> output_slots;
  // The fused node's input layout is fixed for the lifetime of the session, so
  // the operand scan only has to run once; afterwards the two indices are used
  // directly.
  int64_t x_index = -1, w_index = -1;
  bool reported = false;  // ANNC_EP_DEBUG: print this node's operand shapes once
};

// ---------------------------------------------------------------------------
// Optional per-node timing (ANNC_EP_TIMING=1): splits this implementation's own
// cost (operand identification, output allocation, packing) from the ANNC
// kernel call, so the接入 overhead is a measured number rather than a residual.
// Counters are atomic: ORT runs the 27 independent fused nodes on intra-op
// threads.
// ---------------------------------------------------------------------------
struct NodeTiming {
  std::atomic<int64_t> calls{0};
  std::atomic<int64_t> input_ns{0};   // identify x/W + read their shapes
  std::atomic<int64_t> output_ns{0};  // KernelContext_GetOutput + data pointers
  std::atomic<int64_t> kernel_ns{0};  // the ANNC kernel itself
  // Shape of the first call, so a slow kernel can be pinned to the operands it
  // was actually given (the report's shapes are symbolic).
  std::atomic<int64_t> first_d0{0}, first_d1{0}, first_k{0}, first_n{0};
  std::atomic<bool> have_shape{false};

  void NoteShape(int64_t d0, int64_t d1, int64_t k, int64_t n) {
    bool expected = false;
    if (have_shape.compare_exchange_strong(expected, true)) {
      first_d0.store(d0);
      first_d1.store(d1);
      first_k.store(k);
      first_n.store(n);
    }
  }

  // Context-ABI sites: the EP's own view of one Compute call, entry to return.
  // Comparing this with the profiler's per-node time splits the cost into
  // "ORT around the node" versus "the EP (prepare + kernel)".
  std::atomic<int64_t> context_calls{0};
  std::atomic<int64_t> context_ns{0};
  std::atomic<int64_t> min_context_ns{INT64_MAX};

  void NoteContextCall(int64_t nanos) {
    context_calls.fetch_add(1);
    context_ns.fetch_add(nanos);
    int64_t current = min_context_ns.load();
    while (nanos < current &&
           !min_context_ns.compare_exchange_weak(current, nanos)) {
    }
  }

  ~NodeTiming() {
    if (context_calls.load() != 0) {
      const int64_t calls = context_calls.load();
      std::fprintf(stderr,
                   "[annc-ep] context ComputeImpl over %lld calls (mean/min "
                   "us): total=%.3f/%.3f\n",
                   static_cast<long long>(calls),
                   context_ns.load() / static_cast<double>(calls) / 1000.0,
                   min_context_ns.load() / 1000.0);
    }
    const int64_t n = calls.load();
    if (n == 0) return;
    const double per = 1.0 / static_cast<double>(n);
    const double in_us = input_ns.load() * per / 1000.0;
    const double out_us = output_ns.load() * per / 1000.0;
    const double ker_us = kernel_ns.load() * per / 1000.0;
    std::fprintf(stderr,
                 "[annc-ep] timing over %lld calls (per call): input=%.3fus "
                 "output=%.3fus kernel=%.3fus overhead=%.3fus total=%.3fus "
                 "first-shape=[%lld,%lld,%lld]x[%lld,%lld]\n",
                 static_cast<long long>(n), in_us, out_us, ker_us,
                 in_us + out_us, in_us + out_us + ker_us,
                 static_cast<long long>(first_d0.load()),
                 static_cast<long long>(first_d1.load()),
                 static_cast<long long>(first_k.load()),
                 static_cast<long long>(first_k.load()),
                 static_cast<long long>(first_n.load()));
  }
};

NodeTiming& Timing() {
  static NodeTiming timing;
  return timing;
}

bool TimingEnabled() {
  static const bool enabled = std::getenv("ANNC_EP_TIMING") != nullptr;
  return enabled;
}

struct MemRef3D {
  float *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
struct MemRef2D {
  float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};

struct AnncNodeComputeInfo : OrtNodeComputeInfo {
  explicit AnncNodeComputeInfo(std::string name) : fused_name(std::move(name)) {}
  std::string fused_name;   // ORT's name for the fused node (diagnostics only)
  std::string anchor_name;  // node producing the site's output (diagnostics only)
  std::string kernel_name;  // resolved ANNC kernel, empty if unresolved
  std::string library;      // shared library holding the kernel
  // Context-ABI site contract, copied from the report at Compile time.
  std::string abi;
  std::vector<int64_t> arg_ranks;
  std::vector<AnncElementType> arg_types;
  std::vector<annc_ep::OutputContract> outputs;
  // Sites whose arguments or outputs the rank rule cannot identify get their
  // mapping resolved here, by name (empty = resolve at first Compute).
  std::vector<int64_t> arg_indices;
  // Declared output slot -> the fused node's output index (empty = identity).
  std::vector<uint32_t> output_slots;
};

// ---------------------------------------------------------------------------
// Context-ABI execution (annc_execution_v2 sites)
//
// The kernel receives the execution context (which carries the ORT thread
// pool) plus one ranked-memref descriptor per site argument, and allocates its
// own outputs through the context callbacks. Each argument resolves to the
// fused-node input whose rank and element type match the contract; distinct
// argument ranks (enforced at claim time) make that a unique fit, and a
// second fit for the same argument fails closed instead of picking one. The
// mapping is cached on the first call (under the
// state's mutex, because ORT may run the same node concurrently) since the
// fused node's input list is fixed for the session's lifetime.
// ---------------------------------------------------------------------------
OrtStatus* ResolveContextInputs(ComputeState* state,
                                OrtKernelContext* kernel_context) {
  size_t num_inputs = 0;
  if (OrtStatus* status =
          g_api->KernelContext_GetInputCount(kernel_context, &num_inputs)) {
    return status;
  }
  // FillContextContract pushes the contract's argument ranks and types in
  // pairs; keep the diagnostics below from indexing a shorter type vector if
  // that ever stops being true (BindArgumentsToInputs itself is total).
  if (state->arg_ranks.size() != state->arg_types.size()) {
    return g_api->CreateStatus(
        ORT_FAIL,
        "ANNC EP: the site contract's argument ranks and types are not in "
        "step");
  }
  // Fused nodes legitimately carry more inputs than the kernel takes: the
  // region's shape chain exposes its constants alongside the operands (the
  // legacy path scans for its x/w the same way). The binding rule (rank +
  // element type; extras ignored; an argument two inputs fit is refused) lives
  // in BindArgumentsToInputs, unit-tested against the real site's input list.
  std::vector<annc_ep::InputShapeKey> inputs;
  inputs.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    const OrtValue* value = nullptr;
    if (OrtStatus* status =
            g_api->KernelContext_GetInput(kernel_context, i, &value)) {
      return status;
    }
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (OrtStatus* status = g_api->GetTensorTypeAndShape(value, &info)) {
      return status;
    }
    size_t rank = 0;
    if (OrtStatus* status = g_api->GetDimensionsCount(info, &rank)) {
      g_api->ReleaseTensorTypeAndShapeInfo(info);
      return status;
    }
    ONNXTensorElementDataType ort_type =
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    if (OrtStatus* status = g_api->GetTensorElementType(info, &ort_type)) {
      g_api->ReleaseTensorTypeAndShapeInfo(info);
      return status;
    }
    g_api->ReleaseTensorTypeAndShapeInfo(info);
    annc_ep::InputShapeKey key;
    key.rank = static_cast<int64_t>(rank);
    // 0 = no ANNC counterpart (ANNC_ELEMENT_TYPE_* start at 1), which fits no
    // declared argument.
    (void)annc_ep::AnncElementFromOrt(ort_type, &key.type);
    inputs.push_back(key);
  }
  const annc_ep::ArgumentBinding binding = annc_ep::BindArgumentsToInputs(
      inputs, state->arg_ranks, state->arg_types);
  if (g_debug) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      int64_t bound_to = -1;
      for (size_t arg = 0; arg < binding.indices.size(); ++arg) {
        if (binding.indices[arg] == static_cast<int64_t>(i)) {
          bound_to = static_cast<int64_t>(arg);
        }
      }
      if (bound_to >= 0) {
        std::fprintf(stderr,
                     "[annc-ep] ctx input %zu (rank %lld, type %d) -> kernel "
                     "argument %lld\n",
                     i, static_cast<long long>(inputs[i].rank),
                     static_cast<int>(inputs[i].type),
                     static_cast<long long>(bound_to));
      } else {
        std::fprintf(stderr,
                     "[annc-ep] ctx input %zu (rank %lld, type %d): fits no "
                     "argument; ignored\n",
                     i, static_cast<long long>(inputs[i].rank),
                     static_cast<int>(inputs[i].type));
      }
    }
  }
  if (binding.ambiguous_arg >= 0) {
    return g_api->CreateStatus(
        ORT_FAIL,
        ("ANNC EP: fused-node inputs " +
         std::to_string(binding.ambiguous_inputs[0]) + " and " +
         std::to_string(binding.ambiguous_inputs[1]) +
         " both fit kernel argument " +
         std::to_string(binding.ambiguous_arg))
            .c_str());
  }
  // Resolve into a local vector and publish it only on success: a failed
  // resolution must leave the cache empty, otherwise the next call would take
  // the `arg_indices` fast path with -1 entries and fail forever.
  for (size_t arg = 0; arg < binding.indices.size(); ++arg) {
    if (binding.indices[arg] < 0) {
      return g_api->CreateStatus(
          ORT_FAIL, ("ANNC EP: no input matches kernel argument " +
                     std::to_string(arg) + " (rank " +
                     std::to_string(state->arg_ranks[arg]) + ", type " +
                     std::to_string(static_cast<int>(state->arg_types[arg])) +
                     ")")
                        .c_str());
    }
  }
  state->arg_indices = binding.indices;
  return nullptr;
}

OrtStatus* ComputeContextSite(ComputeState* state,
                              OrtKernelContext* kernel_context) {
  if (state->symbol == nullptr) {
    return g_api->CreateStatus(ORT_FAIL,
                               "ANNC EP: context kernel is not loaded");
  }
  const bool timing = std::getenv("ANNC_EP_TIMING") != nullptr;
  const auto t_entry = timing ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
  annc_ep::ContextCall call;
  call.symbol = state->symbol;
  call.arg_ranks = state->arg_ranks;
  call.arg_types = state->arg_types;
  call.outputs = state->outputs;
  call.output_slots = state->output_slots;
  {
    std::lock_guard<std::mutex> lock(state->context_mutex);
    if (state->arg_indices.empty()) {
      if (OrtStatus* status = ResolveContextInputs(state, kernel_context)) {
        return status;
      }
    }
    call.arg_indices = state->arg_indices;
  }
  // Name-resolved sites arrive with `arg_indices` already pinned, so they never
  // reach ResolveContextInputs and its per-input log. Without a line here an
  // executing site would look identical to one ORT dropped -- print the
  // resolved mapping, once per node.
  if (g_debug && !state->reported) {
    state->reported = true;
    std::fprintf(stderr, "[annc-ep] ctx compute: abi=%s, %zu arg(s) ->",
                 state->abi.c_str(), state->arg_ranks.size());
    for (size_t arg = 0; arg < state->arg_ranks.size(); ++arg) {
      const int64_t input =
          arg < state->arg_indices.size() ? state->arg_indices[arg] : -1;
      std::fprintf(stderr, " [arg %zu (rank %lld) <- input %lld]", arg,
                   static_cast<long long>(state->arg_ranks[arg]),
                   static_cast<long long>(input));
    }
    std::fprintf(stderr, ", %zu output slot(s)\n", state->outputs.size());
  }
  OrtStatus* status = annc_ep::ComputeWithContext(g_api, kernel_context, call);
  if (timing) {
    const auto t_end = std::chrono::steady_clock::now();
    Timing().NoteContextCall(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_entry)
            .count());
  }
  return status;
}

OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                        OrtNodeComputeContext* /*context*/,
                                        void** compute_state) noexcept {
  auto* state = new ComputeState();
  auto* info = static_cast<AnncNodeComputeInfo*>(this_ptr);
  if (!info->kernel_name.empty()) {
    KernelHandle* handle = Kernels().Get(info->library, info->kernel_name);
    if (handle != nullptr) {
      state->fn = handle->fn;
      state->symbol = handle->symbol;
    }
  }
  state->abi = info->abi;
  state->arg_ranks = info->arg_ranks;
  state->arg_types = info->arg_types;
  state->outputs = info->outputs;
  state->output_slots = info->output_slots;
  // Name-resolved sites arrive with their argument mapping already fixed; the
  // rest resolve on the first Compute by rank and element type.
  state->arg_indices = info->arg_indices;
  *compute_state = state;
  return nullptr;
}

void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* /*this_ptr*/, void* compute_state) noexcept {
  delete static_cast<ComputeState*>(compute_state);
}

OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                    OrtKernelContext* kernel_context) noexcept {
  auto* state = static_cast<ComputeState*>(compute_state);
  auto* node_info = static_cast<AnncNodeComputeInfo*>(this_ptr);

  // ANNC_EP_NOOP=1 is a measurement aid, not a mode: it keeps ORT's whole
  // plugin-EP path (node dispatch, OrtKernelContext creation, output
  // allocation) and drops only the ANNC kernel invocation and packing. The
  // difference against a normal run is this implementation's own cost; what
  // remains is what the EP charges per node.
  static const bool noop = std::getenv("ANNC_EP_NOOP") != nullptr;
  if (noop) {
    size_t input_count = 0;
    if (OrtStatus* status = g_api->KernelContext_GetInputCount(kernel_context, &input_count)) {
      return status;
    }
    const int64_t dims[3] = {1, 1, 1};
    OrtValue* out = nullptr;
    return g_api->KernelContext_GetOutput(kernel_context, 0, dims, 3, &out);
  }

  if (state->abi == "annc_execution_v2") {
    return ComputeContextSite(state, kernel_context);
  }

  size_t num_inputs = 0;
  if (OrtStatus* status = g_api->KernelContext_GetInputCount(kernel_context, &num_inputs)) {
    return status;
  }
  const bool timing = TimingEnabled();
  const auto t_input_start = timing ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};

  // Identify the operands by rank rather than by position: the fused node's
  // inputs are (transpose_out, flat_shape, W) in graph order, and the shape
  // tensor must be ignored. The result is cached in the compute state.
  const OrtValue* x = nullptr;
  const OrtValue* w = nullptr;
  int64_t d0 = 0, d1 = 0, k = 0, n = 0;

  if (state->x_index >= 0 && state->w_index >= 0) {
    // Fast path: shape and kernel were resolved on a previous call.
    if (OrtStatus* status =
            g_api->KernelContext_GetInput(kernel_context, state->x_index, &x)) {
      return status;
    }
    if (OrtStatus* status =
            g_api->KernelContext_GetInput(kernel_context, state->w_index, &w)) {
      return status;
    }
    OrtTensorTypeAndShapeInfo* x_info = nullptr;
    if (OrtStatus* status = g_api->GetTensorTypeAndShape(x, &x_info)) return status;
    int64_t x_dims[3] = {0, 0, 0};
    size_t rank = 0;
    (void)g_api->GetDimensionsCount(x_info, &rank);
    if (rank == 3) (void)g_api->GetDimensions(x_info, x_dims, 3);
    g_api->ReleaseTensorTypeAndShapeInfo(x_info);
    d0 = x_dims[0];
    d1 = x_dims[1];
    k = x_dims[2];

    OrtTensorTypeAndShapeInfo* w_info = nullptr;
    if (OrtStatus* status = g_api->GetTensorTypeAndShape(w, &w_info)) return status;
    int64_t w_dims[2] = {0, 0};
    size_t w_rank = 0;
    (void)g_api->GetDimensionsCount(w_info, &w_rank);
    if (w_rank == 2) (void)g_api->GetDimensions(w_info, w_dims, 2);
    g_api->ReleaseTensorTypeAndShapeInfo(w_info);
    n = w_dims[1];
  }

  // Scan until BOTH operands are found: W may precede x in the fused input list.
  for (size_t i = 0; (x == nullptr || w == nullptr) && i < num_inputs; ++i) {
    const OrtValue* value = nullptr;
    if (OrtStatus* status = g_api->KernelContext_GetInput(kernel_context, i, &value)) {
      return status;
    }
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (OrtStatus* status = g_api->GetTensorTypeAndShape(value, &info)) return status;
    size_t rank = 0;
    (void)g_api->GetDimensionsCount(info, &rank);
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    (void)g_api->GetTensorElementType(info, &type);
    const bool is_float = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    if (is_float && rank == 3 && x == nullptr) {
      int64_t dims[3] = {0, 0, 0};
      if (g_api->GetDimensions(info, dims, 3) == nullptr) {
        d0 = dims[0];
        d1 = dims[1];
        k = dims[2];
        x = value;
        state->x_index = static_cast<int64_t>(i);
      }
    } else if (is_float && rank == 2 && w == nullptr) {
      int64_t dims[2] = {0, 0};
      if (g_api->GetDimensions(info, dims, 2) == nullptr) {
        n = dims[1];
        w = value;
        state->w_index = static_cast<int64_t>(i);
      }
    }
    g_api->ReleaseTensorTypeAndShapeInfo(info);
  }
  if (x == nullptr || w == nullptr || d0 <= 0 || d1 <= 0 || k <= 0 || n <= 0) {
    return g_api->CreateStatus(ORT_FAIL, "ANNC EP: could not identify x/W inputs");
  }

  // The kernel was resolved and cached in CreateState (per fused node, from the
  // report the runtime compile wrote for this session). A miss means the
  // session's site table and the compiled library disagree.
  if (state->fn == nullptr) {
    return g_api->CreateStatus(ORT_FAIL,
                               "ANNC EP: kernel for this fused node did not "
                               "resolve from the compiled library");
  }
  void (*fn)(void*, void*, void*) = state->fn;
  const auto t_output_start = timing ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};

  const int64_t out_dims[3] = {d0, d1, n};
  OrtValue* out = nullptr;
  if (OrtStatus* status =
          g_api->KernelContext_GetOutput(kernel_context, 0, out_dims, 3, &out)) {
    return status;
  }

  const void* x_raw = nullptr;
  const void* w_raw = nullptr;
  void* out_raw = nullptr;
  if (OrtStatus* status = g_api->GetTensorData(x, &x_raw)) return status;
  if (OrtStatus* status = g_api->GetTensorData(w, &w_raw)) return status;
  if (OrtStatus* status = g_api->GetTensorMutableData(out, &out_raw)) return status;

  auto* x_data = const_cast<float*>(static_cast<const float*>(x_raw));
  auto* w_data = const_cast<float*>(static_cast<const float*>(w_raw));
  auto* y_data = static_cast<float*>(out_raw);

  MemRef3D x_ref{x_data, x_data, 0, {d0, d1, k}, {d1 * k, k, 1}};
  MemRef2D w_ref{w_data, w_data, 0, {k, n}, {n, 1}};
  MemRef3D y_ref{y_data, y_data, 0, {d0, d1, n}, {d1 * n, n, 1}};

  if (g_debug && !state->reported) {
    state->reported = true;
    std::fprintf(stderr,
                 "[annc-ep] site '%s' kernel='%s': m=%lld k=%lld n=%lld "
                 "(macs=%lld)\n",
                 node_info->fused_name.c_str(), node_info->kernel_name.c_str(),
                 static_cast<long long>(d0 * d1), static_cast<long long>(k),
                 static_cast<long long>(n),
                 static_cast<long long>(d0) * d1 * k * n);
  }
  const auto t_kernel_start = timing ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
  fn(&x_ref, &w_ref, &y_ref);
  if (timing) {
    const auto t_end = std::chrono::steady_clock::now();
    NodeTiming& t = Timing();
    t.calls.fetch_add(1);
    t.NoteShape(d0, d1, k, n);
    t.input_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                             t_output_start - t_input_start).count());
    t.output_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                              t_kernel_start - t_output_start).count());
    t.kernel_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                              t_end - t_kernel_start).count());
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// EP
// ---------------------------------------------------------------------------
// A site this session claimed, plus the identity of its nodes. The identity is
// the ORT node id, not a name: the names the report carries live in the
// exported model's namespace, while Compile() sees the original graph.
struct ClaimedSite {
  annc_ep::Site site;
  std::vector<size_t> node_ids;
};

struct AnncEp : OrtEp {
  explicit AnncEp(std::string name) : name(std::move(name)) {}
  std::string name;

  // Runtime-compile session state. `sites` holds the sites this session
  // claimed; `kernel_library` is the shared library they were compiled into.
  // Both stay empty when the session claimed nothing (zero-site model or a
  // compile that failed and fell back to the CPU EP).
  std::vector<ClaimedSite> sites;
  std::string kernel_library;
  mutable std::mutex state_mutex;

  // Returns the site whose fused region is exactly `node_ids`, or nullptr.
  const annc_ep::Site* FindSiteByNodeIds(const std::vector<size_t>& node_ids) const {
    for (const ClaimedSite& claimed : sites) {
      if (claimed.node_ids.size() != node_ids.size()) continue;
      bool all_present = true;
      for (size_t id : claimed.node_ids) {
        bool found = false;
        for (size_t candidate : node_ids) {
          if (candidate == id) {
            found = true;
            break;
          }
        }
        if (!found) {
          all_present = false;
          break;
        }
      }
      if (all_present) return &claimed.site;
    }
    return nullptr;
  }
};

const char* ORT_API_CALL EpGetName(const OrtEp* this_ptr) noexcept {
  return static_cast<const AnncEp*>(this_ptr)->name.c_str();
}


// Runtime-compile path (the default): the EP does no pattern matching at
// all. It exports the graph ORT hands it, asks the ANNC toolchain which sites
// it can fuse there, and claims exactly those. Every step that can fail runs
// before the first AddNodesToFuse, because a failure in Compile() fails the
// whole session while a node left unclaimed simply stays on the CPU EP.
OrtStatus* ORT_API_CALL GetCapabilityRuntimeCompileImpl(
    OrtEp* this_ptr, const OrtGraph* graph,
    OrtEpGraphSupportInfo* graph_support_info) noexcept {
  auto* ep = static_cast<AnncEp*>(this_ptr);
  std::lock_guard<std::mutex> ep_state_lock(ep->state_mutex);

  // Control-flow bodies arrive here too, and a nested graph cannot be exported
  // as a standalone model (outer-scope values, missing initializers).
  const OrtNode* parent = nullptr;
  if (g_api->Graph_GetParentNode(graph, &parent) != nullptr || parent != nullptr) {
    return nullptr;
  }

  // ORT may ask an EP to evaluate capability for more than one top-level graph
  // during session construction. Do not retain claims or a library from a
  // prior graph, otherwise Compile() can match stale node ids. The lock keeps
  // concurrent capability calls from racing on this per-session state.
  ep->sites.clear();
  ep->kernel_library.clear();

  // Keep driver state local to this capability evaluation. The disk cache and
  // per-signature flock provide sharing without exposing mutable driver state
  // to concurrent ORT capability calls.
  annc_ep::CompileDriver driver;
  if (!driver.Available()) {
    static std::once_flag warned;
    std::call_once(warned, [&driver]() {
      std::fprintf(stderr, "[annc-ep] runtime compile disabled: %s\n",
                   driver.UnavailableReason().c_str());
    });
    return nullptr;
  }

  // Graph_GetNodes returns topological order; OrtGraphToProto walks the graph
  // the same way, so exported node "n<i>" is nodes[i].
  size_t num_nodes = 0;
  if (g_api->Graph_GetNumNodes(graph, &num_nodes) != nullptr) return nullptr;
  std::vector<const OrtNode*> nodes(num_nodes, nullptr);
  if (num_nodes > 0 && g_api->Graph_GetNodes(graph, nodes.data(), num_nodes) != nullptr) {
    return nullptr;
  }

  std::string error;
  const std::string model_path = annc_ep::ProcessTempDir() + "/model.onnx";
  annc_ep::NameMaps maps;
  if (!annc_ep::ExportGraphToModel(graph, model_path, &maps, &error)) {
    std::fprintf(stderr, "[annc-ep] graph export failed: %s\n", error.c_str());
    return nullptr;
  }

  const std::string toolchain = CacheToolchainFingerprint(&driver);
  if (toolchain.empty()) {
    std::fprintf(stderr, "[annc-ep] cannot fingerprint the ANNC toolchain\n");
    return nullptr;
  }
  const std::string signature = annc_ep::ModelSignature(model_path, toolchain, &error);
  if (signature.empty()) {
    std::fprintf(stderr, "[annc-ep] model signature failed: %s\n", error.c_str());
    return nullptr;
  }

  std::string library;
  annc_ep::SiteReport report;
  if (!driver.Compile(model_path, signature, &library, &report, &error)) {
    std::fprintf(stderr, "[annc-ep] runtime compile failed: %s\n", error.c_str());
    return nullptr;
  }

  OrtNodeFusionOptions fusion_options = {};
  fusion_options.ort_version_supported = ORT_API_VERSION;
  fusion_options.drop_constant_initializers = false;  // keep W as an input

  std::unordered_set<std::string> claimed_names;
  size_t claimed = 0, dropped = 0;
  for (const annc_ep::Site& reported : report.sites) {
    // Translate the report's names back to the graph's own names (the report
    // speaks the exported model's). A site whose names are not all in the
    // table can still be claimed when the rank/type rule identifies its
    // arguments on its own; one that needs names is dropped here rather than
    // failing later at Compile (which can only fail the whole session).
    annc_ep::Site site = reported;
    site.namesResolved = false;
    if (auto translated = TranslatedSite(reported, maps)) {
      site = *translated;
      site.namesResolved = true;
    }
    if (annc_ep::SiteNeedsResolvedNames(site) && !site.namesResolved) {
      ++dropped;
      std::fprintf(stderr,
                   "[annc-ep] drop site '%s': its argument/output names are "
                   "not in the exported model's rename table\n",
                   site.kernelName.c_str());
      continue;
    }
    std::vector<const OrtNode*> to_fuse;
    bool valid = true;
    for (const std::string& member : site.memberNodes) {
      auto it = maps.node_index_by_name.find(member);
      if (it == maps.node_index_by_name.end() ||
          static_cast<size_t>(it->second) >= nodes.size()) {
        valid = false;  // a member the EP cannot resolve: drop this site only
        break;
      }
      // Sites must not overlap, and a member must not already be claimed.
      if (!claimed_names.insert(member).second) {
        valid = false;
        break;
      }
      to_fuse.push_back(nodes[static_cast<size_t>(it->second)]);
    }
    if (!valid || to_fuse.empty()) {
      ++dropped;
      if (g_debug) {
        std::fprintf(stderr, "[annc-ep] drop site '%s': members not resolvable\n",
                     site.kernelName.c_str());
      }
      continue;
    }
    // Validate the kernel before claiming: after a successful dlopen/dlsym the
    // mapping survives file deletion, so Compile() can no longer fail on it.
    if (Kernels().Get(library, site.kernelName) == nullptr) {
      ++dropped;
      continue;
    }
    // Record the claimed region by node id: Compile() receives the original
    // graph, where the exported names do not exist.
    std::vector<size_t> node_ids;
    node_ids.reserve(to_fuse.size());
    bool ids_resolved = true;
    for (const OrtNode* node : to_fuse) {
      size_t id = 0;
      if (g_api->Node_GetId(node, &id) != nullptr) {
        ids_resolved = false;
        break;
      }
      node_ids.push_back(id);
    }
    if (!ids_resolved) {
      ++dropped;
      continue;
    }
    if (OrtStatus* status = g_ep_api->EpGraphSupportInfo_AddNodesToFuse(
            graph_support_info, to_fuse.data(), to_fuse.size(), &fusion_options)) {
      // One invalid grouping voids every grouping ORT received, and the EP has
      // no way to observe that from here; report it and let the caller log it.
      return status;
    }
    ep->sites.push_back(ClaimedSite{site, std::move(node_ids)});
    ++claimed;
  }
  if (claimed > 0) ep->kernel_library = library;

  if (g_debug) {
    std::fprintf(stderr,
                 "[annc-ep] GetCapability(runtime-compile): nodes=%zu sites=%zu claimed=%zu "
                 "dropped=%zu signature=%s cache=%s\n",
                 num_nodes, report.sites.size(), claimed, dropped,
                 signature.c_str(), library.c_str());
  }
  return nullptr;
}


// The subgraph ORT hands to Compile() is exactly the block that was claimed, so
// the node producing its single output IS the block's final Reshape — the node
// the ANNC toolchain names the kernel after. (The fused node's own name is
// generated by ORT, "<provider>_sub_graph_<n>", and carries nothing useful.)
std::string FindAnchorNodeName(const OrtGraph* graph) {
  size_t num_outputs = 0;
  if (g_api->Graph_GetNumOutputs(graph, &num_outputs) != nullptr || num_outputs == 0) {
    return {};
  }
  std::vector<const OrtValueInfo*> outputs(num_outputs, nullptr);
  if (g_api->Graph_GetOutputs(graph, outputs.data(), num_outputs) != nullptr) return {};

  size_t num_nodes = 0;
  if (g_api->Graph_GetNumNodes(graph, &num_nodes) != nullptr) return {};
  std::vector<const OrtNode*> nodes(num_nodes, nullptr);
  if (num_nodes != 0 &&
      g_api->Graph_GetNodes(graph, nodes.data(), num_nodes) != nullptr) {
    return {};
  }
  for (const OrtNode* node : nodes) {
    for (const OrtValueInfo* output : NodeOutputs(node)) {
      for (const OrtValueInfo* wanted : outputs) {
        if (output == wanted) return NodeNameOf(node);
      }
    }
  }
  return {};
}

// Ids of the nodes ORT put into this fused subgraph. The runtime-compile path
// claimed its sites by node id, so the subgraph is identified by the set it
// contains.
std::vector<size_t> SubgraphNodeIds(const OrtGraph* graph) {
  std::vector<size_t> ids;
  size_t num_nodes = 0;
  if (g_api->Graph_GetNumNodes(graph, &num_nodes) != nullptr || num_nodes == 0) {
    return ids;
  }
  std::vector<const OrtNode*> nodes(num_nodes, nullptr);
  if (g_api->Graph_GetNodes(graph, nodes.data(), num_nodes) != nullptr) return ids;
  ids.reserve(num_nodes);
  for (const OrtNode* node : nodes) {
    size_t id = 0;
    if (g_api->Node_GetId(node, &id) != nullptr) return {};
    ids.push_back(id);
  }
  return ids;
}

// Copies a validated context-ABI site contract into the compute info. The
// declared output slots must be exactly the fused subgraph's outputs, and
// every argument/output element type must be one the EP can describe, or the
// kernel would misinterpret the buffers it is handed.
//
// Sites the rank/type rule cannot identify (masked_mul: several rank-3 data
// operands, and one output per chain Mul) are pinned here instead, by tensor
// name: the claim translated the report's names back to the graph's own, so
// each declared argument names one of the fused node's inputs and each
// declared output one of its outputs. Anything that does not line up is a
// contract disagreement and refuses the whole session (Compile has no
// per-node refusal) -- correctly, because running the kernel against the
// wrong buffers would be silent corruption.
bool FillContextContract(const annc_ep::Site& site, const OrtGraph* graph,
                         const OrtNode* fused_node,
                         AnncNodeComputeInfo* info) {
  size_t graph_outputs = 0;
  if (g_api->Graph_GetNumOutputs(graph, &graph_outputs) != nullptr) return false;
  if (graph_outputs != site.outputs.size()) return false;
  auto to_element_type = [](const std::string& dtype, AnncElementType* out) {
    if (dtype == "f32") {
      *out = ANNC_ELEMENT_TYPE_F32;
      return true;
    }
    if (dtype == "i32") {
      *out = ANNC_ELEMENT_TYPE_I32;
      return true;
    }
    if (dtype == "i64") {
      *out = ANNC_ELEMENT_TYPE_I64;
      return true;
    }
    return false;
  };
  for (const annc_ep::Site::Arg& arg : site.args) {
    AnncElementType type = 0;
    if (!to_element_type(arg.dtype, &type)) return false;
    info->arg_ranks.push_back(arg.rank);
    info->arg_types.push_back(type);
  }
  for (const annc_ep::Site::Arg& output : site.outputs) {
    AnncElementType type = 0;
    if (!to_element_type(output.dtype, &type)) return false;
    info->outputs.push_back(annc_ep::OutputContract{output.rank, type});
  }

  const bool by_name =
      annc_ep::SiteNeedsResolvedNames(site) && site.namesResolved;
  if (by_name) {
    const std::vector<std::string> input_names = NodeInputNames(fused_node);
    const std::vector<std::string> output_names = NodeOutputNames(fused_node);
    if (output_names.size() != site.outputs.size()) return false;
    info->arg_indices.assign(site.args.size(), -1);
    for (size_t arg = 0; arg < site.args.size(); ++arg) {
      for (size_t input = 0; input < input_names.size(); ++input) {
        if (input_names[input] != site.args[arg].name) continue;
        // Two arguments claiming the same input means the contract does not
        // describe this fused node; refuse instead of reusing a buffer.
        if (std::find(info->arg_indices.begin(), info->arg_indices.end(),
                      static_cast<int64_t>(input)) != info->arg_indices.end()) {
          return false;
        }
        info->arg_indices[arg] = static_cast<int64_t>(input);
        break;
      }
      if (info->arg_indices[arg] < 0) return false;
    }
    info->output_slots.assign(site.outputs.size(), 0);
    for (size_t slot = 0; slot < site.outputs.size(); ++slot) {
      auto it = std::find(output_names.begin(), output_names.end(),
                          site.outputs[slot].name);
      if (it == output_names.end()) return false;
      info->output_slots[slot] =
          static_cast<uint32_t>(it - output_names.begin());
    }
  }
  return true;
}

OrtStatus* ORT_API_CALL CompileImpl(OrtEp* this_ptr, const OrtGraph** graphs,
                                    const OrtNode** fused_nodes, size_t count,
                                    OrtNodeComputeInfo** node_compute_infos,
                                    OrtNode** ep_context_nodes) noexcept {
  auto* ep = static_cast<AnncEp*>(this_ptr);
  std::lock_guard<std::mutex> ep_state_lock(ep->state_mutex);
  (void)ep_context_nodes;
  size_t unresolved = 0;
  for (size_t i = 0; i < count; ++i) {
    const char* name = nullptr;
    if (fused_nodes[i] != nullptr) (void)g_api->Node_GetName(fused_nodes[i], &name);
    auto* info = new AnncNodeComputeInfo(name ? name : "");
    if (graphs[i] != nullptr) {
      info->anchor_name = FindAnchorNodeName(graphs[i]);
      const std::vector<size_t> node_ids = SubgraphNodeIds(graphs[i]);
      const annc_ep::Site* site = ep->FindSiteByNodeIds(node_ids);
      if (site != nullptr) {
        info->kernel_name = site->kernelName;
        info->library = ep->kernel_library;
        info->abi = site->abi;
        if (site->abi == "annc_execution_v2" &&
            !FillContextContract(*site, graphs[i], fused_nodes[i], info)) {
          // A contract that disagrees with the fused subgraph would run the
          // kernel against the wrong buffers; refuse the node loudly instead.
          ++unresolved;
          std::fprintf(stderr,
                       "[annc-ep] context site '%s' does not match the fused "
                       "subgraph (outputs/ranks/types/names)\n",
                       site->kernelName.c_str());
        }
      } else {
        // Unreachable while GetCapability claims only reported sites and every
        // claimed kernel resolved there; loud because it means the claim and
        // the compile view disagree.
        ++unresolved;
        std::fprintf(stderr,
                     "[annc-ep] no claimed site matches fused node '%s' (%zu nodes)\n",
                     name ? name : "", node_ids.size());
      }
    } else {
      ++unresolved;
    }
    if (g_debug) {
      std::fprintf(stderr,
                   "[annc-ep] Compile[%zu] fused='%s' anchor='%s' kernel='%s' lib='%s'\n",
                   i, name ? name : "", info->anchor_name.c_str(),
                   info->kernel_name.c_str(), info->library.c_str());
    }
    node_compute_infos[i] = info;
    node_compute_infos[i]->ort_version_supported = ORT_API_VERSION;
    node_compute_infos[i]->CreateState = CreateStateImpl;
    node_compute_infos[i]->Compute = ComputeImpl;
    node_compute_infos[i]->ReleaseState = ReleaseStateImpl;
  }
  if (unresolved != 0) {
    return g_api->CreateStatus(ORT_FAIL, "ANNC EP: unresolved kernels (see stderr)");
  }
  return nullptr;
}

void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                              OrtNodeComputeInfo** node_compute_infos,
                                              size_t num_node_compute_infos) noexcept {
  (void)this_ptr;
  for (size_t i = 0; i < num_node_compute_infos; ++i) {
    delete static_cast<AnncNodeComputeInfo*>(node_compute_infos[i]);
  }
}

void ORT_API_CALL FactoryReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  delete static_cast<AnncEp*>(ep);
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
struct AnncEpFactory : OrtEpFactory {
  AnncEpFactory() {
    ort_version_supported = ORT_API_VERSION;
    GetName = FactoryGetName;
    GetVendor = FactoryGetVendor;
    GetSupportedDevices = FactoryGetSupportedDevices;
    CreateEp = FactoryCreateEp;
    ReleaseEp = FactoryReleaseEpImpl;
    // The header calls these "optional", but ORT dereferences them without a
    // null check when ort_version_supported says the library knows about them:
    // GetVendorId in the provider-bridge constructor
    // (plugin_ep/ep_factory_provider_bridge.h:23), GetVersion inside
    // CreateEpDevice (plugin_ep/ep_api.cc:65), and the allocator / data-transfer
    // / stream set at ep_factory_provider_bridge.h:44-65. Leaving any of them
    // null segfaults at registration.
    GetVendorId = FactoryGetVendorId;
    GetVersion = FactoryGetVersion;
    CreateAllocator = FactoryCreateAllocator;
    ReleaseAllocator = FactoryReleaseAllocator;
    CreateDataTransfer = FactoryCreateDataTransfer;
    IsStreamAware = FactoryIsStreamAware;
    CreateSyncStreamForDevice = FactoryCreateSyncStreamForDevice;
    GetNumCustomOpDomains = FactoryGetNumCustomOpDomains;
    GetCustomOpDomains = FactoryGetCustomOpDomains;
    // Genuinely guarded by a null check in the bridge (lines 73/82/95/107/116).
    ValidateCompiledModelCompatibilityInfo = nullptr;
    GetHardwareDeviceIncompatibilityDetails = nullptr;
    CreateExternalResourceImporterForDevice = nullptr;
    InitGraphicsInterop = nullptr;
    DeinitGraphicsInterop = nullptr;
  }

  static const char* ORT_API_CALL FactoryGetName(const OrtEpFactory*) noexcept {
    return "AnncExecutionProvider";
  }
  static const char* ORT_API_CALL FactoryGetVendor(const OrtEpFactory*) noexcept {
    return "ANNC";
  }

  static uint32_t ORT_API_CALL FactoryGetVendorId(const OrtEpFactory*) noexcept {
    return 0;  // no registered vendor id assigned yet
  }

  static const char* ORT_API_CALL FactoryGetVersion(const OrtEpFactory*) noexcept {
    return "0.1.0";
  }

  // ANNC runs on the host CPU with host memory: no device allocator, no
  // transfers, no streams. Returning nullptr for the allocator is explicitly
  // allowed (the parameter is _Outptr_result_maybenull_); ORT then uses its
  // own default allocator, which is what the ANNC kernels expect.
  static OrtStatus* ORT_API_CALL FactoryCreateAllocator(
      OrtEpFactory*, const OrtMemoryInfo*, const OrtKeyValuePairs*,
      OrtAllocator** allocator) noexcept {
    *allocator = nullptr;
    return nullptr;
  }

  static void ORT_API_CALL FactoryReleaseAllocator(OrtEpFactory*, OrtAllocator*) noexcept {}

  static OrtStatus* ORT_API_CALL FactoryCreateDataTransfer(
      OrtEpFactory*, OrtDataTransferImpl** data_transfer) noexcept {
    *data_transfer = nullptr;
    return nullptr;
  }

  static bool ORT_API_CALL FactoryIsStreamAware(const OrtEpFactory*) noexcept {
    return false;
  }

  static OrtStatus* ORT_API_CALL FactoryCreateSyncStreamForDevice(
      OrtEpFactory*, const OrtMemoryDevice*, const OrtKeyValuePairs*,
      OrtSyncStreamImpl** /*stream*/) noexcept {
    // Unreachable while IsStreamAware() is false.
    return g_api->CreateStatus(ORT_NOT_IMPLEMENTED, "ANNC EP has no streams");
  }

  static OrtStatus* ORT_API_CALL FactoryGetNumCustomOpDomains(
      OrtEpFactory*, size_t* num_domains) noexcept {
    *num_domains = 0;
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL FactoryGetCustomOpDomains(
      OrtEpFactory*, OrtCustomOpDomain**, size_t) noexcept {
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL FactoryGetSupportedDevices(
      OrtEpFactory* this_ptr, const OrtHardwareDevice* const* devices,
      size_t num_devices, OrtEpDevice** ep_devices, size_t max_ep_devices,
      size_t* num_ep_devices) noexcept {
    size_t count = 0;
    for (size_t i = 0; i < num_devices && count < max_ep_devices; ++i) {
      // HardwareDevice_Type is an ORT_API_T accessor: it returns the value.
      if (g_api->HardwareDevice_Type(devices[i]) != OrtHardwareDeviceType_CPU) {
        continue;  // ANNC kernels target the host CPU
      }
      OrtEpDevice* ep_device = nullptr;
      if (OrtStatus* status =
              g_ep_api->CreateEpDevice(this_ptr, devices[i], nullptr, nullptr, &ep_device)) {
        return status;
      }
      ep_devices[count++] = ep_device;
    }
    *num_ep_devices = count;
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL FactoryCreateEp(
      OrtEpFactory*, const OrtHardwareDevice* const* /*devices*/,
      const OrtKeyValuePairs* const* /*ep_metadata_pairs*/, size_t /*num_devices*/,
      const OrtSessionOptions* /*session_options*/, const OrtLogger* /*logger*/,
      OrtEp** ep) noexcept {
    *ep = new AnncEp("AnncExecutionProvider");
    (*ep)->ort_version_supported = ORT_API_VERSION;
    (*ep)->GetName = EpGetName;
    (*ep)->GetCapability = GetCapabilityRuntimeCompileImpl;
    (*ep)->Compile = CompileImpl;
    (*ep)->ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
    (*ep)->GetPreferredDataLayout = nullptr;
    (*ep)->ShouldConvertDataLayoutForOp = nullptr;
    (*ep)->SetDynamicOptions = nullptr;
    (*ep)->OnRunStart = nullptr;
    (*ep)->OnRunEnd = nullptr;
    (*ep)->CreateAllocator = nullptr;
    (*ep)->CreateSyncStreamForDevice = nullptr;
    (*ep)->GetCompiledModelCompatibilityInfo = nullptr;
    (*ep)->GetKernelRegistry = nullptr;
    (*ep)->IsConcurrentRunSupported = nullptr;
    (*ep)->Sync = nullptr;
    (*ep)->CreateProfiler = nullptr;
    (*ep)->IsGraphCaptureEnabled = nullptr;
    (*ep)->IsGraphCaptured = nullptr;
    (*ep)->ReplayGraph = nullptr;
    (*ep)->GetGraphCaptureNodeAssignmentPolicy = nullptr;
    (*ep)->GetAvailableResource = nullptr;
    return nullptr;
  }
};

}  // namespace

extern "C" {

OrtStatus* ORT_API_CALL CreateEpFactories(const char* /*registration_name*/,
                                          const OrtApiBase* ort_api_base,
                                          const OrtLogger* /*default_logger*/,
                                          OrtEpFactory** factories, size_t max_factories,
                                          size_t* num_factories) {
  g_api = ort_api_base->GetApi(ORT_API_VERSION);
  g_ep_api = g_api->GetEpApi();
  g_debug = std::getenv("ANNC_EP_DEBUG") != nullptr;
  if (max_factories < 1) {
    return g_api->CreateStatus(ORT_INVALID_ARGUMENT, "need room for at least one factory");
  }
  factories[0] = new AnncEpFactory();
  *num_factories = 1;
  return nullptr;
}

OrtStatus* ORT_API_CALL ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<AnncEpFactory*>(factory);
  return nullptr;
}

}  // extern "C"
