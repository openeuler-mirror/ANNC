// annc_ep_compile.h — runtime-compile support for the ANNC ONNX Runtime EP
//
// The EP no longer decides which subgraphs ANNC can fuse: it exports the graph
// ORT hands it, runs the ANNC toolchain on that export, and claims exactly the
// sites the toolchain reports. This header covers everything the EP needs for
// that: turning an OrtGraph into a self-contained ONNX model with deterministic
// names, fingerprinting it for the disk cache, invoking the compile driver,
// and reading back its site report.
#ifndef ANNC_EP_COMPILE_H
#define ANNC_EP_COMPILE_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct OrtGraph;

namespace annc_ep {

// ---------------------------------------------------------------------------
// Graph export
// ---------------------------------------------------------------------------

// Deterministic renaming applied to the exported model:
//   node i                          -> "n<i>"
//   tensor produced by node i : p   -> "t<i>_<p>"
//   graph input produced by nobody  -> "in_<k>"
//   initializer not in graph.input  -> "w_<k>"
// The EP's own names are what both sides of the interface speak, so no lookup
// depends on exporter-chosen (possibly empty or duplicate) names.
struct NameMaps {
  // Proto node index -> renamed node name. Index i corresponds to
  // Graph_GetNodes()[i] (OrtGraphToProto walks the graph in that order).
  std::vector<std::string> node_names;
  std::unordered_map<std::string, int> node_index_by_name;
  std::unordered_map<std::string, std::string> tensor_to_renamed;
  std::unordered_map<std::string, std::string> renamed_to_tensor;
};

// Serializes `graph` to `modelPath` as an ONNX model (initializers inline, so
// the file is self-contained) and applies the renaming above. Returns false
// and fills `error` on any failure; the caller then claims nothing.
bool ExportGraphToModel(const OrtGraph* graph, const std::string& modelPath,
                        NameMaps* maps, std::string* error);

// ---------------------------------------------------------------------------
// Signature
// ---------------------------------------------------------------------------

// SHA-256 of a file's contents, lowercase hex. Empty string on failure.
std::string Sha256File(const std::string& path, std::string* error);

// SHA-256 of an in-memory buffer, lowercase hex.
std::string Sha256String(const std::string& data);

// Structural signature of an exported model: everything that decides what the
// ANNC pipeline will emit, with the volatile and weight-bearing parts removed
// (producer/doc strings and graph name). Initializer payloads are retained
// when the graph contains Mul and MatMul: pre-affine folding can embed them
// in a generated kernel. Otherwise runtime initializer inputs stay excluded.
// `toolchainFingerprint` comes from CompileDriver::ToolchainInfo().
std::string ModelSignature(const std::string& modelPath,
                           const std::string& toolchainFingerprint,
                           std::string* error);

// ---------------------------------------------------------------------------
// Compile driver
// ---------------------------------------------------------------------------

struct Site {
  struct Arg {
    std::string name;
    std::string role;
    std::string dtype;
    int64_t rank = -1;
    std::vector<int64_t> shape;
  };

  std::string pattern;      // pattern implemented by the current EP ABI
  std::string kernelName;   // ANNC kernel function name (the .so symbol stem)
  std::string executionMode;
  std::string abi;
  std::string outputTensor;  // fused output tensor, in the EP's naming
  std::vector<std::string> memberNodes;
  std::vector<Arg> args;
  std::vector<Arg> outputs;
  std::vector<int64_t> kernelArgOrder;
  int64_t flattenXIndex = -1;  // index of the dynamic (rank-3) input, if known
  int64_t weightIndex = -1;    // index of the fixed (rank-2) input, if known
  // True when every argument/output name was translated back to the original
  // graph's tensor names at claim time. Only then can Compile() bind the
  // kernel arguments by name; otherwise the EP falls back to binding by
  // rank+element type, which needs distinct argument ranks.
  bool namesResolved = false;
};

struct SiteReport {
  std::vector<Site> sites;
};

// Validates the part of the site ABI consumed by the EP. Reports a reason
// suitable for diagnostics; invalid sites must be ignored before capability
// registration or cache publication.
bool ValidateSite(const Site& site, std::string* error);

// Fusion-pattern switches (ANNC_EP_FUSION_*): "0" disables the pattern,
// unset or any other value keeps it enabled. Unknown patterns (including
// "matmul") have no switch and are always enabled. Read per capability query.
bool FusionPatternEnabled(const std::string& pattern);

// True when the site cannot be executed with the rank-based fallback binding:
// repeated argument ranks (masked_mul's data operands) or more than one output
// whose order the report cannot pin down. Such a site may only be claimed when
// its argument/output names resolved back to the graph's own names.
bool SiteNeedsResolvedNames(const Site& site);

// Invokes `annc-onnx-pipeline` (the tool that runs the ANNC stages and writes
// the site report) and reads its output back.
//
// Environment (read once per process):
//   ANNC_EP_PIPELINE        absolute path to annc-onnx-pipeline; when unset the
//                           tool is looked up next to this library, then PATH
//   ANNC_EP_COMPILE         "0" disables runtime compilation entirely
//   ANNC_EP_CACHE_DIR       disk cache root (default: $XDG_CACHE_HOME/annc-ep)
//   ANNC_EP_COMPILE_TIMEOUT_S  child timeout, default 600
//   ANNC_EP_KEEP_TEMPS      keep the per-compile work directory
class CompileDriver {
 public:
  // True when a driver was located and ANNC_EP_COMPILE is not "0".
  bool Available() const;
  // The reason Available() is false, for one-line diagnostics.
  const std::string& UnavailableReason() const { return unavailable_reason_; }

  // Fingerprint of the toolchain this process would use: the driver's
  // `--toolchain-info` output, hashed. Memoized; empty on failure.
  std::string ToolchainInfo();

  // Returns the .so path + site report for `modelPath`, compiling on a cache
  // miss. On any failure returns false, fills `error`, and leaves the caller
  // to fall back to the CPU EP. Failures are not memoized: a later session on
  // the same model retries the compile (only the driver lookup is memoized).
  bool Compile(const std::string& modelPath, const std::string& signature,
               std::string* kernelLibrary, SiteReport* report,
               std::string* error);

 private:
  bool LocateDriver();
  bool ReadReport(const std::string& path, SiteReport* report,
                  std::string* error) const;

  bool located_ = false;
  bool available_ = false;
  std::string driver_path_;
  std::string unavailable_reason_;
  std::string toolchain_info_;  // memoized, "" until queried
  std::string cache_key_;  // set by Compile() for CacheEntryPath()
};

// The directory holding the compiled artefacts of `signature` (creating the
// cache root as needed). Exposed for diagnostics.
std::string CacheEntryPath(const std::string& signature);

// A per-process scratch directory under the cache root (created on demand),
// where the EP writes the exported model before compiling it.
std::string ProcessTempDir();

}  // namespace annc_ep

#endif  // ANNC_EP_COMPILE_H
