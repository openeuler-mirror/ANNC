// PRELOAD plugin entry. The exported surface is the in-process rewrite core
// (rewriteGraphDefInPlace, default C++ visibility); undefined TF/protobuf
// symbols are resolved by the host predictor process at load time — the
// tensorflow_addons ANNC_TENSORFLOW_PRELOAD mechanism.
//
// The C ABI wrapper below is the dlopen surface for the Grappler-side caller
// (annc_optimizer): it keeps the optimizer free of a build-time link to this
// object, so the build order of the two plugins never matters.

#include "graph_rewriter_entry.h"

#include "graph_rewriter_core.h"

#include "fusion_meta.h"

#include <string>

extern "C" int annc_graph_rewriter_abi() { return 1; }

// Rewrites *graph in place; the declaration lives in graph_rewriter_entry.h.
extern "C" bool annc_rewrite_graphdef_in_place(
    tensorflow::GraphDef *graph, const char *metadata_json_path,
    const char *shared_lib_path, const char *atir_module_path,
    const char *kernel_name_override, int verbose, const char **error_out) {
  using namespace annc::converter_v2;
  static thread_local std::string last_error;
  last_error.clear();
  auto metadata = readFusionMetadataJson(metadata_json_path);
  if (!metadata.ok()) {
    last_error = metadata.error;
    if (error_out) *error_out = last_error.c_str();
    return false;
  }
  RewritePaths paths;
  if (shared_lib_path) paths.sharedLibPath = shared_lib_path;
  if (atir_module_path) paths.atirModulePath = atir_module_path;
  if (kernel_name_override) paths.kernelNameOverride = kernel_name_override;
  paths.verbose = verbose != 0;
  bool ok = rewriteGraphDefInPlace(graph, std::move(metadata.value), paths,
                                   &last_error);
  if (error_out) *error_out = last_error.c_str();
  return ok;
}
