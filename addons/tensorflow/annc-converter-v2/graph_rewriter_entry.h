#ifndef ANNC_TOOLS_ANNC_CONVERTER_V2_GRAPH_REWRITER_ENTRY_H
#define ANNC_TOOLS_ANNC_CONVERTER_V2_GRAPH_REWRITER_ENTRY_H

// The PRELOAD plugin's C ABI surface — the single source of truth shared by
// the plugin entry (definition) and the Grappler-side caller
// (annc_optimizer.cc, which dlsym's the entry by kAnncRewriteEntryName).
// Keeping the declaration here lets the compiler catch signature drift
// between the two sides.

namespace tensorflow {
class GraphDef;
}

// dlsym name of annc_rewrite_graphdef_in_place.
inline constexpr char kAnncRewriteEntryName[] = "annc_rewrite_graphdef_in_place";

extern "C" {

// dlsym anchor for deployment checks.
int annc_graph_rewriter_abi();

// Rewrites *graph in place from a fusion metadata JSON file. The path
// arguments may be null; *error_out (when non-null) receives a pointer to a
// buffer that stays valid until the next call from the same thread.
bool annc_rewrite_graphdef_in_place(tensorflow::GraphDef *graph,
                                    const char *metadata_json_path,
                                    const char *shared_lib_path,
                                    const char *atir_module_path,
                                    const char *kernel_name_override,
                                    int verbose, const char **error_out);
}

#endif  // ANNC_TOOLS_ANNC_CONVERTER_V2_GRAPH_REWRITER_ENTRY_H
