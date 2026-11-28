#ifndef ANNC_TOOLS_ANNC_CONVERTER_V2_GRAPH_REWRITER_CORE_H
#define ANNC_TOOLS_ANNC_CONVERTER_V2_GRAPH_REWRITER_CORE_H

// In-process rewrite core shared by the CLI and the PRELOAD plugin. Operates
// on an in-memory GraphDef with no file IO, so the plugin path (Grappler
// callback inside the predictor process) and the CLI path produce identical
// graphs for identical inputs.

#include <string>
#include <vector>

#include "fusion_meta.h"
#include "tensorflow/core/framework/graph.pb.h"

namespace annc::converter_v2 {

// Rewrite parameters that do not depend on where the graph came from.
struct RewritePaths {
  std::string sharedLibPath;
  std::string atirModulePath;
  std::string kernelNameOverride;
  bool verbose = false;
};

// Rewrites *graph in place. On failure returns false and fills *error (when
// non-null); the graph stays loadable either way — unresolvable sites are
// dropped per the documented semantics, so a success result may still carry
// the original graph when every site was dropped.
bool rewriteGraphDefInPlace(tensorflow::GraphDef *graph,
                            std::vector<FusionInfo> fusionInfos,
                            const RewritePaths &paths, std::string *error);

}  // namespace annc::converter_v2

#endif  // ANNC_TOOLS_ANNC_CONVERTER_V2_GRAPH_REWRITER_CORE_H
