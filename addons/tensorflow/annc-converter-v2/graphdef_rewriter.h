#ifndef ANNC_TOOLS_ANNC_CONVERTER_V2_GRAPH_DEF_REWRITER_H
#define ANNC_TOOLS_ANNC_CONVERTER_V2_GRAPH_DEF_REWRITER_H

// ANNCFused node assembly for the standalone GraphDef rewriter (std-only,
// TensorFlow dependency like tensorflow_addons). The emitted node attributes
// are the runtime contract of tensorflow_addons/annc_fused_op*.cc; keep in
// sync with the v1 tool. The in-process rewrite entry lives in
// graph_rewriter_core.h.

#include <string>
#include <unordered_map>
#include <vector>

#include "fusion_meta.h"
#include "tensorflow/core/framework/graph.pb.h"

namespace annc::converter_v2 {

class ANNCFusedNodeBuilder {
 public:
  struct FusionOutputTarget {
    const FusionInfo *fusion;
    int64_t slot;
  };
  using FusionOutputMap = std::unordered_map<std::string, FusionOutputTarget>;

  ANNCFusedNodeBuilder(const tensorflow::GraphDef &original,
                       std::string sharedLibPath, std::string atirModulePath,
                       const FusionOutputMap &fusionByOutput);

  std::string rewriteDataInput(const std::string &input) const;
  bool replacesOutputNode(const std::string &nodeName) const;
  bool appendOutputAliases(tensorflow::GraphDef &graph,
                           std::string *error) const;
  tensorflow::NodeDef *appendNode(tensorflow::GraphDef &graph,
                                  const FusionInfo &fusion,
                                  std::string *error = nullptr) const;

 private:
  const tensorflow::GraphDef &original;
  std::string sharedLibPath;
  std::string atirModulePath;
  const FusionOutputMap &fusionByOutput;
};

}  // namespace annc::converter_v2

#endif  // ANNC_TOOLS_ANNC_CONVERTER_V2_GRAPH_DEF_REWRITER_H
