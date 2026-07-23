#ifndef ANNC_TOOLS_ANNC_CONVERTER_ANNC_FUSED_NODE_BUILDER_H
#define ANNC_TOOLS_ANNC_CONVERTER_ANNC_FUSED_NODE_BUILDER_H

#include <string>
#include <unordered_map>

#include "FusionMetadata/FusionMetadata.h"
#include "tensorflow/core/framework/graph.pb.h"

namespace annc::fusion {

class ANNCFusedNodeBuilder {
 public:
  using FusionOutputMap = std::unordered_map<std::string, const FusionInfo *>;

  ANNCFusedNodeBuilder(const tensorflow::GraphDef &original,
                       std::string sharedLibPath,
                       const FusionOutputMap &fusionByOutput);

  std::string rewriteDataInput(const std::string &input) const;
  tensorflow::NodeDef *appendNode(tensorflow::GraphDef &graph,
                                  const FusionInfo &fusion) const;

 private:
  const tensorflow::GraphDef &original;
  std::string sharedLibPath;
  const FusionOutputMap &fusionByOutput;
};

}  // namespace annc::fusion

#endif  // ANNC_TOOLS_ANNC_CONVERTER_ANNC_FUSED_NODE_BUILDER_H
