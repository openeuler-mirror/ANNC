#ifndef ANNC_TOOLS_ANNC_CONVERTER_GRAPH_DEF_REWRITER_H
#define ANNC_TOOLS_ANNC_CONVERTER_GRAPH_DEF_REWRITER_H

#include <string>
#include <vector>

#include "FusionMetadata/FusionMetadata.h"

namespace annc::fusion {

struct GraphDefRewriteOptions {
  std::string inputGraphPath;
  std::string outputGraphPath;
  std::string kernelNameOverride;
  std::string sharedLibPath;
  bool textFormat = false;
  bool verbose = false;
};

bool rewriteGraphDefWithANNCFused(std::vector<FusionInfo> fusionInfos,
                                  const GraphDefRewriteOptions &options);

}  // namespace annc::fusion

#endif  // ANNC_TOOLS_ANNC_CONVERTER_GRAPH_DEF_REWRITER_H
