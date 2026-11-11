#include "common/pb2_common.h"
#include "graph/pb2_graph_pass.h"

int GraphPassManager::run(const tensorflow::GraphDef& graph,
                          BuildCtx& ctx) const {
  for (const auto& pass : passes_) {
    int rc = pass->runOnGraph(graph, ctx);
    if (rc != 0) {
      std::cerr << "[graph-pass] " << pass->getName().str()
                << " failed, rc=" << rc << "\n";
      return rc;
    }
  }
  return 0;
}
