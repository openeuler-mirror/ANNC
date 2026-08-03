#include "GraphDefRewriter.h"

#include <string>
#include <unordered_map>

#include "ANNCFusedNodeBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "google/protobuf/text_format.h"

namespace annc::fusion {
namespace {

bool writeBinaryGraphDef(const tensorflow::GraphDef &graph,
                         const std::string &path) {
  std::string out;
  if (!graph.SerializeToString(&out)) return false;
  return tensorflow::WriteStringToFile(tensorflow::Env::Default(), path, out)
      .ok();
}

bool readBinaryGraphDef(const std::string &path, tensorflow::GraphDef *graph) {
  std::string data;
  if (!tensorflow::ReadFileToString(tensorflow::Env::Default(), path, &data)
           .ok()) {
    return false;
  }
  return graph->ParseFromString(data);
}

bool readTextGraphDef(const std::string &path, tensorflow::GraphDef *graph) {
  std::string data;
  if (!tensorflow::ReadFileToString(tensorflow::Env::Default(), path, &data)
           .ok()) {
    return false;
  }
  return google::protobuf::TextFormat::ParseFromString(data, graph);
}

}  // namespace

bool rewriteGraphDefWithANNCFused(std::vector<FusionInfo> fusionInfos,
                                  const GraphDefRewriteOptions &options) {
  tensorflow::GraphDef graph;
  if (!readBinaryGraphDef(options.inputGraphPath, &graph) &&
      !readTextGraphDef(options.inputGraphPath, &graph)) {
    llvm::errs() << "[annc-converter] Error: failed to read GraphDef: "
                 << options.inputGraphPath << "\n";
    return false;
  }

  if (fusionInfos.empty()) {
    llvm::errs() << "[annc-converter] Error: no ANNCFused metadata entries\n";
    return false;
  }

  if (!options.kernelNameOverride.empty()) {
    if (fusionInfos.size() != 1) {
      llvm::errs() << "[annc-converter] Error: --kernel_name override is only "
                      "supported for single-fusion metadata\n";
      return false;
    }
    fusionInfos.front().kernelName = options.kernelNameOverride;
  }

  std::unordered_map<std::string, const FusionInfo *> fusionByOutput;
  for (const auto &fusion : fusionInfos) {
    if (options.verbose) {
      llvm::outs() << "[annc-converter] Rewriting fusion node from ATIR: "
                   << fusion.name << " pattern=" << fusion.pattern << "\n";
    }
    fusionByOutput[fusion.outputs.front().tfName] = &fusion;
  }

  ANNCFusedNodeBuilder fusedNodeBuilder(graph, options.sharedLibPath,
                                        fusionByOutput);
  tensorflow::GraphDef rewritten;
  for (const auto &node : graph.node()) {
    tensorflow::NodeDef *out = rewritten.add_node();
    *out = node;

    out->clear_input();
    for (const auto &input : node.input()) {
      out->add_input(fusedNodeBuilder.rewriteDataInput(input));
    }
  }

  for (const auto &fusion : fusionInfos) {
    fusedNodeBuilder.appendNode(rewritten, fusion);
  }

  if (options.textFormat) {
    return tensorflow::WriteStringToFile(tensorflow::Env::Default(),
                                         options.outputGraphPath,
                                         rewritten.DebugString())
        .ok();
  }
  return writeBinaryGraphDef(rewritten, options.outputGraphPath);
}

}  // namespace annc::fusion
