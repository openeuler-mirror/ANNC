#include "GraphDefRewriter.h"

#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>

#include "ANNCFusedNodeBuilder.h"
#include "FusionMetadata/TensorEndpoint.h"
#include "google/protobuf/text_format.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/core/framework/graph.pb.h"

namespace annc::fusion {
namespace {

bool writeBinaryGraphDef(const tensorflow::GraphDef &graph,
                         const std::string &path) {
  std::string serialized;
  if (!graph.SerializeToString(&serialized)) return false;
  std::ofstream output(path, std::ios::binary);
  if (!output.is_open()) return false;
  output.write(serialized.data(),
               static_cast<std::streamsize>(serialized.size()));
  return output.good();
}

bool readBinaryGraphDef(const std::string &path, tensorflow::GraphDef *graph) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) return false;
  std::string data((std::istreambuf_iterator<char>(input)),
                   std::istreambuf_iterator<char>());
  if (input.bad()) return false;
  return graph->ParseFromString(data);
}

bool readTextGraphDef(const std::string &path, tensorflow::GraphDef *graph) {
  std::ifstream input(path);
  if (!input.is_open()) return false;
  std::string data((std::istreambuf_iterator<char>(input)),
                   std::istreambuf_iterator<char>());
  if (input.bad()) return false;
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

  const bool needsAotLibrary = std::any_of(
      fusionInfos.begin(), fusionInfos.end(), [](const FusionInfo &fusion) {
        return fusion.executionMode == "aot";
      });
  const bool needsJitTemplate = std::any_of(
      fusionInfos.begin(), fusionInfos.end(), [](const FusionInfo &fusion) {
        return fusion.executionMode == "jit";
      });
  if (needsAotLibrary && options.sharedLibPath.empty()) {
    llvm::errs() << "[annc-converter] Error: AOT fusion metadata requires "
                    "--shared_lib_path\n";
    return false;
  }
  if (needsJitTemplate && options.atirModulePath.empty()) {
    llvm::errs() << "[annc-converter] Error: JIT fusion metadata requires "
                    "--atir_module_path\n";
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

  ANNCFusedNodeBuilder::FusionOutputMap fusionByOutput;
  for (const auto &fusion : fusionInfos) {
    if (options.verbose) {
      llvm::outs() << "[annc-converter] Rewriting fusion node from ATIR: "
                   << fusion.name << " pattern=" << fusion.pattern << "\n";
    }
    for (size_t slot = 0; slot < fusion.outputs.size(); ++slot) {
      auto endpoint = parseTensorEndpoint(fusion.outputs[slot].tfName);
      if (!endpoint || endpoint->control) {
        llvm::errs()
            << "[annc-converter] Error: invalid fusion output endpoint: "
            << fusion.outputs[slot].tfName << "\n";
        return false;
      }
      auto [it, inserted] =
          fusionByOutput.emplace(endpoint->canonicalDataName(),
                                 ANNCFusedNodeBuilder::FusionOutputTarget{
                                     &fusion, static_cast<int64_t>(slot)});
      if (!inserted) {
        llvm::errs()
            << "[annc-converter] Error: duplicate fusion output endpoint: "
            << endpoint->canonicalDataName() << "\n";
        return false;
      }
    }
  }

  ANNCFusedNodeBuilder fusedNodeBuilder(graph, options.sharedLibPath,
                                        options.atirModulePath, fusionByOutput);
  // Preserve graph-level metadata, including versions, expected by Grappler.
  tensorflow::GraphDef rewritten = graph;
  rewritten.clear_node();
  for (const auto &node : graph.node()) {
    if (fusedNodeBuilder.replacesOutputNode(node.name())) continue;
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
  std::string aliasError;
  if (!fusedNodeBuilder.appendOutputAliases(rewritten, &aliasError)) {
    llvm::errs() << "[annc-converter] Error: " << aliasError << "\n";
    return false;
  }

  return writeBinaryGraphDef(rewritten, options.outputGraphPath);
}

}  // namespace annc::fusion
