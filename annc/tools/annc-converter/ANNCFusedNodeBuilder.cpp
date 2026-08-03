#include "ANNCFusedNodeBuilder.h"

#include <unordered_set>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"

namespace annc::fusion {
namespace {

std::string cleanTensorName(std::string name) {
  if (!name.empty() && name[0] == '^') name = name.substr(1);
  size_t colon = name.find(':');
  if (colon != std::string::npos) name = name.substr(0, colon);
  return name;
}

std::string tensorSuffix(const std::string &name) {
  if (!name.empty() && name[0] == '^') return "";
  size_t colon = name.find(':');
  return colon == std::string::npos ? "" : name.substr(colon);
}

const tensorflow::NodeDef *findNode(const tensorflow::GraphDef &graph,
                                    const std::string &name) {
  for (const auto &node : graph.node()) {
    if (node.name() == name) return &node;
  }
  return nullptr;
}

int outputRank(const tensorflow::NodeDef &node) {
  auto it = node.attr().find("_output_shapes");
  if (it != node.attr().end() && it->second.list().shape_size() > 0) {
    return it->second.list().shape(0).dim_size();
  }
  auto shapeIt = node.attr().find("shape");
  if (shapeIt != node.attr().end()) {
    return shapeIt->second.shape().dim_size();
  }
  return 2;
}

tensorflow::DataType nodeDType(const tensorflow::NodeDef &node) {
  auto t = node.attr().find("T");
  if (t != node.attr().end()) return t->second.type();
  auto dstT = node.attr().find("DstT");
  if (dstT != node.attr().end()) return dstT->second.type();
  auto srcT = node.attr().find("SrcT");
  if (srcT != node.attr().end()) return srcT->second.type();
  auto tparams = node.attr().find("Tparams");
  if (tparams != node.attr().end() && tparams->second.list().type_size() > 0) {
    return tparams->second.list().type(0);
  }
  auto dtype = node.attr().find("dtype");
  if (dtype != node.attr().end()) return dtype->second.type();
  return tensorflow::DT_FLOAT;
}

int64_t rankOf(const FusionArg &arg) {
  return arg.rank >= 0 ? arg.rank : static_cast<int64_t>(arg.shape.size());
}

int64_t rankOf(const FusionArg &arg, int64_t fallback) {
  if (arg.rank >= 0) return arg.rank;
  if (!arg.shape.empty()) return static_cast<int64_t>(arg.shape.size());
  return fallback;
}

int64_t countArgsWithRole(llvm::ArrayRef<FusionArg> args,
                          llvm::StringRef role) {
  int64_t count = 0;
  for (const FusionArg &arg : args) {
    if (arg.role == role) ++count;
  }
  return count;
}

std::string formatRuntimeOutputShape(const FusionInfo &fusion,
                                     llvm::ArrayRef<int64_t> outputShape) {
  std::unordered_set<int64_t> dynamicDims(fusion.dynamicDims.begin(),
                                          fusion.dynamicDims.end());
  std::string shape;
  for (size_t i = 0; i < outputShape.size(); ++i) {
    if (i > 0) shape += ",";
    if (dynamicDims.count(static_cast<int64_t>(i)) > 0) {
      shape += "?";
    } else {
      shape += std::to_string(outputShape[i]);
    }
  }
  return shape;
}

std::string chooseFusionDevice(const tensorflow::GraphDef &graph,
                               llvm::ArrayRef<FusionArg> args,
                               llvm::ArrayRef<FusionArg> outputs) {
  std::string outputName = outputs.empty() ? "" : outputs.front().tfName;
  const tensorflow::NodeDef *output = findNode(graph, outputName);
  if (output && !output->device().empty()) return output->device();

  for (const FusionArg &arg : args) {
    const tensorflow::NodeDef *node =
        findNode(graph, cleanTensorName(arg.tfName));
    if (node && !node->device().empty()) return node->device();
  }

  return "/job:localhost/replica:0/task:0/device:CPU:0";
}

tensorflow::DataType fusionInputDType(const tensorflow::GraphDef &graph,
                                      const std::string &input) {
  const tensorflow::NodeDef *node = findNode(graph, cleanTensorName(input));
  return node ? nodeDType(*node) : tensorflow::DT_FLOAT;
}

tensorflow::DataType firstOrDefault(
    llvm::ArrayRef<tensorflow::DataType> types,
    tensorflow::DataType fallback = tensorflow::DT_FLOAT) {
  return types.empty() ? fallback : types.front();
}

}  // namespace

ANNCFusedNodeBuilder::ANNCFusedNodeBuilder(
    const tensorflow::GraphDef &original, std::string sharedLibPath,
    const FusionOutputMap &fusionByOutput)
    : original(original),
      sharedLibPath(std::move(sharedLibPath)),
      fusionByOutput(fusionByOutput) {}

std::string ANNCFusedNodeBuilder::rewriteDataInput(
    const std::string &input) const {
  if (!input.empty() && input[0] == '^') return input;
  std::string src = cleanTensorName(input);
  auto fusedIt = fusionByOutput.find(src);
  if (fusedIt == fusionByOutput.end()) return input;
  return fusedIt->second->name + tensorSuffix(input);
}

tensorflow::NodeDef *ANNCFusedNodeBuilder::appendNode(
    tensorflow::GraphDef &graph, const FusionInfo &fusion) const {
  llvm::ArrayRef<FusionArg> args = fusion.args;
  llvm::ArrayRef<FusionArg> outputs = fusion.outputs;
  const std::string &outputName = outputs.front().tfName;

  const tensorflow::NodeDef *reluNode = findNode(original, outputName);
  int rank = reluNode ? outputRank(*reluNode) : 2;
  tensorflow::DataType dtype =
      reluNode ? nodeDType(*reluNode) : tensorflow::DT_FLOAT;

  tensorflow::NodeDef *fused = graph.add_node();
  fused->set_name(fusion.name);
  fused->set_op("ANNCFused");
  fused->set_device(chooseFusionDevice(original, args, outputs));
  for (const FusionArg &arg : args) {
    fused->add_input(rewriteDataInput(arg.tfName));
  }

  auto *attrs = fused->mutable_attr();
  (*attrs)["kernel_name"].set_s(fusion.kernelName);
  (*attrs)["shared_lib_path"].set_s(sharedLibPath);
  (*attrs)["abi"].set_s(fusion.abi);
  int64_t numOutputs = static_cast<int64_t>(outputs.size());
  (*attrs)["num_outputs"].set_i(numOutputs);
  (*attrs)["T"].set_type(dtype);

  llvm::SmallVector<tensorflow::DataType> inputTypes;
  inputTypes.reserve(args.size());
  for (const FusionArg &arg : args) {
    inputTypes.push_back(fusionInputDType(original, arg.tfName));
  }

  llvm::SmallVector<tensorflow::DataType> constantTypes;
  llvm::SmallVector<tensorflow::DataType> fixedTypes;
  llvm::SmallVector<tensorflow::DataType> dynamicTypes;
  for (size_t i = 0; i < args.size() && i < inputTypes.size(); ++i) {
    if (args[i].role == "constant") {
      constantTypes.push_back(inputTypes[i]);
    } else if (args[i].role == "fixed") {
      fixedTypes.push_back(inputTypes[i]);
    } else if (args[i].role == "dynamic") {
      dynamicTypes.push_back(inputTypes[i]);
    }
  }

  llvm::SmallVector<tensorflow::DataType> outputTypes;
  for (int64_t i = 0; i < numOutputs; ++i) {
    outputTypes.push_back(dtype);
  }
  (*attrs)["Tconstants"].set_type(firstOrDefault(constantTypes));
  (*attrs)["Tfixed"].set_type(firstOrDefault(fixedTypes));
  (*attrs)["Tdynamic"].set_type(firstOrDefault(dynamicTypes));
  (*attrs)["Toutputs"].set_type(firstOrDefault(outputTypes, dtype));

  auto *rankList = (*attrs)["output_ranks"].mutable_list();
  for (const FusionArg &output : outputs) {
    rankList->add_i(rankOf(output, rank));
  }
  auto *inputRanks = (*attrs)["input_ranks"].mutable_list();
  for (const FusionArg &arg : args) {
    inputRanks->add_i(rankOf(arg));
  }
  auto *outputShapes = (*attrs)["output_shapes"].mutable_list();
  for (const FusionArg &output : outputs) {
    outputShapes->add_s(formatRuntimeOutputShape(fusion, output.shape));
  }
  auto *kernelArgOrder = (*attrs)["kernel_arg_order"].mutable_list();
  for (int64_t value : fusion.kernelArgOrder) kernelArgOrder->add_i(value);
  auto *dynamicDims = (*attrs)["dynamic_dims"].mutable_list();
  for (int64_t value : fusion.dynamicDims) dynamicDims->add_i(value);
  (*attrs)["symbolic_signature"].set_s(fusion.symbolicSignature);
  (*attrs)["Nconstants"].set_i(countArgsWithRole(args, "constant"));
  (*attrs)["Nfixed"].set_i(countArgsWithRole(args, "fixed"));
  (*attrs)["Ndynamic"].set_i(countArgsWithRole(args, "dynamic"));
  (*attrs)["fallback_function"].mutable_func()->set_name(
      fusion.fallbackFunction);
  (*attrs)["fusion_pattern"].set_s(fusion.pattern);

  return fused;
}

}  // namespace annc::fusion
