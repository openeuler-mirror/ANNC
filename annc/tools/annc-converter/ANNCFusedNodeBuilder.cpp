#include "ANNCFusedNodeBuilder.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "FusionMetadata/TensorEndpoint.h"
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

std::string formatRuntimeOutputShape(llvm::ArrayRef<int64_t> outputShape) {
  std::string shape;
  for (size_t i = 0; i < outputShape.size(); ++i) {
    if (i > 0) shape += ",";
    if (outputShape[i] < 0) {
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

tensorflow::DataType fusionDType(llvm::StringRef dtype) {
  if (dtype == "f32" || dtype == "float32") return tensorflow::DT_FLOAT;
  if (dtype == "f64" || dtype == "float64") return tensorflow::DT_DOUBLE;
  if (dtype == "f16" || dtype == "float16") return tensorflow::DT_HALF;
  if (dtype == "bf16" || dtype == "bfloat16") return tensorflow::DT_BFLOAT16;
  if (dtype == "i8" || dtype == "int8") return tensorflow::DT_INT8;
  if (dtype == "i16" || dtype == "int16") return tensorflow::DT_INT16;
  if (dtype == "i32" || dtype == "int32") return tensorflow::DT_INT32;
  if (dtype == "i64" || dtype == "si64" || dtype == "int64")
    return tensorflow::DT_INT64;
  if (dtype == "ui8" || dtype == "uint8") return tensorflow::DT_UINT8;
  if (dtype == "ui16" || dtype == "uint16") return tensorflow::DT_UINT16;
  if (dtype == "ui32" || dtype == "uint32") return tensorflow::DT_UINT32;
  if (dtype == "ui64" || dtype == "uint64") return tensorflow::DT_UINT64;
  if (dtype == "bool") return tensorflow::DT_BOOL;
  if (dtype == "string") return tensorflow::DT_STRING;
  return tensorflow::DT_INVALID;
}

void setTypeList(tensorflow::AttrValue *attr,
                 llvm::ArrayRef<tensorflow::DataType> types) {
  auto *list = attr->mutable_list();
  for (tensorflow::DataType type : types) list->add_type(type);
}

int declaredOutputCount(const tensorflow::NodeDef &node) {
  auto shapes = node.attr().find("_output_shapes");
  if (shapes == node.attr().end()) return -1;
  return shapes->second.list().shape_size();
}

bool hasConsumers(const tensorflow::GraphDef &graph,
                  const std::string &nodeName) {
  for (const tensorflow::NodeDef &node : graph.node()) {
    for (const std::string &input : node.input()) {
      auto endpoint = parseTensorEndpoint(input);
      if (!endpoint) {
        llvm::consumeError(endpoint.takeError());
        continue;
      }
      if (!endpoint->control && endpoint->node == nodeName) return true;
    }
  }
  return false;
}

}  // namespace

ANNCFusedNodeBuilder::ANNCFusedNodeBuilder(
    const tensorflow::GraphDef &original, std::string sharedLibPath,
    std::string atirModulePath,
    const FusionOutputMap &fusionByOutput)
    : original(original),
      sharedLibPath(std::move(sharedLibPath)),
      atirModulePath(std::move(atirModulePath)),
      fusionByOutput(fusionByOutput) {}

std::string ANNCFusedNodeBuilder::rewriteDataInput(
    const std::string &input) const {
  auto endpoint = parseTensorEndpoint(input);
  if (!endpoint || endpoint->control) return input;
  auto fusedIt = fusionByOutput.find(endpoint->canonicalDataName());
  if (fusedIt == fusionByOutput.end()) return input;
  return fusedIt->second.fusion->name + ":" +
         std::to_string(fusedIt->second.slot);
}

bool ANNCFusedNodeBuilder::replacesOutputNode(
    const std::string &nodeName) const {
  for (const auto &[name, target] : fusionByOutput) {
    (void)target;
    auto endpoint = parseTensorEndpoint(name);
    if (!endpoint) {
      llvm::consumeError(endpoint.takeError());
      continue;
    }
    if (!endpoint->control && endpoint->node == nodeName) return true;
  }
  return false;
}

bool ANNCFusedNodeBuilder::appendOutputAliases(tensorflow::GraphDef &graph,
                                               std::string *error) const {
  using AliasTarget = std::pair<int64_t, const FusionOutputTarget *>;
  for (const tensorflow::NodeDef &originalNode : original.node()) {
    std::vector<AliasTarget> targets;
    for (const auto &[name, target] : fusionByOutput) {
      auto endpoint = parseTensorEndpoint(name);
      if (!endpoint) {
        llvm::consumeError(endpoint.takeError());
        continue;
      }
      if (!endpoint->control && endpoint->node == originalNode.name()) {
        targets.emplace_back(endpoint->port, &target);
      }
    }
    if (targets.empty()) continue;
    if (hasConsumers(original, originalNode.name())) continue;
    std::sort(targets.begin(), targets.end(),
              [](const AliasTarget &lhs, const AliasTarget &rhs) {
                return lhs.first < rhs.first;
              });

    const int outputCount = declaredOutputCount(originalNode);
    const int requiredCount =
        outputCount >= 0 ? outputCount : static_cast<int>(targets.size());
    if (requiredCount != static_cast<int>(targets.size())) {
      if (error) {
        *error = "output node '" + originalNode.name() + "' must replace all " +
                 std::to_string(requiredCount) + " output slots";
      }
      return false;
    }
    for (int port = 0; port < requiredCount; ++port) {
      if (targets[port].first != port) {
        if (error) {
          *error = "output node '" + originalNode.name() +
                   "' must replace contiguous output slots starting at 0";
        }
        return false;
      }
    }

    tensorflow::NodeDef *alias = graph.add_node();
    alias->set_name(originalNode.name());
    alias->set_op(requiredCount == 1 ? "Identity" : "IdentityN");
    alias->set_device(originalNode.device());
    llvm::SmallVector<tensorflow::DataType> types;
    for (const auto &[port, target] : targets) {
      (void)port;
      if (!target->fusion || target->slot < 0 ||
          target->slot >=
              static_cast<int64_t>(target->fusion->outputs.size())) {
        if (error) {
          *error = "output node '" + originalNode.name() +
                   "' has an invalid fusion output slot";
        }
        return false;
      }
      if (target->fusion->name == originalNode.name()) {
        if (error) {
          *error = "fusion node name conflicts with output alias '" +
                   originalNode.name() + "'";
        }
        return false;
      }
      alias->add_input(target->fusion->name + ":" +
                       std::to_string(target->slot));
      tensorflow::DataType type =
          fusionDType(target->fusion->outputs[target->slot].dtype);
      if (type == tensorflow::DT_INVALID) {
        if (error) {
          *error = "output node '" + originalNode.name() +
                   "' has an unsupported dtype";
        }
        return false;
      }
      types.push_back(type);
    }
    if (requiredCount == 1) {
      (*alias->mutable_attr())["T"].set_type(types.front());
    } else {
      setTypeList(&(*alias->mutable_attr())["T"], types);
    }
    auto shapes = originalNode.attr().find("_output_shapes");
    if (shapes != originalNode.attr().end()) {
      (*alias->mutable_attr())["_output_shapes"] = shapes->second;
    }
  }
  return true;
}

tensorflow::NodeDef *ANNCFusedNodeBuilder::appendNode(
    tensorflow::GraphDef &graph, const FusionInfo &fusion) const {
  llvm::ArrayRef<FusionArg> args = fusion.args;
  llvm::ArrayRef<FusionArg> outputs = fusion.outputs;
  const std::string &outputName = outputs.front().tfName;

  const tensorflow::NodeDef *reluNode =
      findNode(original, cleanTensorName(outputName));
  int rank = reluNode ? outputRank(*reluNode) : 2;
  constexpr tensorflow::DataType legacyType = tensorflow::DT_FLOAT;

  tensorflow::NodeDef *fused = graph.add_node();
  fused->set_name(fusion.name);
  fused->set_op("ANNCFused");
  fused->set_device(chooseFusionDevice(original, args, outputs));
  for (const FusionArg &arg : args) {
    fused->add_input(rewriteDataInput(arg.tfName));
  }

  auto *attrs = fused->mutable_attr();
  (*attrs)["kernel_name"].set_s(fusion.kernelName);
  (*attrs)["template_fingerprint"].set_s(fusion.templateFingerprint);
  (*attrs)["shared_lib_path"].set_s(sharedLibPath);
  (*attrs)["atir_module_path"].set_s(atirModulePath);
  (*attrs)["abi"].set_s(fusion.abi);
  int64_t numOutputs = static_cast<int64_t>(outputs.size());
  (*attrs)["num_outputs"].set_i(numOutputs);
  (*attrs)["T"].set_type(legacyType);

  llvm::SmallVector<tensorflow::DataType> inputTypes;
  inputTypes.reserve(args.size());
  for (const FusionArg &arg : args) {
    inputTypes.push_back(fusionInputDType(original, arg.tfName));
  }

  llvm::SmallVector<tensorflow::DataType> constantTypes;
  llvm::SmallVector<tensorflow::DataType> fixedTypes;
  llvm::SmallVector<tensorflow::DataType> dynamicTypes;
  for (size_t i = 0; i < args.size() && i < inputTypes.size(); ++i) {
    tensorflow::DataType type = fusionDType(args[i].dtype);
    if (type == tensorflow::DT_INVALID) type = inputTypes[i];
    if (args[i].role == "constant") {
      constantTypes.push_back(type);
    } else if (args[i].role == "fixed") {
      fixedTypes.push_back(type);
    } else if (args[i].role == "dynamic") {
      dynamicTypes.push_back(type);
    }
  }

  llvm::SmallVector<tensorflow::DataType> outputTypes;
  outputTypes.reserve(outputs.size());
  for (const FusionArg &output : outputs) {
    tensorflow::DataType outputType = fusionDType(output.dtype);
    if (outputType == tensorflow::DT_INVALID) outputType = legacyType;
    outputTypes.push_back(outputType);
  }
  setTypeList(&(*attrs)["Tconstants"], constantTypes);
  setTypeList(&(*attrs)["Tfixed"], fixedTypes);
  setTypeList(&(*attrs)["Tdynamic"], dynamicTypes);
  setTypeList(&(*attrs)["Toutputs"], outputTypes);

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
    outputShapes->add_s(formatRuntimeOutputShape(output.shape));
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
