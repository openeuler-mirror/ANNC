#include "graphdef_rewriter.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include <google/protobuf/io/tokenizer.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"

namespace annc::converter_v2 {
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

// Mirrors the v1 converter helper: synthetic helpers spell "anchor_field",
// real-model TF graphs "anchor/field"; slash variant on exact-name miss.
std::string slashVariant(const std::string &name) {
  static const char *kSuffixes[] = {"_begin",   "_end", "_stack", "_indices",
                                    "_pack",    "_const", "_shape",
                                    "_dim"};
  for (const char *suffix : kSuffixes) {
    size_t pos = name.rfind(suffix);
    if (pos != std::string::npos && pos + strlen(suffix) == name.size()) {
      std::string alt = name.substr(0, pos);
      alt += '/';
      alt += suffix + 1;
      return alt;
    }
  }
  return "";
}

// Exact name first; on miss, the slash variant. Returns the name that
// resolves (or the input unchanged if neither exists).
std::string resolveDataName(const tensorflow::GraphDef &graph,
                            const std::string &input) {
  std::string base = cleanTensorName(input);
  if (findNode(graph, base) != nullptr) return input;
  std::string alt = slashVariant(base);
  if (!alt.empty() && findNode(graph, alt) != nullptr) {
    // Keep the original port suffix (":0" etc) if any.
    size_t colon = input.find_last_of(':');
    if (colon != std::string::npos) return alt + input.substr(colon);
    return alt;
  }
  return input;
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

int64_t countArgsWithRole(const std::vector<FusionArg> &args,
                          const std::string &role) {
  int64_t count = 0;
  for (const FusionArg &arg : args) {
    if (arg.role == role) ++count;
  }
  return count;
}

std::string formatRuntimeOutputShape(const std::vector<int64_t> &outputShape) {
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
                               const std::vector<FusionArg> &args,
                               const std::vector<FusionArg> &outputs) {
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

tensorflow::DataType fusionDType(const std::string &dtype) {
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
                 const std::vector<tensorflow::DataType> &types) {
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
      if (!endpoint.ok()) continue;
      if (!endpoint.value.control && endpoint.value.node == nodeName) return true;
    }
  }
  return false;
}
}  // namespace

// Temp file + rename: a failed write never leaves a truncated GraphDef.
static bool writeBinaryGraphDef(const tensorflow::GraphDef &graph,
                                const std::string &path) {
  std::string serialized;
  if (!graph.SerializeToString(&serialized)) return false;
  const std::string tmpPath = path + ".tmp." + std::to_string(::getpid());
  {
    std::ofstream output(tmpPath, std::ios::binary);
    if (!output.is_open()) {
      std::cerr << "[annc-converter-v2] Error: cannot open output file: "
                << path << "\n";
      return false;
    }
    output.write(serialized.data(),
                 static_cast<std::streamsize>(serialized.size()));
    output.close();
    if (!output.good()) {
      std::cerr << "[annc-converter-v2] Error: failed to write output: "
                << path << "\n";
      return false;
    }
  }
  if (std::rename(tmpPath.c_str(), path.c_str()) != 0) {
    std::cerr << "[annc-converter-v2] Error: cannot finalize output: " << path
              << "\n";
    return false;
  }
  return true;
}

static bool readBinaryGraphDef(const std::string &path,
                               tensorflow::GraphDef *graph) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) return false;
  std::string data((std::istreambuf_iterator<char>(input)),
                   std::istreambuf_iterator<char>());
  if (input.bad()) return false;
  return graph->ParseFromString(data);
}

namespace {

// Caps protobuf text-format errors at one line (the default parser would
// spray its own multi-line log on stderr).
struct QuietTextErrorCollector : google::protobuf::io::ErrorCollector {
  std::string firstError;
  void RecordError(int line, int column,
                   absl::string_view message) override {
    if (firstError.empty())
      firstError = "line " + std::to_string(line + 1) + ": " +
                   std::string(message);
  }
  void RecordWarning(int, int, absl::string_view) override {}
};

}  // namespace

static bool readTextGraphDef(const std::string &path,
                             tensorflow::GraphDef *graph) {
  std::ifstream input(path);
  if (!input.is_open()) return false;
  std::string data((std::istreambuf_iterator<char>(input)),
                   std::istreambuf_iterator<char>());
  if (input.bad()) return false;
  QuietTextErrorCollector collector;
  google::protobuf::TextFormat::Parser parser;
  parser.RecordErrorsTo(&collector);
  google::protobuf::io::ArrayInputStream stream(data.data(),
                                                static_cast<int>(data.size()));
  if (!parser.Parse(&stream, graph)) {
    std::cerr << "[annc-converter-v2] Error: invalid text GraphDef in " << path
              << (collector.firstError.empty()
                      ? ""
                      : " (" + collector.firstError + ")")
              << "\n";
    return false;
  }
  return true;
}

ANNCFusedNodeBuilder::ANNCFusedNodeBuilder(
    const tensorflow::GraphDef &original, std::string sharedLibPath,
    std::string atirModulePath, const FusionOutputMap &fusionByOutput)
    : original(original),
      sharedLibPath(std::move(sharedLibPath)),
      atirModulePath(std::move(atirModulePath)),
      fusionByOutput(fusionByOutput) {}

std::string ANNCFusedNodeBuilder::rewriteDataInput(
    const std::string &input) const {
  auto endpoint = parseTensorEndpoint(input);
  if (!endpoint.ok() || endpoint.value.control) return input;
  auto fusedIt = fusionByOutput.find(endpoint.value.canonicalDataName());
  if (fusedIt == fusionByOutput.end()) return input;
  return fusedIt->second.fusion->name + ":" +
         std::to_string(fusedIt->second.slot);
}

bool ANNCFusedNodeBuilder::replacesOutputNode(
    const std::string &nodeName) const {
  for (const auto &[name, target] : fusionByOutput) {
    (void)target;
    auto endpoint = parseTensorEndpoint(name);
    if (!endpoint.ok()) continue;
    if (!endpoint.value.control && endpoint.value.node == nodeName) return true;
  }
  return false;
}

// Appends Identity aliases named after each replaced original output node so
// downstream consumers and graph outputs keep working (the caller owns
// dropping the replaced nodes). *error, when non-null, gets the reason.
bool ANNCFusedNodeBuilder::appendOutputAliases(tensorflow::GraphDef &graph,
                                               std::string *error) const {
  using AliasTarget = std::pair<int64_t, const FusionOutputTarget *>;
  for (const tensorflow::NodeDef &originalNode : original.node()) {
    std::vector<AliasTarget> targets;
    for (const auto &[name, target] : fusionByOutput) {
      auto endpoint = parseTensorEndpoint(name);
      if (!endpoint.ok()) continue;
      if (!endpoint.value.control && endpoint.value.node == originalNode.name()) {
        targets.emplace_back(endpoint.value.port, &target);
      }
    }
    if (targets.empty()) continue;
    // Consumer checks run on the graph being built: a consumer that is
    // itself a replaced fusion output no longer counts.
    if (hasConsumers(graph, originalNode.name())) continue;
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
    // Roll back the just-appended alias on any late rejection.
    auto fail = [&](const std::string &reason) {
      if (error) *error = reason;
      graph.mutable_node()->RemoveLast();
      return false;
    };
    alias->set_name(originalNode.name());
    alias->set_op(requiredCount == 1 ? "Identity" : "IdentityN");
    alias->set_device(originalNode.device());
    std::vector<tensorflow::DataType> types;
    for (const auto &[port, target] : targets) {
      (void)port;
      if (!target->fusion || target->slot < 0 ||
          target->slot >=
              static_cast<int64_t>(target->fusion->outputs.size())) {
        return fail("output node '" + originalNode.name() +
                    "' has an invalid fusion output slot");
      }
      if (target->fusion->name == originalNode.name()) {
        return fail("fusion node name conflicts with output alias '" +
                    originalNode.name() + "'");
      }
      alias->add_input(target->fusion->name + ":" +
                       std::to_string(target->slot));
      tensorflow::DataType type =
          fusionDType(target->fusion->outputs[target->slot].dtype);
      if (type == tensorflow::DT_INVALID) {
        return fail("output node '" + originalNode.name() +
                    "' has an unsupported dtype");
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
    tensorflow::GraphDef &graph, const FusionInfo &fusion,
    std::string *error) const {
  if (fusion.outputs.empty()) return nullptr;  // contract: run validate first
  const std::vector<FusionArg> &outputs = fusion.outputs;
  const bool contextV3 = fusion.abi == "annc_context_v3";
  const bool grouped =
      contextV3 || fusion.abi == "mlir_ciface" ||
      fusion.abi == "annc_execution_v2";
  // Grouping carries each arg's pattern index: the permutation is
  // index-based, so duplicate tfNames (MatMul(x, x)) stay distinct.
  std::vector<FusionArg> orderedArgs;
  std::vector<int64_t> transportOfCanonical;
  if (grouped) {
    for (const FusionArg &arg : fusion.args) {
      if (arg.role != "constant" && arg.role != "fixed" &&
          arg.role != "dynamic") {
        if (error) {
          *error = "fusion '" + fusion.name + "' arg '" + arg.tfName +
                   "' has unknown role '" + arg.role + "'";
        }
        return nullptr;  // role grouping would silently drop it
      }
    }
    const int64_t argCount = static_cast<int64_t>(fusion.args.size());
    orderedArgs.reserve(fusion.args.size());
    for (const char *role : {"constant", "fixed", "dynamic"}) {
      for (int64_t i = 0; i < argCount; ++i) {
        if (fusion.args[i].role == role) {
          orderedArgs.push_back(fusion.args[i]);
          transportOfCanonical.push_back(i);
        }
      }
    }
    // Invert transport-position -> pattern index into the permutation.
    std::vector<int64_t> inverse(argCount, 0);
    for (int64_t p = 0; p < static_cast<int64_t>(orderedArgs.size()); ++p)
      inverse[transportOfCanonical[p]] = p;
    transportOfCanonical.swap(inverse);
  } else {
    orderedArgs.assign(fusion.args.begin(), fusion.args.end());
  }
  const std::vector<FusionArg> &args = orderedArgs;
  // Context v3 records the permutation in context_input_order, legacy ABIs
  // in kernel_arg_order (outputs appended; metadata may override).
  std::vector<int64_t> resolvedKernelArgOrder = fusion.kernelArgOrder;
  if (!contextV3 && grouped && resolvedKernelArgOrder.empty()) {
    resolvedKernelArgOrder = transportOfCanonical;
    for (size_t i = 0; i < outputs.size(); ++i)
      resolvedKernelArgOrder.push_back(
          static_cast<int64_t>(orderedArgs.size() + i));
  }
  const std::string &outputName = outputs.front().tfName;

  const tensorflow::NodeDef *outputNode =
      findNode(original, cleanTensorName(outputName));
  // outputRank() falls back to 2 (every generated pattern is rank-2).
  int rank = outputNode ? outputRank(*outputNode) : 2;
  constexpr tensorflow::DataType legacyType = tensorflow::DT_FLOAT;

  tensorflow::NodeDef *fused = graph.add_node();
  fused->set_name(fusion.name);
  fused->set_op("ANNCFused");
  fused->set_device(chooseFusionDevice(original, args, outputs));
  for (const FusionArg &arg : args) {
    fused->add_input(rewriteDataInput(resolveDataName(original, arg.tfName)));
  }

  auto *attrs = fused->mutable_attr();
  (*attrs)["kernel_name"].set_s(fusion.kernelName);
  (*attrs)["template_fingerprint"].set_s(fusion.templateFingerprint);
  if (fusion.executionMode == "jit") {
    (*attrs)["shared_lib_path"].set_s("");
    (*attrs)["atir_module_path"].set_s(atirModulePath);
  } else {
    (*attrs)["shared_lib_path"].set_s(sharedLibPath);
    (*attrs)["atir_module_path"].set_s("");
  }
  (*attrs)["abi"].set_s(fusion.abi);
  int64_t numOutputs = static_cast<int64_t>(outputs.size());
  (*attrs)["num_outputs"].set_i(numOutputs);
  (*attrs)["T"].set_type(legacyType);

  std::vector<tensorflow::DataType> inputTypes;
  inputTypes.reserve(args.size());
  for (const FusionArg &arg : args) {
    inputTypes.push_back(fusionInputDType(original, arg.tfName));
  }

  std::vector<tensorflow::DataType> constantTypes;
  std::vector<tensorflow::DataType> fixedTypes;
  std::vector<tensorflow::DataType> dynamicTypes;
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

  std::vector<tensorflow::DataType> outputTypes;
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
  // kernel_arg_order is emitted unconditionally (empty = "no override" for
  // the runtime); omitting it would differ from the v1 tool.
  auto *kernelArgOrderAttr = (*attrs)["kernel_arg_order"].mutable_list();
  for (int64_t value : resolvedKernelArgOrder)
    kernelArgOrderAttr->add_i(value);
  if (contextV3) {
    // Context v3: context_input_order locates each canonical input index.
    auto *inputOrder = (*attrs)["context_input_order"].mutable_list();
    for (int64_t value : transportOfCanonical) inputOrder->add_i(value);
    auto *contextAttrs = (*attrs)["context_i64_attrs"].mutable_list();
    for (int64_t value : fusion.contextI64Attrs) contextAttrs->add_i(value);
  }
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

namespace {

bool fusionEndpointsResolve(
    const FusionInfo &fusion, const tensorflow::GraphDef &graph,
    const ANNCFusedNodeBuilder::FusionOutputMap &fusionByOutput,
    std::string *reason) {
  // Outputs must resolve in the original graph itself: judging them against
  // fusion outputs would always pass and let a dangling site through.
  for (const FusionArg &output : fusion.outputs) {
    auto endpoint = parseTensorEndpoint(output.tfName);
    if (endpoint.ok() && !endpoint.value.control) {
      if (findNode(graph, endpoint.value.node) != nullptr) continue;
      std::string alt = slashVariant(endpoint.value.node);
      if (!alt.empty() && findNode(graph, alt) != nullptr) continue;
    }
    *reason = "output '" + output.tfName + "' is not a graph node";
    return false;
  }
  // A control edge at a replaced node would dangle once it is removed.
  for (const FusionArg &output : fusion.outputs) {
    auto endpoint = parseTensorEndpoint(output.tfName);
    if (!endpoint.ok() || endpoint.value.control) continue;
    for (const tensorflow::NodeDef &node : graph.node()) {
      for (const std::string &input : node.input()) {
        auto edge = parseTensorEndpoint(input);
        if (edge.ok() && edge.value.control &&
            edge.value.node == endpoint.value.node) {
          *reason = "control edge '^" + endpoint.value.node +
                    "' would dangle after replacement";
          return false;
        }
      }
    }
  }
  // Inputs may resolve in the original graph or in another fusion's output.
  auto inputResolves = [&](const std::string &name) {
    auto endpoint = parseTensorEndpoint(name);
    if (!endpoint.ok() || endpoint.value.control) return false;
    if (findNode(graph, endpoint.value.node) != nullptr) return true;
    std::string alt = slashVariant(endpoint.value.node);
    if (!alt.empty() && findNode(graph, alt) != nullptr) return true;
    return fusionByOutput.count(endpoint.value.canonicalDataName()) != 0;
  };
  for (const FusionArg &arg : fusion.args) {
    if (!inputResolves(arg.tfName)) {
      *reason = "input '" + arg.tfName + "' is not a graph node";
      return false;
    }
  }
  // Constant-role wiring must be Const/HostConst: CSE renames constants by
  // loc and names degrade (observed: begin wired to a strided_slice), which
  // feeds the kernel dynamic values. Non-Const: drop the site whole.
  for (const FusionArg &arg : fusion.args) {
    if (arg.role != "constant") continue;
    auto endpoint = parseTensorEndpoint(arg.tfName);
    if (!endpoint.ok() || endpoint.value.control) continue;  // filtered above
    const tensorflow::NodeDef *node = findNode(graph, endpoint.value.node);
    if (node == nullptr) {
      std::string alt = slashVariant(endpoint.value.node);
      node = alt.empty() ? nullptr : findNode(graph, alt);
    }
    if (node != nullptr && node->op() != "Const" && node->op() != "HostConst") {
      *reason = "constant-role input '" + arg.tfName + "' resolves to " +
                node->op() + " node";
      return false;
    }
  }
  return true;
}

// Drop unresolvable fusions until the set stabilizes (dropping one may
// unblock others), together with self-wired and cyclic sites (Kahn peeling
// leaves exactly the cyclic ones).
void dropUnresolvableFusions(std::vector<FusionInfo> &fusionInfos,
                             const tensorflow::GraphDef &graph) {
  bool changed = true;
  while (changed) {
    changed = false;
    ANNCFusedNodeBuilder::FusionOutputMap tentative;
    std::map<std::string, const std::string *> outputOwner;
    for (const FusionInfo &fusion : fusionInfos) {
      for (size_t slot = 0; slot < fusion.outputs.size(); ++slot) {
        auto endpoint = parseTensorEndpoint(fusion.outputs[slot].tfName);
        if (endpoint.ok() && !endpoint.value.control) {
          tentative.emplace(endpoint.value.canonicalDataName(),
                            ANNCFusedNodeBuilder::FusionOutputTarget{
                                &fusion, static_cast<int64_t>(slot)});
          outputOwner[endpoint.value.canonicalDataName()] = &fusion.name;
        }
      }
    }
    // Fusion-to-fusion dependency edges for the peeling pass.
    std::map<std::string, std::set<std::string>> deps;
    for (const FusionInfo &fusion : fusionInfos)
      deps[fusion.name];  // an entry for every site
    for (const FusionInfo &fusion : fusionInfos) {
      for (const FusionArg &arg : fusion.args) {
        auto endpoint = parseTensorEndpoint(arg.tfName);
        if (!endpoint.ok() || endpoint.value.control) continue;
        auto owner = outputOwner.find(endpoint.value.canonicalDataName());
        if (owner != outputOwner.end())
          deps[fusion.name].insert(*owner->second);
      }
    }
    std::set<std::string> cyclic;
    {
      std::set<std::string> peeledNames;
      bool progressed = true;
      while (progressed) {
        progressed = false;
        for (const auto &[name, targets] : deps) {
          if (peeledNames.count(name) != 0) continue;
          if (targets.count(name) != 0) continue;  // self-wiring: never peels
          if (std::all_of(targets.begin(), targets.end(),
                          [&](const std::string &target) {
                            return peeledNames.count(target) != 0;
                          })) {
            peeledNames.insert(name);
            progressed = true;
          }
        }
      }
      for (const auto &entry : deps)
        if (peeledNames.count(entry.first) == 0) cyclic.insert(entry.first);
    }
    std::vector<FusionInfo> kept;
    kept.reserve(fusionInfos.size());
    for (const FusionInfo &fusion : fusionInfos) {
      std::string reason;
      const bool onCycle = cyclic.count(fusion.name) != 0;
      if (onCycle ||
          !fusionEndpointsResolve(fusion, graph, tentative, &reason)) {
        if (onCycle) reason = "cyclic output wiring";
        std::cerr << "[annc-converter-v2] Warning: dropping fusion '"
                  << fusion.name << "' (" << fusion.pattern
                  << "): " << reason << "\n";
        changed = true;
        continue;
      }
      kept.push_back(fusion);
    }
    fusionInfos = std::move(kept);
  }
}

}  // namespace

bool rewriteGraphDefWithANNCFused(std::vector<FusionInfo> fusionInfos,
                                  const GraphDefRewriteOptions &options) {
  tensorflow::GraphDef graph;
  if (!readBinaryGraphDef(options.inputGraphPath, &graph) &&
      !readTextGraphDef(options.inputGraphPath, &graph)) {
    std::cerr << "[annc-converter-v2] Error: failed to read GraphDef: "
              << options.inputGraphPath << "\n";
    return false;
  }

  if (fusionInfos.empty()) {
    std::cerr << "[annc-converter-v2] Error: no ANNCFused metadata entries\n";
    return false;
  }

  dropUnresolvableFusions(fusionInfos, graph);
  if (fusionInfos.empty()) {
    std::cerr << "[annc-converter-v2] Warning: no usable fusion site; "
                 "writing the input graph unchanged\n";
    return writeBinaryGraphDef(graph, options.outputGraphPath);
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
    std::cerr << "[annc-converter-v2] Error: AOT fusion metadata requires "
                 "--shared_lib_path\n";
    return false;
  }
  if (needsJitTemplate && options.atirModulePath.empty()) {
    std::cerr << "[annc-converter-v2] Error: JIT fusion metadata requires "
                 "--atir_module_path\n";
    return false;
  }

  if (!options.kernelNameOverride.empty()) {
    if (fusionInfos.size() != 1) {
      std::cerr << "[annc-converter-v2] Error: --kernel_name override is only "
                   "supported for single-fusion metadata\n";
      return false;
    }
    fusionInfos.front().kernelName = options.kernelNameOverride;
  }

  ANNCFusedNodeBuilder::FusionOutputMap fusionByOutput;
  std::set<std::string> fusionNames;
  for (const auto &fusion : fusionInfos) {
    if (!fusionNames.insert(fusion.name).second) {
      std::cerr << "[annc-converter-v2] Error: duplicate fusion name: "
                << fusion.name << "\n";
      return false;
    }
    if (options.verbose) {
      std::cout << "[annc-converter-v2] Rewriting fusion node from ATIR: "
                << fusion.name << " pattern=" << fusion.pattern << "\n";
    }
    for (size_t slot = 0; slot < fusion.outputs.size(); ++slot) {
      auto endpoint = parseTensorEndpoint(fusion.outputs[slot].tfName);
      if (!endpoint.ok() || endpoint.value.control) {
        std::cerr
            << "[annc-converter-v2] Error: invalid fusion output endpoint: "
            << fusion.outputs[slot].tfName << "\n";
        return false;
      }
      auto [it, inserted] = fusionByOutput.emplace(
          endpoint.value.canonicalDataName(),
          ANNCFusedNodeBuilder::FusionOutputTarget{&fusion,
                                                   static_cast<int64_t>(slot)});
      if (!inserted) {
        std::cerr << "[annc-converter-v2] Error: duplicate fusion output "
                     "endpoint: "
                  << endpoint.value.canonicalDataName() << "\n";
        return false;
      }
    }
  }

  ANNCFusedNodeBuilder fusedNodeBuilder(graph, options.sharedLibPath,
                                        options.atirModulePath,
                                        fusionByOutput);
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
    if (fusedNodeBuilder.appendNode(rewritten, fusion) == nullptr) {
      std::cerr << "[annc-converter-v2] Error: failed to emit the ANNCFused "
                   "node for fusion '"
                << fusion.name << "'\n";
      return false;
    }
  }
  std::string aliasError;
  if (!fusedNodeBuilder.appendOutputAliases(rewritten, &aliasError)) {
    std::cerr << "[annc-converter-v2] Error: " << aliasError << "\n";
    return false;
  }

  return writeBinaryGraphDef(rewritten, options.outputGraphPath);
}

}  // namespace annc::converter_v2
