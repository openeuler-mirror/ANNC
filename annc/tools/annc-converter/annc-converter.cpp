#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <optional>
#include <set>
#include <map>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <sstream>
#include <unordered_set>
#include "llvm/ADT/DenseMap.h"

#include "Dialect/Atir/AtirOps.h"
#include "Adaptor/tensorflow/TFSavedModelParser.h"
#include "Helper.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/InitAllDialects.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/lib/core/status.h"
#include "google/protobuf/text_format.h"
#include "nlohmann/json.hpp"

using namespace mlir;
using namespace annc;

namespace {

static OwningOpRef<ModuleOp> parseModuleFromFile(StringRef path,
                                                 MLIRContext *context) {
  return parseSourceFile<ModuleOp>(path, context);
}

static std::string cleanTensorName(std::string name) {
  if (!name.empty() && name[0] == '^') name = name.substr(1);
  size_t colon = name.find(':');
  if (colon != std::string::npos) name = name.substr(0, colon);
  return name;
}

static std::string tensorSuffix(const std::string &name) {
  if (!name.empty() && name[0] == '^') return "";
  size_t colon = name.find(':');
  return colon == std::string::npos ? "" : name.substr(colon);
}

static const tensorflow::NodeDef *findNode(
    const tensorflow::GraphDef &graph, const std::string &name) {
  for (const auto &node : graph.node()) {
    if (node.name() == name) return &node;
  }
  return nullptr;
}

static int outputRank(const tensorflow::NodeDef &node) {
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

static tensorflow::DataType nodeDType(const tensorflow::NodeDef &node) {
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

static bool writeBinaryGraphDef(const tensorflow::GraphDef &graph,
                                const std::string &path) {
  std::string out;
  if (!graph.SerializeToString(&out)) return false;
  return tensorflow::WriteStringToFile(tensorflow::Env::Default(), path, out).ok();
}

static bool readBinaryGraphDef(const std::string &path,
                               tensorflow::GraphDef *graph) {
  std::string data;
  if (!tensorflow::ReadFileToString(tensorflow::Env::Default(), path, &data).ok()) {
    return false;
  }
  return graph->ParseFromString(data);
}

static bool readTextGraphDef(const std::string &path,
                             tensorflow::GraphDef *graph) {
  std::string data;
  if (!tensorflow::ReadFileToString(tensorflow::Env::Default(), path, &data).ok()) {
    return false;
  }
  return google::protobuf::TextFormat::ParseFromString(data, graph);
}

struct FusionInfo {
  std::string name;
  std::string pattern;
  std::string kernelName;
  std::string outputTensor;
  std::vector<std::string> originalNodes;
  std::vector<std::string> inputs;
  std::vector<std::vector<int64_t>> inputShapes;
  std::vector<int64_t> outputShape;
  std::string abi = "mlir_ciface";
  int64_t nConstants = 0;
  int64_t nFixed = 0;
  int64_t nDynamic = 0;
  int64_t numOutputs = 1;
  std::vector<int64_t> outputRanks;
  std::vector<int64_t> inputRanks;
  std::vector<int64_t> dynamicDims;
  std::vector<int64_t> kernelArgOrder;
  std::string symbolicSignature;
  std::string fallbackFunction;
};

static std::string formatRuntimeOutputShape(const FusionInfo &fusion) {
  std::unordered_set<int64_t> dynamicDims(fusion.dynamicDims.begin(),
                                          fusion.dynamicDims.end());
  std::string shape;
  for (size_t i = 0; i < fusion.outputShape.size(); ++i) {
    if (i > 0) shape += ",";
    if (dynamicDims.count(static_cast<int64_t>(i)) > 0) {
      shape += "?";
    } else {
      shape += std::to_string(fusion.outputShape[i]);
    }
  }
  return shape;
}

static std::optional<FusionInfo> parseFusionInfoJson(const nlohmann::json &j) {
  if (!j.is_object()) return std::nullopt;
  FusionInfo info;
  try {
    info.name = j.value("name", "");
    info.pattern = j.value("pattern", "");
    info.kernelName = j.value("kernel_name", "");
    info.outputTensor = j.value("output_tensor", "");
    info.originalNodes = j.value("original_nodes", std::vector<std::string>{});
    info.inputs = j.value("inputs", std::vector<std::string>{});
    info.inputShapes =
        j.value("input_shapes", std::vector<std::vector<int64_t>>{});
    info.outputShape = j.value("output_shape", std::vector<int64_t>{});
    info.abi = j.value("abi", "mlir_ciface");
    if (info.abi.empty()) info.abi = "mlir_ciface";
    info.nConstants = j.value("n_constants", int64_t{0});
    info.nFixed = j.value("n_fixed", int64_t{0});
    info.nDynamic = j.value("n_dynamic", int64_t{0});
    info.numOutputs = j.value("num_outputs", int64_t{1});
    info.outputRanks = j.value("output_ranks", std::vector<int64_t>{});
    info.inputRanks = j.value("input_ranks", std::vector<int64_t>{});
    info.dynamicDims = j.value("dynamic_dims", std::vector<int64_t>{});
    info.kernelArgOrder = j.value("kernel_arg_order", std::vector<int64_t>{});
    info.symbolicSignature = j.value("symbolic_signature", "");
    info.fallbackFunction = j.value("fallback_function", "");
  } catch (const std::exception &) {
    return std::nullopt;
  }

  if (info.name.empty() || info.originalNodes.empty() || info.inputs.empty() ||
      info.outputTensor.empty()) {
    return std::nullopt;
  }
  return info;
}

static std::vector<FusionInfo> readFusionInfosJson(const std::string &path) {
  std::ifstream in(path);
  if (!in.is_open()) return {};
  nlohmann::json j;
  try {
    in >> j;
  } catch (const std::exception &) {
    return {};
  }

  std::vector<FusionInfo> infos;
  if (j.is_object() && j.contains("fusions") && j["fusions"].is_array()) {
    for (const auto &item : j["fusions"]) {
      auto info = parseFusionInfoJson(item);
      if (info.has_value()) infos.push_back(std::move(*info));
    }
    return infos;
  }

  auto info = parseFusionInfoJson(j);
  if (info.has_value()) infos.push_back(std::move(*info));
  return infos;
}

static std::string chooseFusionDevice(const tensorflow::GraphDef &graph,
                                      const FusionInfo &fusion) {
  const tensorflow::NodeDef *output = findNode(graph, fusion.outputTensor);
  if (output && !output->device().empty()) return output->device();

  for (const auto &nodeName : fusion.originalNodes) {
    const tensorflow::NodeDef *node = findNode(graph, nodeName);
    if (node && !node->device().empty()) return node->device();
  }

  for (const auto &input : fusion.inputs) {
    const tensorflow::NodeDef *node = findNode(graph, cleanTensorName(input));
    if (node && !node->device().empty()) return node->device();
  }

  return "/job:localhost/replica:0/task:0/device:CPU:0";
}

static std::string rewriteFusionInput(
    const std::string &input,
    const std::unordered_map<std::string, const FusionInfo *> &fusionByOutput) {
  std::string prefix;
  std::string clean = input;
  if (!clean.empty() && clean[0] == '^') {
    prefix = "^";
    clean = clean.substr(1);
  }
  std::string src = cleanTensorName(clean);
  auto fusedIt = fusionByOutput.find(src);
  if (fusedIt == fusionByOutput.end()) return input;
  return prefix + fusedIt->second->name + tensorSuffix(clean);
}

static tensorflow::DataType fusionInputDType(
    const tensorflow::GraphDef &graph, const std::string &input) {
  const tensorflow::NodeDef *node = findNode(graph, cleanTensorName(input));
  return node ? nodeDType(*node) : tensorflow::DT_FLOAT;
}

static tensorflow::DataType firstOrDefault(
    ArrayRef<tensorflow::DataType> types,
    tensorflow::DataType fallback = tensorflow::DT_FLOAT) {
  return types.empty() ? fallback : types.front();
}

static void appendANNCFusedNode(tensorflow::GraphDef &rewritten,
                                const tensorflow::GraphDef &original,
                                const FusionInfo &fusion,
                                const std::string &sharedLibPath,
                                const std::unordered_map<
                                    std::string, const FusionInfo *> &fusionByOutput) {
  const tensorflow::NodeDef *reluNode = findNode(original, fusion.outputTensor);
  int rank = reluNode ? outputRank(*reluNode) : 2;
  tensorflow::DataType dtype =
      reluNode ? nodeDType(*reluNode) : tensorflow::DT_FLOAT;

  tensorflow::NodeDef *fused = rewritten.add_node();
  fused->set_name(fusion.name);
  fused->set_op("ANNCFused");
  fused->set_device(chooseFusionDevice(original, fusion));
  for (const auto &input : fusion.inputs) {
    fused->add_input(rewriteFusionInput(input, fusionByOutput));
  }

  auto *attrs = fused->mutable_attr();
  (*attrs)["kernel_name"].set_s(fusion.kernelName);
  (*attrs)["shared_lib_path"].set_s(sharedLibPath);
  (*attrs)["abi"].set_s(fusion.abi);
  (*attrs)["num_outputs"].set_i(fusion.numOutputs);
  (*attrs)["T"].set_type(dtype);

  SmallVector<tensorflow::DataType> inputTypes;
  inputTypes.reserve(fusion.inputs.size());
  for (const auto &input : fusion.inputs) {
    inputTypes.push_back(fusionInputDType(original, input));
  }

  SmallVector<tensorflow::DataType> constantTypes;
  SmallVector<tensorflow::DataType> fixedTypes;
  SmallVector<tensorflow::DataType> dynamicTypes;
  size_t inputIndex = 0;
  for (int64_t i = 0; i < fusion.nConstants && inputIndex < inputTypes.size();
       ++i, ++inputIndex) {
    constantTypes.push_back(inputTypes[inputIndex]);
  }
  for (int64_t i = 0; i < fusion.nFixed && inputIndex < inputTypes.size();
       ++i, ++inputIndex) {
    fixedTypes.push_back(inputTypes[inputIndex]);
  }
  for (int64_t i = 0; i < fusion.nDynamic && inputIndex < inputTypes.size();
       ++i, ++inputIndex) {
    dynamicTypes.push_back(inputTypes[inputIndex]);
  }

  SmallVector<tensorflow::DataType> outputTypes;
  for (int64_t i = 0; i < fusion.numOutputs; ++i) {
    outputTypes.push_back(dtype);
  }
  (*attrs)["Tconstants"].set_type(firstOrDefault(constantTypes));
  (*attrs)["Tfixed"].set_type(firstOrDefault(fixedTypes));
  (*attrs)["Tdynamic"].set_type(firstOrDefault(dynamicTypes));
  (*attrs)["Toutputs"].set_type(firstOrDefault(outputTypes, dtype));

  auto *rankList = (*attrs)["output_ranks"].mutable_list();
  if (fusion.outputRanks.empty()) {
    rankList->add_i(rank);
  } else {
    for (int64_t value : fusion.outputRanks) rankList->add_i(value);
  }
  auto *inputRanks = (*attrs)["input_ranks"].mutable_list();
  if (fusion.inputRanks.empty()) {
    for (const auto &shape : fusion.inputShapes) {
      inputRanks->add_i(static_cast<int64_t>(shape.size()));
    }
  } else {
    for (int64_t value : fusion.inputRanks) inputRanks->add_i(value);
  }
  auto *outputShapes = (*attrs)["output_shapes"].mutable_list();
  if (!fusion.outputShape.empty()) {
    outputShapes->add_s(formatRuntimeOutputShape(fusion));
  }
  auto *kernelArgOrder = (*attrs)["kernel_arg_order"].mutable_list();
  for (int64_t value : fusion.kernelArgOrder) kernelArgOrder->add_i(value);
  auto *dynamicDims = (*attrs)["dynamic_dims"].mutable_list();
  for (int64_t value : fusion.dynamicDims) dynamicDims->add_i(value);
  (*attrs)["symbolic_signature"].set_s(fusion.symbolicSignature);
  (*attrs)["Nconstants"].set_i(fusion.nConstants);
  (*attrs)["Nfixed"].set_i(fusion.nFixed);
  (*attrs)["Ndynamic"].set_i(fusion.nDynamic);
  (*attrs)["fallback_function"].mutable_func()->set_name(fusion.fallbackFunction);
  (*attrs)["fusion_pattern"].set_s(fusion.pattern);
  for (const auto &node : fusion.originalNodes) {
    (*attrs)["annc_original_nodes"].mutable_list()->add_s(node);
  }
}

static bool rewriteGraphDefWithANNCFused(
    std::vector<FusionInfo> fusionInfos, const std::string &inputGraphPath,
    const std::string &outputGraphPath, const std::string &kernelName,
    const std::string &sharedLibPath, bool textFormat, bool verbose) {
  tensorflow::GraphDef graph;
  if (!readBinaryGraphDef(inputGraphPath, &graph) &&
      !readTextGraphDef(inputGraphPath, &graph)) {
    llvm::errs() << "[annc-converter] Error: failed to read GraphDef: "
                 << inputGraphPath << "\n";
    return false;
  }

  if (fusionInfos.empty()) {
    llvm::errs() << "[annc-converter] Error: no ANNCFused metadata entries\n";
    return false;
  }

  if (!kernelName.empty()) {
    if (fusionInfos.size() != 1) {
      llvm::errs() << "[annc-converter] Error: --kernel_name override is only "
                      "supported for single-fusion metadata\n";
      return false;
    }
    fusionInfos.front().kernelName = kernelName;
  }

  std::set<std::string> removed;
  std::unordered_map<std::string, const FusionInfo *> fusionByOutput;
  for (const auto &fusion : fusionInfos) {
    if (verbose) {
      llvm::outs() << "[annc-converter] Rewriting fusion node from ATIR: "
                   << fusion.name << " pattern=" << fusion.pattern << "\n";
    }
    removed.insert(fusion.originalNodes.begin(), fusion.originalNodes.end());
    fusionByOutput[fusion.outputTensor] = &fusion;
  }

  for (const auto &fusion : fusionInfos) {
    for (const auto &node : graph.node()) {
      if (node.op() == "ReadVariableOp" && node.input_size() > 0) {
        std::string src = cleanTensorName(node.input(0));
        if (std::find(fusion.inputs.begin(), fusion.inputs.end(), src) !=
            fusion.inputs.end()) {
          removed.insert(node.name());
        }
      }
      if ((node.op() == "VarIsInitializedOp" ||
           node.op() == "AssignVariableOp") &&
          node.input_size() > 0) {
        std::string src = cleanTensorName(node.input(0));
        if (std::find(fusion.inputs.begin(), fusion.inputs.end(), src) !=
            fusion.inputs.end()) {
          removed.insert(node.name());
        }
      }
    }
  }

  tensorflow::GraphDef rewritten;
  for (const auto &node : graph.node()) {
    if (removed.count(node.name())) continue;
    tensorflow::NodeDef *out = rewritten.add_node();
    *out = node;

    out->clear_input();
    for (const auto &input : node.input()) {
      std::string rewrittenInput = rewriteFusionInput(input, fusionByOutput);
      if (rewrittenInput != input ||
          !removed.count(cleanTensorName(input))) {
        out->add_input(rewrittenInput);
      }
    }
  }

  for (const auto &fusion : fusionInfos) {
    appendANNCFusedNode(rewritten, graph, fusion, sharedLibPath, fusionByOutput);
  }

  if (textFormat) {
    return tensorflow::WriteStringToFile(tensorflow::Env::Default(), outputGraphPath,
                                         rewritten.DebugString()).ok();
  }
  return writeBinaryGraphDef(rewritten, outputGraphPath);
}

}  // namespace

// ============================================================================
// Step 1-2: Atir MLIR → NodeInfo 解析器
// ============================================================================

// ---- 辅助函数 ----

// dtype: atir::Type → TF dtype string
static std::string atirTypeToDtypeStr(Type elementType) {
  if (elementType.isF32()) return "float32";
  if (elementType.isF64()) return "float64";
  if (elementType.isF16()) return "float16";
  if (elementType.isInteger(8))  return "int8";
  if (elementType.isInteger(16)) return "int16";
  if (elementType.isInteger(32)) return "int32";
  if (elementType.isInteger(64)) return "int64";
  if (elementType.isInteger(1))  return "bool";
  return "unknown";
}

// base64 编码 (复用 TFSavedModelParser.cpp 的逻辑)
static std::string base64Encode(const std::vector<uint8_t> &data) {
  static const char chars[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string result;
  result.resize(data.size() * 4 / 3 + 4, '\0');
  size_t j = 0;
  size_t i = 0;
  for (; i + 2 < data.size(); i += 3) {
    uint32_t value = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
    result[j++] = chars[(value >> 18) & 0x3F];
    result[j++] = chars[(value >> 12) & 0x3F];
    result[j++] = chars[(value >> 6) & 0x3F];
    result[j++] = chars[value & 0x3F];
  }
  if (i < data.size()) {
    uint32_t value = data[i] << 16;
    if (i + 1 < data.size()) value |= data[i + 1] << 8;
    result[j++] = chars[(value >> 18) & 0x3F];
    result[j++] = chars[(value >> 12) & 0x3F];
    if (i + 1 < data.size())
      result[j++] = chars[(value >> 6) & 0x3F];
    else
      result[j++] = '=';
    result[j++] = '=';
  }
  result.resize(j);
  return result;
}

// base64 解码
static std::vector<uint8_t> base64Decode(const std::string& encoded) {
  static const int8_t lookup[] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
    52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
    15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
    -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
    41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  };

  std::vector<uint8_t> result;
  if (encoded.empty()) return result;

  int val = 0, valb = -8;
  for (unsigned char c : encoded) {
    if (lookup[c] == -1) break;
    val = (val << 6) + lookup[c];
    valb += 6;
    if (valb >= 0) {
      result.push_back(static_cast<uint8_t>((val >> valb) & 0xFF));
      valb -= 8;
    }
  }
  return result;
}
static std::vector<uint8_t> denseElementsToRawBytes(DenseElementsAttr attr) {
  if (!attr || attr.empty()) return {};

  // 尝试获取原始字节 (适用于 float/int 等连续存储类型)
  // DenseElementsAttr 在元素数量较大时使用 raw_data 模式
  if (attr.isSplat()) {
    // 单值: 需要复制 N 次
    auto shapedType = dyn_cast<ShapedType>(attr.getType());
    if (!shapedType) return {};
    size_t elemSize = shapedType.getElementTypeBitWidth() / 8;
    size_t numElems = shapedType.getNumElements();

    // 读取单值的字节
    std::vector<uint8_t> singleVal(elemSize);
    auto rawSplat = attr.getRawData();
    memcpy(singleVal.data(), rawSplat.data(), elemSize);

    // 复制 N 次
    std::vector<uint8_t> result(numElems * elemSize);
    for (size_t i = 0; i < numElems; i++) {
      memcpy(result.data() + i * elemSize, singleVal.data(), elemSize);
    }
    return result;
  }

  // 非单值: 直接获取原始字节
  auto rawData = attr.getRawData();
  return std::vector<uint8_t>(rawData.begin(), rawData.end());
}

// shape: 处理动态维度 (ShapedType::kDynamic → -1)
static std::vector<int64_t> getShapeVector(atir::TensorType tensorType) {
  auto shape = tensorType.getShape();
  std::vector<int64_t> result;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic) {
      result.push_back(-1);
    } else {
      result.push_back(dim);
    }
  }
  return result;
}

// 从 Value 获取节点名
// - BlockArgument: 通过 NameLoc 提取
// - Op result: 通过 NameLoc 提取 op 名，多输出加 :N 后缀
static std::string getValueNodeName(Value val) {
  // 先尝试 NameLoc
  std::string name = getValLocName(val);
  if (!name.empty()) return name;

  // 回退: 如果是 Op result，用 op 名
  if (auto *defOp = val.getDefiningOp()) {
    name = getLocName(defOp);
    if (!name.empty()) {
      // 多输出: 加 :N 后缀
      auto results = defOp->getResults();
      if (results.size() > 1) {
        for (unsigned i = 0; i < results.size(); i++) {
          if (results[i] == val) {
            return name + ":" + std::to_string(i);
          }
        }
      }
      return name;
    }
  }

  // 最终回退: 用 BlockArgument 编号
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    return "arg" + std::to_string(blockArg.getArgNumber());
  }

  return "unknown";
}

// ============================================================================
// AtirMLIRParser — 核心解析类
// ============================================================================

class AtirMLIRParser {
public:
  AtirMLIRParser(MLIRContext &context) : context_(context) {}

  // 主入口: 解析 ModuleOp → vector<NodeInfo>
  std::vector<NodeInfo> parse(ModuleOp module) {
    nodes_.clear();
    valueToName_.clear();

    // 遍历所有 func.func
    for (auto funcOp : module.getOps<func::FuncOp>()) {
      parseFuncOp(funcOp);
    }

    return std::move(nodes_);
  }

private:
  MLIRContext &context_;
  std::vector<NodeInfo> nodes_;
  llvm::DenseMap<Value, std::string> valueToName_;

  // 注册 Value → name 映射
  void registerValueName(Value val, const std::string &name) {
    valueToName_[val] = name;
  }

  // 查询 Value 对应的节点名
  std::string lookupValueName(Value val) {
    auto it = valueToName_.find(val);
    if (it != valueToName_.end()) return it->second;
    return getValueNodeName(val);
  }

  // 解析 func.func
  void parseFuncOp(func::FuncOp funcOp) {
    // 1. 提取函数输入 → Placeholder NodeInfo
    for (BlockArgument arg : funcOp.getArguments()) {
      auto tensorType = dyn_cast<atir::TensorType>(arg.getType());
      if (!tensorType) continue;

      NodeInfo node;
      node.name = getValueNodeName(arg);
      node.op_type = "Placeholder";
      node.isInputNode = true;

      std::vector<int64_t> shape = getShapeVector(tensorType);
      std::string dtype = atirTypeToDtypeStr(tensorType.getElementType());
      node.addOutput(0, node.name, shape, dtype);

      registerValueName(arg, node.name);
      nodes_.push_back(std::move(node));
    }

    // 2. 遍历函数体中的所有 Op (单 block 函数)
    // 注: ForOp/IfOp 含 region 多 block，当前 MVP 暂不处理
    for (Operation &op : funcOp.getBody().front()) {
      dispatchOp(&op);
    }

    // 3. 标记输出节点 (从 return op 的操作数追溯)
    for (Block &block : funcOp.getBody()) {
      if (auto retOp = dyn_cast<func::ReturnOp>(block.getTerminator())) {
        for (Value operand : retOp.getOperands()) {
          std::string name = lookupValueName(operand);
          for (auto &node : nodes_) {
            if (node.name == name) {
              node.isOutputNode = true;
              break;
            }
          }
        }
      }
    }
  }

  // Op 分发
  void dispatchOp(Operation *op) {
    // 忽略终止符
    if (isa<func::ReturnOp>(op)) return;

    // 忽略 BufferOp (内部产物)
    if (isa<atir::BufferOp>(op)) return;

    // 忽略 NoneOp
    if (isa<atir::NoneOp>(op)) return;

    // 忽略 Atir_ReturnOp (终止符，非计算)
    if (isa<atir::ReturnOp>(op)) return;

    // 含 region 的 Op: 暂不支持，报错
    if (isa<atir::ForOp>(op)) {
      llvm::errs() << "[annc-converter] Error: atir.ForOp not supported "
                      "(contains region, cannot reverse to TF)\n";
      return;
    }
    if (isa<atir::IfOp>(op)) {
      llvm::errs() << "[annc-converter] Error: atir.IfOp not supported "
                      "(contains region, cannot reverse to TF)\n";
      return;
    }
    if (isa<atir::SwitchCaseOp>(op)) {
      llvm::errs() << "[annc-converter] Error: atir.SwitchCaseOp not supported "
                      "(contains region, cannot reverse to TF)\n";
      return;
    }
    if (isa<atir::ParallelOp>(op)) {
      llvm::errs() << "[annc-converter] Error: atir.ParallelOp not supported "
                      "(contains region, cannot reverse to TF)\n";
      return;
    }

    // ConstantOp
    if (auto constOp = dyn_cast<atir::ConstantOp>(op)) {
      parseConstantOp(constOp);
      return;
    }

    // VariableOp
    if (auto varOp = dyn_cast<atir::VariableOp>(op)) {
      parseVariableOp(varOp);
      return;
    }

    // MatMulOp
    if (auto matmulOp = dyn_cast<atir::MatMulOp>(op)) {
      parseMatMulOp(matmulOp);
      return;
    }

    // AddOp
    if (auto addOp = dyn_cast<atir::AddOp>(op)) {
      parseAddOp(addOp);
      return;
    }

    // ReluOp
    if (auto reluOp = dyn_cast<atir::ReluOp>(op)) {
      parseReluOp(reluOp);
      return;
    }

    // ConcatOp
    if (auto concatOp = dyn_cast<atir::ConcatOp>(op)) {
      parseConcatOp(concatOp);
      return;
    }

    // LoadOp → 不支持 (缺 StridedSlice 必需属性)
    if (auto loadOp = dyn_cast<atir::LoadOp>(op)) {
      parseLoadOp(loadOp);
      return;
    }

    // CustomizeOp
    if (auto customOp = dyn_cast<atir::CustomizeOp>(op)) {
      parseCustomizeOp(customOp);
      return;
    }

    // 未识别的 Op: 跳过并警告
    llvm::errs() << "[annc-converter] Warning: unsupported op: "
                 << op->getName() << "\n";
  }

  // ---- 各 Op 的解析实现 ----

  void parseConstantOp(atir::ConstantOp op) {
    NodeInfo node;
    node.name = getLocName(op);
    node.op_type = "Const";

    auto tensorType = dyn_cast<atir::TensorType>(op.getData().getType());
    if (!tensorType) return;

    std::vector<int64_t> shape = getShapeVector(tensorType);
    std::string dtype = atirTypeToDtypeStr(tensorType.getElementType());
    node.addOutput(0, node.name, shape, dtype);

    // 提取常量数据 (cacheData 中的 DenseElementsAttr)
    auto cacheData = tensorType.getCacheData();
    if (cacheData && !cacheData.empty()) {
      std::vector<uint8_t> rawBytes = denseElementsToRawBytes(cacheData);
      if (!rawBytes.empty()) {
        node.raw_data = base64Encode(rawBytes);
      }
    }

    registerValueName(op.getResult(), node.name);
    nodes_.push_back(std::move(node));
  }

  // Atir VariableOp 没有嵌入数据，TF 的 VariableV2 需要 checkpoint 文件
  // MVP: 转为 Placeholder (作为模型输入，由调用方提供初始值)
  void parseVariableOp(atir::VariableOp op) {
    NodeInfo node;
    node.name = getLocName(op);
    node.op_type = "Placeholder";
    node.isInputNode = true;

    auto tensorType = dyn_cast<atir::TensorType>(op.getOutput().getType());
    if (!tensorType) return;

    std::vector<int64_t> shape = getShapeVector(tensorType);
    std::string dtype = atirTypeToDtypeStr(tensorType.getElementType());
    node.addOutput(0, node.name, shape, dtype);

    registerValueName(op.getResult(), node.name);
    nodes_.push_back(std::move(node));
  }

  void parseMatMulOp(atir::MatMulOp op) {
    NodeInfo node;
    node.name = getLocName(op);
    node.op_type = "MatMul";

    // 输入: lhs, rhs (跳过 C/BufferOp 的输入)
    node.inputs.push_back(lookupValueName(op.getLhs()));
    node.inputs.push_back(lookupValueName(op.getRhs()));

    // bias: withBias=true 时添加
    if (op.getWithBias()) {
      auto bias = op.getBias();
      if (bias && bias != mlir::Value()) {
        node.inputs.push_back(lookupValueName(bias));
      }
    }

    auto tensorType = dyn_cast<atir::TensorType>(op.getOutput().getType());
    if (tensorType) {
      std::vector<int64_t> shape = getShapeVector(tensorType);
      std::string dtype = atirTypeToDtypeStr(tensorType.getElementType());
      node.addOutput(0, node.name, shape, dtype);
    }

    // TF 属性: transpose_a, transpose_b
    node.tf_attrs["transpose_a"] = "b:" + std::string(op.getLeftTranspose() ? "true" : "false");
    node.tf_attrs["transpose_b"] = "b:" + std::string(op.getRightTranspose() ? "true" : "false");

    // 记录融合属性供 Step 3 使用
    if (op.getWithBias()) node.tf_attrs["_fused_withBias"] = "true";
    if (op.getDoRelu()) {
      node.tf_attrs["_fused_do_relu"] = "true";
      if (op.getReluLimit().convertToDouble() > 0.0)
        node.tf_attrs["_fused_relu_limit"] = std::to_string(op.getReluLimit().convertToDouble());
    }

    // 当前先不拆分，直接输出为 MatMul 节点 (融合拆分在 Step 3 splitFusedNodes 中处理)

    registerValueName(op.getResult(), node.name);
    nodes_.push_back(std::move(node));
  }

  void parseAddOp(atir::AddOp op) {
    NodeInfo node;
    node.name = getLocName(op);
    node.op_type = "AddV2";

    // variadic inputs
    for (Value input : op.getInputs()) {
      node.inputs.push_back(lookupValueName(input));
    }

    auto tensorType = dyn_cast<atir::TensorType>(op.getOutput().getType());
    if (tensorType) {
      std::vector<int64_t> shape = getShapeVector(tensorType);
      std::string dtype = atirTypeToDtypeStr(tensorType.getElementType());
      node.addOutput(0, node.name, shape, dtype);
    }

        // 记录融合属性供 Step 3 splitFusedNodes 使用
    if (op.getDoRelu()) {
      node.tf_attrs["_fused_do_relu"] = "true";
      if (op.getReluLimit().convertToDouble() > 0.0)
        node.tf_attrs["_fused_relu_limit"] = std::to_string(op.getReluLimit().convertToDouble());
    }

    registerValueName(op.getResult(), node.name);
    nodes_.push_back(std::move(node));
  }

  void parseReluOp(atir::ReluOp op) {
    NodeInfo node;
    node.name = getLocName(op);
    node.op_type = "Relu";

    node.inputs.push_back(lookupValueName(op.getInput()));

    auto tensorType = dyn_cast<atir::TensorType>(op.getOutput().getType());
    if (tensorType) {
      std::vector<int64_t> shape = getShapeVector(tensorType);
      std::string dtype = atirTypeToDtypeStr(tensorType.getElementType());
      node.addOutput(0, node.name, shape, dtype);
    }

    registerValueName(op.getResult(), node.name);
    nodes_.push_back(std::move(node));
  }

  void parseConcatOp(atir::ConcatOp op) {
    NodeInfo node;
    node.name = getLocName(op);
    node.op_type = "ConcatV2";

    for (Value input : op.getInputs()) {
      node.inputs.push_back(lookupValueName(input));
    }

    auto tensorType = dyn_cast<atir::TensorType>(op.getOutput().getType());
    if (tensorType) {
      std::vector<int64_t> shape = getShapeVector(tensorType);
      std::string dtype = atirTypeToDtypeStr(tensorType.getElementType());
      node.addOutput(0, node.name, shape, dtype);
    }

    // ConcatV2: axis 是最后一个输入 (TF 要求)，不是属性
    // 创建 axis Const 节点
    int64_t axis = op.getAxis();
    std::string axisNodeName = node.name + "/axis";

    NodeInfo axisNode;
    axisNode.name = axisNodeName;
    axisNode.op_type = "Const";
    axisNode.addOutput(0, axisNodeName, {}, "int32");  // scalar
    // axis 值 → raw bytes → base64
    int32_t axisVal = static_cast<int32_t>(axis);
    std::vector<uint8_t> axisBytes(
        reinterpret_cast<const uint8_t*>(&axisVal),
        reinterpret_cast<const uint8_t*>(&axisVal) + sizeof(int32_t));
    axisNode.raw_data = base64Encode(axisBytes);
    nodes_.push_back(std::move(axisNode));

    // axis 作为最后一个输入
    node.inputs.push_back(axisNodeName);

    // 记录融合属性供 Step 3 splitFusedNodes 使用
    if (op.getDoRelu()) {
      node.tf_attrs["_fused_do_relu"] = "true";
      if (op.getReluLimit().convertToDouble() > 0.0)
        node.tf_attrs["_fused_relu_limit"] = std::to_string(op.getReluLimit().convertToDouble());
    }

    registerValueName(op.getResult(), node.name);
    nodes_.push_back(std::move(node));
  }

  // Atir LoadOp 缺少 TF StridedSlice 必需属性 (begin_mask/end_mask 等)
  // MVP: 报错跳过
  void parseLoadOp(atir::LoadOp op) {
    llvm::errs() << "[annc-converter] Error: atir.LoadOp → StridedSlice not supported "
                     "(missing required TF attributes: begin_mask, end_mask, etc.)\n";
    // 仍然注册 Value 名称，避免下游引用断裂
    std::string name = getLocName(op);
    registerValueName(op.getResult(), name);
  }

  void parseCustomizeOp(atir::CustomizeOp op) {
    NodeInfo node;
    node.name = getLocName(op);
    // opType 属性作为 TF op 类型
    node.op_type = op.getOpType().str();

    for (Value input : op.getInputs()) {
      node.inputs.push_back(lookupValueName(input));
    }

    for (unsigned i = 0; i < op.getResults().size(); i++) {
      auto tensorType = dyn_cast<atir::TensorType>(op.getResult(i).getType());
      if (tensorType) {
        std::vector<int64_t> shape = getShapeVector(tensorType);
        std::string dtype = atirTypeToDtypeStr(tensorType.getElementType());
        std::string outName = node.name;
        if (op.getResults().size() > 1) {
          outName += ":" + std::to_string(i);
        }
        node.addOutput(i, outName, shape, dtype);
      }
    }

    // 多输出: 为每个 result 注册名称
    if (op.getResults().size() == 1) {
      registerValueName(op.getResult(0), node.name);
    } else {
      for (unsigned i = 0; i < op.getResults().size(); i++) {
        registerValueName(op.getResult(i), node.name + ":" + std::to_string(i));
      }
    }

    nodes_.push_back(std::move(node));
  }
};

// ============================================================================
// Step 3: 融合节点拆分
// ============================================================================

// 核心拆分函数: 后处理 NodeInfo 列表，拆分融合节点
// 算法: 两遍 —
//   第一遍: 拆分融合节点，记录 name → 最终名 映射
//   第二遍: 用映射表修正所有输入引用
static std::vector<NodeInfo> splitFusedNodes(const std::vector<NodeInfo> &inputNodes) {
  // name mapping: 原节点名 → 拆分链末尾节点名
  std::map<std::string, std::string> nameMap;
  // 拆分链中新增的节点名 (它们的输入已正确，不需要重写)
  std::set<std::string> splitChainNames;
  std::vector<NodeInfo> result;

  for (const auto &node : inputNodes) {
    bool hasFusedBias = (node.tf_attrs.count("_fused_withBias") &&
                         node.tf_attrs.at("_fused_withBias") == "true");
    bool hasFusedRelu = (node.tf_attrs.count("_fused_do_relu") &&
                         node.tf_attrs.at("_fused_do_relu") == "true");

    // ---- MatMul(withBias) → MatMul + BiasAdd [→ Relu] ----
    if (node.op_type == "MatMul" && hasFusedBias) {
      // 1. MatMul: 去掉 bias 输入，去掉融合属性，不是最终输出
      NodeInfo matmul = node;
      if (matmul.inputs.size() >= 3) {
        matmul.inputs.pop_back();
      }
      matmul.tf_attrs.erase("_fused_withBias");
      matmul.tf_attrs.erase("_fused_do_relu");
      matmul.tf_attrs.erase("_fused_relu_limit");
      matmul.isOutputNode = false;
      result.push_back(std::move(matmul));

      // 2. BiasAdd: 输入 = matmul输出 + bias
      NodeInfo biasAdd;
      std::string biasAddName = node.name + "/bias_add";
      biasAdd.name = biasAddName;
      biasAdd.op_type = "BiasAdd";
      biasAdd.inputs.push_back(node.name);           // matmul output
      biasAdd.inputs.push_back(node.inputs.back());   // bias (原第3个输入)
      if (!node.outputs.empty()) {
        biasAdd.addOutput(0, biasAddName,
                          node.outputs[0].shape, node.outputs[0].dtype);
      }

      if (hasFusedRelu) {
        biasAdd.isOutputNode = false;
        result.push_back(std::move(biasAdd));
        splitChainNames.insert(biasAddName);

        // 3. Relu: 输入 = biasAdd输出
        NodeInfo relu;
        std::string reluName = node.name + "/relu";
        relu.name = reluName;
        relu.op_type = "Relu";
        relu.inputs.push_back(biasAddName);
        if (!node.outputs.empty()) {
          relu.addOutput(0, reluName,
                        node.outputs[0].shape, node.outputs[0].dtype);
        }
        relu.isOutputNode = node.isOutputNode;
        result.push_back(std::move(relu));
        splitChainNames.insert(reluName);

        // 映射: 原名 → 最终名 (Relu)
        nameMap[node.name] = reluName;
      } else {
        biasAdd.isOutputNode = node.isOutputNode;
        result.push_back(std::move(biasAdd));
        splitChainNames.insert(biasAddName);

        // 映射: 原名 → BiasAdd
        nameMap[node.name] = biasAddName;
      }
      continue;
    }

    // ---- MatMul(do_relu) / Add(do_relu) / ConcatV2(do_relu) → Op + Relu ----
    if (hasFusedRelu &&
        (node.op_type == "MatMul" || node.op_type == "AddV2" ||
         node.op_type == "ConcatV2")) {
      // 1. 原算子: 去掉融合属性，不是最终输出
      NodeInfo baseOp = node;
      baseOp.tf_attrs.erase("_fused_do_relu");
      baseOp.tf_attrs.erase("_fused_relu_limit");
      baseOp.isOutputNode = false;
      result.push_back(std::move(baseOp));

      // 2. Relu: 输入 = 原算子输出
      NodeInfo relu;
      std::string reluName = node.name + "/relu";
      relu.name = reluName;
      relu.op_type = "Relu";
      relu.inputs.push_back(node.name);
      if (!node.outputs.empty()) {
        relu.addOutput(0, reluName,
                      node.outputs[0].shape, node.outputs[0].dtype);
      }
      relu.isOutputNode = node.isOutputNode;
      result.push_back(std::move(relu));
      splitChainNames.insert(reluName);

      // 映射: 原名 → Relu
      nameMap[node.name] = reluName;
      continue;
    }

    // ---- 无融合: 直接保留，清理残留 _fused_ 属性 ----
    NodeInfo clean = node;
    clean.tf_attrs.erase("_fused_withBias");
    clean.tf_attrs.erase("_fused_do_relu");
    clean.tf_attrs.erase("_fused_relu_limit");
    result.push_back(std::move(clean));
  }

  // 第二遍: 用映射表修正下游节点的输入引用
  // 跳过拆分链内部节点 (它们的输入已正确)
  for (auto &node : result) {
    if (splitChainNames.count(node.name)) continue;
    for (auto &input : node.inputs) {
      auto it = nameMap.find(input);
      if (it != nameMap.end()) {
        input = it->second;
      }
    }
  }

  return result;
}

// ----------------------------------------------------------------------------
// Step 4: GraphDef 构建
// ----------------------------------------------------------------------------

// 拓扑排序 (Kahn算法)
static std::vector<NodeInfo> topologicalSort(const std::vector<NodeInfo>& nodes) {
  std::unordered_map<std::string, size_t> node_to_index;
  for (size_t i = 0; i < nodes.size(); ++i) {
    node_to_index[nodes[i].name] = i;
  }

  std::vector<int> in_degree(nodes.size(), 0);
  std::vector<std::vector<size_t>> adj_list(nodes.size());

  // 构建依赖关系
  for (size_t i = 0; i < nodes.size(); ++i) {
    const NodeInfo& node = nodes[i];
    for (const std::string& input_name : node.inputs) {
      if (node_to_index.find(input_name) != node_to_index.end()) {
        size_t input_index = node_to_index[input_name];
        adj_list[input_index].push_back(i);
        in_degree[i]++;
      }
    }
  }

  // Kahn算法
  std::queue<size_t> q;
  std::vector<NodeInfo> sorted_nodes;

  for (size_t i = 0; i < nodes.size(); ++i) {
    if (in_degree[i] == 0) {
      q.push(i);
    }
  }

  while (!q.empty()) {
    size_t current = q.front();
    q.pop();
    sorted_nodes.push_back(nodes[current]);

    for (size_t neighbor : adj_list[current]) {
      in_degree[neighbor]--;
      if (in_degree[neighbor] == 0) {
        q.push(neighbor);
      }
    }
  }

  // 检测环
  if (sorted_nodes.size() != nodes.size()) {
    llvm::errs() << "[annc-converter] Warning: Cycle detected in graph, using original order\n";
    return nodes;
  }

  return sorted_nodes;
}

// dtype string → tensorflow::DataType
static tensorflow::DataType stringToTFDataType(const std::string& dtype) {
  if (dtype == "float32") return tensorflow::DT_FLOAT;
  if (dtype == "float64") return tensorflow::DT_DOUBLE;
  if (dtype == "float16") return tensorflow::DT_HALF;
  if (dtype == "int8") return tensorflow::DT_INT8;
  if (dtype == "int16") return tensorflow::DT_INT16;
  if (dtype == "int32") return tensorflow::DT_INT32;
  if (dtype == "int64") return tensorflow::DT_INT64;
  if (dtype == "bool") return tensorflow::DT_BOOL;
  return tensorflow::DT_INVALID;
}

// tf_attrs value helper: 带类型标记的属性值
// 格式: "b:true", "b:false"     → bool
//        "i:42", "i:-1"          → int64
//        "f:3.14"                → float
//        "s:hello"               → string
// 无前缀: 尝试推断 (兼容旧数据)
static tensorflow::AttrValue parseAttrValue(const std::string& value) {
  tensorflow::AttrValue attr_value;

  // 带类型前缀
  if (value.rfind("b:", 0) == 0) {
    attr_value.set_b(value.substr(2) == "true");
    return attr_value;
  }
  if (value.rfind("i:", 0) == 0) {
    attr_value.set_i(std::stol(value.substr(2)));
    return attr_value;
  }
  if (value.rfind("f:", 0) == 0) {
    attr_value.set_f(std::stod(value.substr(2)));
    return attr_value;
  }
  if (value.rfind("s:", 0) == 0) {
    attr_value.set_s(value.substr(2));
    return attr_value;
  }

  // 无前缀: 兼容推断
  if (value == "true" || value == "false") {
    attr_value.set_b(value == "true");
  } else {
    // 默认为字符串 (TF属性很少用裸数字)
    attr_value.set_s(value);
  }
  return attr_value;
}

// NodeInfo → tensorflow::NodeDef
static tensorflow::NodeDef nodeInfoToNodeDef(const NodeInfo& node) {
  tensorflow::NodeDef node_def;
  node_def.set_name(node.name);
  node_def.set_op(node.op_type);

  // 设置输入
  for (const auto& input : node.inputs) {
    node_def.add_input(input);
  }

  // 设置通用 TF 属性 (从 outputs 信息自动推断)
  if (!node.outputs.empty()) {
    auto* attr = node_def.mutable_attr();
    tensorflow::DataType dtype = stringToTFDataType(node.outputs[0].dtype);

    // Placeholder / Const 需要 dtype + shape 属性
    if (node.op_type == "Placeholder" || node.op_type == "Const") {
      tensorflow::AttrValue dtype_attr;
      dtype_attr.set_type(dtype);
      (*attr)["dtype"] = dtype_attr;

      tensorflow::AttrValue shape_attr;
      auto* shape_proto = shape_attr.mutable_shape();
      for (int64_t dim : node.outputs[0].shape) {
        shape_proto->add_dim()->set_size(dim);
      }
      (*attr)["shape"] = shape_attr;
    }

    // 计算节点需要 T 属性 (元素类型)
    // TF约定: T属性表示输入/输出的元素类型
    if (node.op_type == "MatMul" || node.op_type == "AddV2" ||
        node.op_type == "BiasAdd" || node.op_type == "Relu" ||
        node.op_type == "ConcatV2") {
      tensorflow::AttrValue t_attr;
      t_attr.set_type(dtype);
      (*attr)["T"] = t_attr;
    }

    // _output_shapes: 输出形状信息 (TF 工具链可能期望)
    tensorflow::AttrValue output_shapes_attr;
    auto* shape_list = output_shapes_attr.mutable_list();
    for (const auto& output : node.outputs) {
      auto* shape_proto = shape_list->add_shape();
      for (int64_t dim : output.shape) {
        shape_proto->add_dim()->set_size(dim);
      }
    }
    (*attr)["_output_shapes"] = output_shapes_attr;
  }

  // 设置 TF 特有属性 (从 tf_attrs)
  for (const auto& [key, value] : node.tf_attrs) {
    auto* attr = node_def.mutable_attr();
    (*attr)[key] = parseAttrValue(value);
  }

  // 常量数据: base64 解码 → TensorProto
  if (!node.raw_data.empty() && node.op_type == "Const") {
    auto* attr = node_def.mutable_attr();
    tensorflow::AttrValue tensor_attr;
    tensorflow::TensorProto* tensor = tensor_attr.mutable_tensor();

    if (!node.outputs.empty()) {
      tensor->set_dtype(stringToTFDataType(node.outputs[0].dtype));
      for (int64_t dim : node.outputs[0].shape) {
        tensor->mutable_tensor_shape()->add_dim()->set_size(dim);
      }
    }

    // base64 解码 → tensor_content (raw bytes)
    std::vector<uint8_t> decoded = base64Decode(node.raw_data);
    if (!decoded.empty()) {
      tensor->set_tensor_content(
          std::string(decoded.begin(), decoded.end()));
    }

    (*attr)["value"] = tensor_attr;
  }

  return node_def;
}

// 构建SavedModel
static tensorflow::SavedModel buildSavedModel(const std::vector<NodeInfo>& nodes) {
  tensorflow::SavedModel saved_model;

  // 创建MetaGraphDef
  auto* meta_graph = saved_model.add_meta_graphs();
  meta_graph->mutable_meta_info_def()->set_meta_graph_version("1.0");

  // 构建GraphDef
  auto* graph_def = meta_graph->mutable_graph_def();

  for (const auto& node : nodes) {
    auto* node_def = graph_def->add_node();
    *node_def = nodeInfoToNodeDef(node);
  }

  // 创建signature_def (serving_default)
  auto* signatures = meta_graph->mutable_signature_def();
  auto& sig_def = (*signatures)["serving_default"];
  sig_def.set_method_name("tensorflow/serving/predict");

  // 输入信息
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  for (const auto& node : nodes) {
    if (node.isInputNode) {
      input_names.push_back(node.name);
    }
    if (node.isOutputNode) {
      output_names.push_back(node.name);
    }
  }

  // 设置输入
  for (const auto& name : input_names) {
    auto* input_info = sig_def.mutable_inputs();
    tensorflow::TensorInfo tensor_info;
    tensor_info.set_name(name);
    // 查找对应节点的实际dtype
    for (const auto& node : nodes) {
      if (node.name == name && !node.outputs.empty()) {
        tensor_info.set_dtype(stringToTFDataType(node.outputs[0].dtype));
        auto* shape = tensor_info.mutable_tensor_shape();
        for (int64_t dim : node.outputs[0].shape) {
          shape->add_dim()->set_size(dim);
        }
        break;
      }
    }
    (*input_info)[name] = tensor_info;
  }

  // 设置输出
  for (const auto& name : output_names) {
    auto* output_info = sig_def.mutable_outputs();
    tensorflow::TensorInfo tensor_info;
    tensor_info.set_name(name);
    // 查找对应节点的实际dtype
    for (const auto& node : nodes) {
      if (node.name == name && !node.outputs.empty()) {
        tensor_info.set_dtype(stringToTFDataType(node.outputs[0].dtype));
        auto* shape = tensor_info.mutable_tensor_shape();
        for (int64_t dim : node.outputs[0].shape) {
          shape->add_dim()->set_size(dim);
        }
        break;
      }
    }
    (*output_info)[name] = tensor_info;
  }

  return saved_model;
}

// 序列化输出
static bool saveSavedModel(const tensorflow::SavedModel& saved_model,
                           const std::string& output_path, bool is_text_format) {
  // 自动创建输出目录
  if (!output_path.empty() && output_path != ".") {
    std::error_code ec;
    std::filesystem::create_directories(output_path, ec);
    if (ec) {
      llvm::errs() << "[annc-converter] Error: failed to create output directory: "
                   << output_path << " (" << ec.message() << ")\n";
      return false;
    }
  }

  std::string full_path = output_path;
  if (is_text_format) {
    full_path += "/saved_model.pbtxt";
  } else {
    full_path += "/saved_model.pb";
  }

  std::ofstream output_file(full_path, std::ios::binary);
  if (!output_file.is_open()) {
    llvm::errs() << "[annc-converter] Error: failed to open output file: "
                 << full_path << "\n";
    return false;
  }

  if (is_text_format) {
    // 文本格式
    std::string text_output;
    if (!google::protobuf::TextFormat::PrintToString(saved_model, &text_output)) {
      llvm::errs() << "[annc-converter] Error: failed to serialize to text format\n";
      return false;
    }
    output_file << text_output;
  } else {
    // 二进制格式
    if (!saved_model.SerializeToOstream(&output_file)) {
      llvm::errs() << "[annc-converter] Error: failed to serialize to binary format\n";
      return false;
    }
  }

  output_file.close();
  return true;
}

// ----------------------------------------------------------------------------
// CLI 入口
// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input .mlir/.bin>"),
      llvm::cl::Optional);

  llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output file"),
      llvm::cl::value_desc("filename"), llvm::cl::init("-"));

  llvm::cl::opt<bool> verbose("verbose", llvm::cl::desc("Verbose output"),
                               llvm::cl::init(false));

  llvm::cl::opt<bool> text_format("text_format", llvm::cl::desc("Output in text format (.pbtxt) instead of binary (.pb)"),
                                   llvm::cl::init(false));

  llvm::cl::opt<bool> graphdef_rewrite(
      "tf-graphdef-rewrite",
      llvm::cl::desc("Rewrite an input TensorFlow GraphDef with ANNCFused"));

  llvm::cl::opt<std::string> input_graphdef(
      "input_graphdef", llvm::cl::desc("Input TensorFlow GraphDef path"),
      llvm::cl::value_desc("path"));

  llvm::cl::opt<std::string> output_graphdef(
      "output_graphdef", llvm::cl::desc("Output TensorFlow GraphDef path"),
      llvm::cl::value_desc("path"));

  llvm::cl::opt<std::string> kernel_name(
      "kernel_name", llvm::cl::desc("ANNCFused kernel name"),
      llvm::cl::init(""));

  llvm::cl::opt<std::string> shared_lib_path(
      "shared_lib_path", llvm::cl::desc("ANNCFused shared library path"),
      llvm::cl::init(""));

  llvm::cl::opt<std::string> metadata_json(
      "metadata_json", llvm::cl::desc("ANNCFused metadata JSON path"),
      llvm::cl::init(""));

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "annc-converter: Atir MLIR → TF SavedModel\n");

  if (graphdef_rewrite) {
    if (inputFilename.empty()) {
      llvm::errs() << "[annc-converter] Error: optimized ATIR input is required "
                      "for --tf-graphdef-rewrite\n";
      return 1;
    }
    if (input_graphdef.empty() || output_graphdef.empty()) {
      llvm::errs() << "[annc-converter] Error: --input_graphdef and "
                      "--output_graphdef are required for --tf-graphdef-rewrite\n";
      return 1;
    }
    if (shared_lib_path.empty()) {
      llvm::errs() << "[annc-converter] Error: --shared_lib_path is required\n";
      return 1;
    }
    if (metadata_json.empty()) {
      llvm::errs() << "[annc-converter] Error: --metadata_json is required "
                      "for --tf-graphdef-rewrite\n";
      return 1;
    }

    auto fusions = readFusionInfosJson(metadata_json);
    if (fusions.empty()) {
      llvm::errs() << "[annc-converter] Error: failed to read ANNCFused "
                      "metadata JSON: "
                   << metadata_json << "\n";
      return 1;
    }

    return rewriteGraphDefWithANNCFused(std::move(fusions), input_graphdef,
                                        output_graphdef, kernel_name,
                                        shared_lib_path,
                                        text_format, verbose)
               ? 0
               : 1;
  }

  if (inputFilename.empty()) {
    llvm::errs() << "[annc-converter] Error: input file is required\n";
    return 1;
  }

  // 1. 注册方言
  DialectRegistry registry;
  registry.insert<func::FuncDialect, atir::AtirDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // 2. 加载 MLIR 文件 (支持 .mlir 文本和 .bin bytecode)
  OwningOpRef<ModuleOp> module = parseModuleFromFile(inputFilename, &context);

  if (!module) {
    llvm::errs() << "[annc-converter] Error: failed to parse MLIR file: "
                 << inputFilename << "\n";
    return 1;
  }

  // 3. 解析 MLIR → NodeInfo
  AtirMLIRParser parser(context);
  std::vector<NodeInfo> nodes = parser.parse(*module);

  if (nodes.empty()) {
    llvm::errs() << "[annc-converter] Error: no nodes extracted from MLIR\n";
    return 1;
  }

  // 4. Step 3: 融合节点拆分
  std::vector<NodeInfo> splitNodes = splitFusedNodes(nodes);

  if (verbose) {
    llvm::outs() << "[annc-converter] After Step 3 (fusion split): "
                 << splitNodes.size() << " nodes";
    if (splitNodes.size() != nodes.size()) {
      llvm::outs() << " (was " << nodes.size() << ", split "
                   << (splitNodes.size() - nodes.size()) << " more)";
    }
    llvm::outs() << "\n";

    for (const auto &node : splitNodes) {
      llvm::outs() << "  " << node.op_type << " \"" << node.name << "\"";
      if (node.isInputNode) llvm::outs() << " [INPUT]";
      if (node.isOutputNode) llvm::outs() << " [OUTPUT]";
      if (!node.inputs.empty()) {
        llvm::outs() << " inputs=[";
        for (size_t i = 0; i < node.inputs.size(); i++) {
          if (i > 0) llvm::outs() << ", ";
          llvm::outs() << node.inputs[i];
        }
        llvm::outs() << "]";
      }
      if (!node.outputs.empty()) {
        auto &out = node.outputs[0];
        llvm::outs() << " shape=[";
        for (size_t i = 0; i < out.shape.size(); i++) {
          if (i > 0) llvm::outs() << ",";
          llvm::outs() << out.shape[i];
        }
        llvm::outs() << "] dtype=" << out.dtype;
      }
      if (!node.tf_attrs.empty()) {
        llvm::outs() << " attrs={";
        bool first = true;
        for (auto &[k, v] : node.tf_attrs) {
          if (!first) llvm::outs() << ", ";
          first = false;
          llvm::outs() << k << "=" << v;
        }
        llvm::outs() << "}";
      }
      llvm::outs() << "\n";
    }
  }

  // 5. Step 4: 拓扑排序
  std::vector<NodeInfo> sortedNodes = topologicalSort(splitNodes);
  if (verbose) {
    llvm::outs() << "[annc-converter] After Step 4 (topological sort): "
                 << sortedNodes.size() << " nodes\n";
  }

  // 6. Step 4: 构建SavedModel
  tensorflow::SavedModel saved_model = buildSavedModel(sortedNodes);

  // 7. Step 5: 序列化输出
  std::string output_dir = outputFilename;
  if (output_dir == "-") {
    output_dir = "."; // 默认当前目录
  }

  if (!saveSavedModel(saved_model, output_dir, text_format)) {
    llvm::errs() << "[annc-converter] Error: failed to save SavedModel\n";
    return 1;
  }

  llvm::outs() << "[annc-converter] Success! SavedModel written to: " << output_dir
               << (text_format ? "/saved_model.pbtxt\n" : "/saved_model.pb\n");
  return 0;
}
