#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
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

using namespace mlir;
using namespace annc;

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

// DenseElementsAttr → raw bytes
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

    // LoadOp → StridedSlice
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

  void parseVariableOp(atir::VariableOp op) {
    NodeInfo node;
    node.name = getLocName(op);
    node.op_type = "VariableV2";

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
    if (op.getWithBias().getValue() && op.getBias()) {
      node.inputs.push_back(lookupValueName(*op.getBias()));
    }

    auto tensorType = dyn_cast<atir::TensorType>(op.getOutput().getType());
    if (tensorType) {
      std::vector<int64_t> shape = getShapeVector(tensorType);
      std::string dtype = atirTypeToDtypeStr(tensorType.getElementType());
      node.addOutput(0, node.name, shape, dtype);
    }

    // TF 属性: transpose_a, transpose_b
    node.tf_attrs["transpose_a"] = op.getLeftTranspose().getValue() ? "true" : "false";
    node.tf_attrs["transpose_b"] = op.getRightTranspose().getValue() ? "true" : "false";

    // 记录融合属性供 Step 3 使用
    if (op.getWithBias().getValue()) node.tf_attrs["_fused_withBias"] = "true";
    if (op.getDoRelu().getValue()) {
      node.tf_attrs["_fused_do_relu"] = "true";
      if (op.getReluLimit().getValue().convertToDouble() > 0.0)
        node.tf_attrs["_fused_relu_limit"] = std::to_string(op.getReluLimit().getValue().convertToDouble());
    }

    // TODO: Step 3 融合拆分 — withBias/do_relu 在此处理
    // 当前先不拆分，直接输出为 MatMul 节点

    registerValueName(op.getResult(), node.name);
    nodes_.push_back(std::move(node));
  }

  void parseAddOp(atir::AddOp op) {
    NodeInfo node;
    node.name = getLocName(op);
    node.op_type = "Add";

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

    // TODO: Step 3 融合拆分 — do_relu 在此处理
    // 记录融合属性供 Step 3 使用
    if (op.getDoRelu().getValue()) {
      node.tf_attrs["_fused_do_relu"] = "true";
      if (op.getReluLimit().getValue().convertToDouble() > 0.0)
        node.tf_attrs["_fused_relu_limit"] = std::to_string(op.getReluLimit().getValue().convertToDouble());
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

    // TF 属性: axis
    node.tf_attrs["axis"] = std::to_string(op.getAxis());

    // TODO: Step 3 融合拆分 — do_relu 在此处理

    registerValueName(op.getResult(), node.name);
    nodes_.push_back(std::move(node));
  }

  void parseLoadOp(atir::LoadOp op) {
    NodeInfo node;
    node.name = getLocName(op);
    node.op_type = "StridedSlice";

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
// CLI 入口
// ============================================================================

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input .mlir/.bin>"),
      llvm::cl::Required);

  llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output file"),
      llvm::cl::value_desc("filename"), llvm::cl::init("-"));

  llvm::cl::opt<bool> verbose("verbose", llvm::cl::desc("Verbose output"),
                               llvm::cl::init(false));

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "annc-converter: Atir MLIR → TF GraphDef\n");

  // 1. 注册方言
  DialectRegistry registry;
  registry.insert<func::FuncDialect, atir::AtirDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // 2. 加载 MLIR 文件 (支持 .mlir 文本和 .bin bytecode)
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(inputFilename, &context);

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

  // 4. 输出解析结果 (Step 4 中将替换为 GraphDef 构建)
  if (verbose) {
    llvm::outs() << "[annc-converter] Parsed " << nodes.size()
                 << " nodes from MLIR\n";

    for (const auto &node : nodes) {
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

  // TODO: Step 3 — 融合节点拆分
  // TODO: Step 4 — NodeInfo → TF GraphDef 构建
  // TODO: Step 5 — 序列化输出

  llvm::outs() << "[annc-converter] Step 1-2 complete: "
               << nodes.size() << " nodes parsed\n";
  return 0;
}
