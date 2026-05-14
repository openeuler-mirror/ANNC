#include "Builder/Builder.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/StringMap.h"
#include <algorithm>
#include <complex>
#include <cstdint>
#include <iostream>
#include <variant>

using namespace mlir;
namespace annc {

namespace {
using NodeHandler = void (MLIRBuilder::*)(const NodeInfo&, llvm::ArrayRef<Type>, llvm::ArrayRef<Value>);

int32_t tfAttrToI32Mask(const NodeInfo::TfAttrValue &v) {
  if (auto *i = std::get_if<int64_t>(&v))
    return static_cast<int32_t>(*i);
  return 0;
}

int32_t lookupTfIntMask(const NodeInfo &node, llvm::StringRef key) {
  auto it = node.attrs.find(key.str());
  if (it == node.attrs.end())
    return 0;
  return tfAttrToI32Mask(it->second);
}

void registerNodeHandlers(llvm::StringMap<NodeHandler>& m) {
  // Op handlers will be registered in subsequent PRs
  auto add = [&m](llvm::ArrayRef<llvm::StringRef> keys, NodeHandler h) {
    for (llvm::StringRef k : keys) m[k] = h;
  };
  add({"Add", "AddV2", "AddN"}, &MLIRBuilder::createAddNode);
  add({"Mul"}, &MLIRBuilder::createMulNode);
  add({"Sub"}, &MLIRBuilder::createSubNode);
  add({"Div", "RealDiv", "Divide"}, &MLIRBuilder::createRealDivNode);
  add({"Less"}, &MLIRBuilder::createLessNode);
  add({"NotEqual"}, &MLIRBuilder::createNotEqualNode);
  add({"Less"}, &MLIRBuilder::createLessNode);
  add({"Greater"}, &MLIRBuilder::createGreaterNode);
  add({"GreaterEqual", "GreaterEqualV2"}, &MLIRBuilder::createGreaterEqualNode);
  add({"Maximum"}, &MLIRBuilder::createMaximumNode);
  add({"Minimum"}, &MLIRBuilder::createMinimumNode);
  add({"Concat"}, &MLIRBuilder::createConcatNode);
  add({"ConcatV2"}, &MLIRBuilder::createConcatV2Node);
  add({"Pack", "Stack"}, &MLIRBuilder::createPackNode);
  add({"Merge"}, &MLIRBuilder::createMergeNode);
  add({"DynamicPartition"}, &MLIRBuilder::createDynamicPartitionNode);
  add({"ParallelDynamicStitch"}, &MLIRBuilder::createParallelDynamicStitchNode);
  add({"Select", "SelectV2", "Where"}, &MLIRBuilder::createWhereNode);
}

const llvm::StringMap<NodeHandler>& getNodeDispatchTable() {
  static llvm::StringMap<NodeHandler> table;
  static bool once = false;
  if (!once) {
    registerNodeHandlers(table);
    once = true;
  }
  return table;
}
}  // namespace

void MLIRBuilder::buildFromNodes(const std::vector<NodeInfo>& nodes) {
  auto unknownLoc = UnknownLoc::get(module_.getContext());
  mainFunc_ = builder_.create<func::FuncOp>(
      unknownLoc, "main", builder_.getFunctionType({}, {}));
  module_.push_back(mainFunc_);

  auto entryBlock = mainFunc_.addEntryBlock();
  builder_.setInsertionPointToStart(entryBlock);

  // 分离输入节点、输出节点和计算节点
  std::vector<NodeInfo> inputNodes;
  std::vector<NodeInfo> outputNodes;
  std::vector<NodeInfo> computeNodes;
  
  for (const auto& node : nodes) {
    if (node.isInputNode) {
      inputNodes.push_back(node);
    } else if (node.isOutputNode) {
      outputNodes.push_back(node);
    } else {
      computeNodes.push_back(node);
    }
  }

  std::vector<Value> inputs = addGraphInputs(inputNodes);
  
  for (const auto& node : computeNodes) {
    addNode(node);
  }
  
  for (const auto& node : outputNodes) {
    addNode(node);
  }
  
  std::vector<Value> outputs = addGraphOutputs(outputNodes);

  llvm::SmallVector<Type> inputTypes{};
  std::for_each(inputs.begin(), inputs.end(),
                [&](Value val) { inputTypes.push_back(val.getType()); });
  llvm::SmallVector<Type> outputTypes{};
  std::for_each(outputs.begin(), outputs.end(),
                [&](Value val) { outputTypes.push_back(val.getType()); });
  mainFunc_.setType(builder_.getFunctionType(inputTypes, outputTypes));
  builder_.create<func::ReturnOp>(unknownLoc, outputs);
}

std::vector<Value> MLIRBuilder::addGraphInputs(const std::vector<NodeInfo>& inputNodes) {
  std::vector<Value> inputValues;
  for (const auto& node : inputNodes) {
    if (node.outputs.empty()) continue;
    const std::string& name = node.name;
    const std::string& dtype = node.outputs[0].dtype;
    const std::vector<int64_t>& shape = node.outputs[0].shape;
    atir::TensorType inType = getTensorType(name, dtype, shape);

    Block* entryBlock = &mainFunc_.getBody().back();
    auto argVal =
        entryBlock->addArgument(inType, getLoc(builder_.getContext(), name));
    inputValues.push_back(argVal);
    tensorValues_[name] = argVal;
  }
  return inputValues;
}

std::vector<Value> MLIRBuilder::addGraphOutputs(const std::vector<NodeInfo>& outputNodes) {
  std::vector<Value> output_vals;
  for (const auto& node : outputNodes) {
    const std::string& out = node.name;
    Value val = tensorValues_[out];
    if (val == nullptr)
      llvm::report_fatal_error(llvm::StringRef("Unknown output value: " + out));
    output_vals.emplace_back(val);
  }
  return output_vals;
}

// Base64解码函数
static std::vector<uint8_t> base64Decode(const std::string& encoded) {
    static const std::string chars = 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    
    std::vector<uint8_t> result;
    int val = 0, valb = -8;
    
    for (char c : encoded) {
        if (c == '=') break;
        size_t pos = chars.find(c);
        if (pos == std::string::npos) continue;
        val = (val << 6) + pos;
        valb += 6;
        if (valb >= 0) {
            result.push_back((val >> valb) & 0xFF);
            valb -= 8;
        }
    }
    return result;
}

mlir::Value MLIRBuilder::addConstantNode(const NodeInfo& node) {
  const std::string& name = node.name;
  if (node.outputs.empty()) {
    llvm::report_fatal_error("Constant node has no output info");
  }
  const std::string& dtype = node.outputs[0].dtype;
  const std::vector<int64_t>& shape = node.outputs[0].shape;
  
  // 解码base64数据（与 TensorProto tensor_content 字节布局一致，小端）
  std::vector<uint8_t> decoded;
  if (!node.raw_data.empty()) {
    decoded = base64Decode(node.raw_data);
  } else { // 避免解码失败，用零值回退
    size_t elemCount = 1;
    for (int64_t d : shape) {
      if (d < 0) continue;
      elemCount *= static_cast<size_t>(d);
    }
    auto fillZeros = [&](size_t elemSize) {
      decoded.assign(elemCount * elemSize, 0);
    };
    if (dtype == "float32" || dtype == "int32") fillZeros(sizeof(int32_t));
    else if (dtype == "float64" || dtype == "int64") fillZeros(sizeof(int64_t));
    else if (dtype == "float16" || dtype == "bfloat16" || dtype == "uint16" ||
             dtype == "int16")
      fillZeros(sizeof(uint16_t));
    else if (dtype == "uint8" || dtype == "int8" || dtype == "bool")
      fillZeros(sizeof(uint8_t));
    else if (dtype == "uint32") fillZeros(sizeof(uint32_t));
    else if (dtype == "uint64") fillZeros(sizeof(uint64_t));
    else if (dtype == "complex64") fillZeros(sizeof(float) * 2);
    else if (dtype == "complex128") fillZeros(sizeof(double) * 2);
  }

  auto emitConstant = [&](DenseElementsAttr elems, Type eltType,
                          Attribute encoding = {}) -> mlir::Value {
    atir::TensorType tensorTy = atir::TensorType::get(
        shape, eltType, builder_.getStringAttr(name), encoding, {}, {}, {}, {},
        {}, {}, elems);
    auto gOp = builder_.create<atir::ConstantOp>(
        getLoc(builder_.getContext(), name), tensorTy,
        builder_.getStringAttr(name), builder_.getStringAttr("public"));
    tensorValues_[name] = gOp.getResult();
    return tensorValues_[name];
  };

  auto elementCountFromShape = [&]() -> size_t {
    if (shape.empty()) return 1;
    size_t n = 1;
    for (int64_t d : shape) {
      if (d < 0) continue;
      n *= static_cast<size_t>(d);
    }
    return n;
  };

  auto requireBytes = [&](size_t elemSize) {
    size_t elemCount = elementCountFromShape();
    if (elemCount == 0 && decoded.empty()) {
      return;
    }
    if (decoded.empty() || decoded.size() % elemSize != 0) {
      llvm::report_fatal_error(llvm::StringRef(
          "Constant raw_data size mismatch for dtype " + dtype));
    }
  };

  DenseElementsAttr elems;

  if (dtype == "float32") {
    requireBytes(sizeof(float));
    const float* data = reinterpret_cast<const float*>(decoded.data());
    size_t numElements = decoded.size() / sizeof(float);
    elems = DenseElementsAttr::get(
        RankedTensorType::get(shape, builder_.getF32Type()),
        ArrayRef<float>(data, numElements));
    return emitConstant(elems, builder_.getF32Type());
  }
  if (dtype == "float64") {
    requireBytes(sizeof(double));
    const double* data = reinterpret_cast<const double*>(decoded.data());
    size_t numElements = decoded.size() / sizeof(double);
    elems = DenseElementsAttr::get(
        RankedTensorType::get(shape, builder_.getF64Type()),
        ArrayRef<double>(data, numElements));
    return emitConstant(elems, builder_.getF64Type());
  }
  if (dtype == "float16") {
    requireBytes(sizeof(uint16_t));
    const uint16_t* data =
        reinterpret_cast<const uint16_t*>(decoded.data());
    size_t numElements = decoded.size() / sizeof(uint16_t);
    elems = DenseElementsAttr::get(
        RankedTensorType::get(shape, builder_.getF16Type()),
        ArrayRef<uint16_t>(data, numElements));
    return emitConstant(elems, builder_.getF16Type());
  }
  if (dtype == "bfloat16") {
    requireBytes(sizeof(uint16_t));
    const uint16_t* data =
        reinterpret_cast<const uint16_t*>(decoded.data());
    size_t numElements = decoded.size() / sizeof(uint16_t);
    auto bf16Type = BFloat16Type::get(builder_.getContext());
    elems = DenseElementsAttr::get(RankedTensorType::get(shape, bf16Type),
                                   ArrayRef<uint16_t>(data, numElements));
    return emitConstant(elems, bf16Type);
  }
  if (dtype == "int64") {
    requireBytes(sizeof(int64_t));
    const int64_t* data = reinterpret_cast<const int64_t*>(decoded.data());
    size_t numElements = decoded.size() / sizeof(int64_t);
    elems = DenseElementsAttr::get(
        RankedTensorType::get(shape, builder_.getI64Type()),
        ArrayRef<int64_t>(data, numElements));
    return emitConstant(elems, builder_.getI64Type());
  }
  if (dtype == "int32") {
    requireBytes(sizeof(int32_t));
    const int32_t* data = reinterpret_cast<const int32_t*>(decoded.data());
    size_t numElements = decoded.size() / sizeof(int32_t);
    elems = DenseElementsAttr::get(
        RankedTensorType::get(shape, builder_.getI32Type()),
        ArrayRef<int32_t>(data, numElements));
    return emitConstant(elems, builder_.getI32Type());
  }
  if (dtype == "int16") {
    requireBytes(sizeof(int16_t));
    const int16_t* data = reinterpret_cast<const int16_t*>(decoded.data());
    size_t numElements = decoded.size() / sizeof(int16_t);
    elems = DenseElementsAttr::get(
        RankedTensorType::get(shape, builder_.getI16Type()),
        ArrayRef<int16_t>(data, numElements));
    return emitConstant(elems, builder_.getI16Type());
  }
  if (dtype == "int8") {
    requireBytes(sizeof(int8_t));
    const int8_t* data = reinterpret_cast<const int8_t*>(decoded.data());
    size_t numElements = decoded.size() / sizeof(int8_t);
    elems = DenseElementsAttr::get(
        RankedTensorType::get(shape, builder_.getI8Type()),
        ArrayRef<int8_t>(data, numElements));
    return emitConstant(elems, builder_.getI8Type());
  }
  if (dtype == "uint8") {
    requireBytes(sizeof(uint8_t));
    const uint8_t* data = reinterpret_cast<const uint8_t*>(decoded.data());
    size_t numElements = decoded.size() / sizeof(uint8_t);
    elems = DenseElementsAttr::get(
        RankedTensorType::get(shape, builder_.getIntegerType(8)),
        ArrayRef<uint8_t>(data, numElements));
    return emitConstant(elems, builder_.getIntegerType(8));
  }
  if (dtype == "uint16") {
    requireBytes(sizeof(uint16_t));
    const uint16_t* data =
        reinterpret_cast<const uint16_t*>(decoded.data());
    size_t numElements = decoded.size() / sizeof(uint16_t);
    elems = DenseElementsAttr::get(
        RankedTensorType::get(shape, builder_.getIntegerType(16)),
        ArrayRef<uint16_t>(data, numElements));
    return emitConstant(elems, builder_.getIntegerType(16));
  }
  if (dtype == "uint32") {
    requireBytes(sizeof(uint32_t));
    const uint32_t* data =
        reinterpret_cast<const uint32_t*>(decoded.data());
    size_t numElements = decoded.size() / sizeof(uint32_t);
    elems = DenseElementsAttr::get(
        RankedTensorType::get(shape, builder_.getIntegerType(32)),
        ArrayRef<uint32_t>(data, numElements));
    return emitConstant(elems, builder_.getIntegerType(32));
  }
  if (dtype == "uint64") {
    requireBytes(sizeof(uint64_t));
    const uint64_t* data =
        reinterpret_cast<const uint64_t*>(decoded.data());
    size_t numElements = decoded.size() / sizeof(uint64_t);
    elems = DenseElementsAttr::get(
        RankedTensorType::get(shape, builder_.getIntegerType(64)),
        ArrayRef<uint64_t>(data, numElements));
    return emitConstant(elems, builder_.getIntegerType(64));
  }
  if (dtype == "bool") {
    requireBytes(sizeof(uint8_t));
    std::vector<int32_t> asI32;
    asI32.reserve(decoded.size());
    for (uint8_t b : decoded)
      asI32.push_back(b ? 1 : 0);
    elems = DenseElementsAttr::get(
        RankedTensorType::get(shape, builder_.getI32Type()),
        ArrayRef<int32_t>(asI32));
    return emitConstant(elems, builder_.getI32Type(),
                        builder_.getStringAttr("bool"));
  }
  if (dtype == "complex64") {
    requireBytes(2 * sizeof(float));
    size_t numElements = decoded.size() / (2 * sizeof(float));
    const float* p = reinterpret_cast<const float*>(decoded.data());
    std::vector<std::complex<float>> vals(numElements);
    for (size_t i = 0; i < numElements; ++i)
      vals[i] = std::complex<float>(p[2 * i], p[2 * i + 1]);
    elems = DenseElementsAttr::get(
        RankedTensorType::get(shape, ComplexType::get(builder_.getF32Type())),
        ArrayRef<std::complex<float>>(vals));
    return emitConstant(elems, ComplexType::get(builder_.getF32Type()));
  }
  if (dtype == "complex128") {
    requireBytes(2 * sizeof(double));
    size_t numElements = decoded.size() / (2 * sizeof(double));
    const double* p = reinterpret_cast<const double*>(decoded.data());
    std::vector<std::complex<double>> vals(numElements);
    for (size_t i = 0; i < numElements; ++i)
      vals[i] = std::complex<double>(p[2 * i], p[2 * i + 1]);
    elems = DenseElementsAttr::get(
        RankedTensorType::get(shape, ComplexType::get(builder_.getF64Type())),
        ArrayRef<std::complex<double>>(vals));
    return emitConstant(elems, ComplexType::get(builder_.getF64Type()));
  }
  if (dtype == "string") {
    llvm::report_fatal_error(
        "Constant tf.string from raw_data is not supported in MLIROpBuilder");
  }

  llvm::report_fatal_error(llvm::StringRef("Unsupported data type for constant: " +
                                           dtype));
}

#define CREATE_NODE(T, FUNC) \
  if (type == T) {           \
    FUNC(node);              \
    return;                  \
  }
void MLIRBuilder::createUnsupportedNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins) {
  (void)outs; (void)ins;
  llvm::report_fatal_error(llvm::StringRef("Op type not fully supported yet: " + node.op_type));
}
void MLIRBuilder::createCustomizeNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins) {
  auto loc = getLoc(builder_.getContext(), node.name);
  llvm::SmallVector<Type, 2> outTypes(outs.begin(), outs.end());
  auto op = builder_.create<atir::CustomizeOp>(loc, outTypes, ins, builder_.getStringAttr(node.op_type));
  for (size_t i = 0; i < node.outputs.size(); ++i)
    tensorValues_[node.outputs[i].name] = op.getResult(static_cast<unsigned>(i));
}

#undef SINGLE_OUT

void MLIRBuilder::addNode(const NodeInfo& node) {
  const std::string& type = node.op_type;
  CREATE_NODE("Constant", addConstantNode)
  CREATE_NODE("Const", addConstantNode)
  if (node.isOutputNode && type == "Identity" && node.inputs.size() == 1) {
    Value src = tensorValues_[node.inputs[0]];
    if (src == nullptr) {
      llvm::report_fatal_error(llvm::StringRef("Unknown input tensor: " + node.inputs[0] +
                                               " when building node " + node.name +
                                               " (" + node.op_type + ")"));
    }
    tensorValues_[node.name] = src;
    if (!node.outputs.empty()) {
      tensorValues_[node.outputs[0].name] = src;
    }
    return;
  }

  std::vector<Type> outs;
  for (const auto& out : node.outputs)
    outs.push_back(getTensorType(out.name, out.dtype, out.shape));

  std::vector<Value> ins;
  for (const auto& inName : node.inputs) {
    Value val = tensorValues_[inName];
    if (val == nullptr)
      llvm::report_fatal_error(llvm::StringRef("Unknown input tensor: " + inName +
                                               " when building node " + node.name +
                                               " (" + node.op_type + ")"));
    ins.push_back(val);
  }

  const llvm::StringMap<NodeHandler>& table = getNodeDispatchTable();
  auto it = table.find(type);
  auto applyTfAttrsToCreatedOp = [&](Operation *createdOp) {
    if (!createdOp || node.attrs.empty())
      return;
    auto ctx = builder_.getContext();
    auto mapTfAttrName = [&](StringRef tfName) -> StringRef {
      if (tfName == "transpose_a") return "left_transpose";
      if (tfName == "transpose_b") return "right_transpose";
      if (tfName == "adj_x") return "transposeA";
      if (tfName == "adj_y") return "transposeB";
      if (tfName == "keep_dims") return "keep_dims";
      return tfName;
    };

    for (const auto &kv : node.attrs) {
      const std::string &tfAttrName = kv.first;
      const auto &tfVal = kv.second;

      StringRef atirAttrName = mapTfAttrName(tfAttrName);
      if (!createdOp->hasAttr(atirAttrName))
        continue;

      Attribute existingAttr = createdOp->getAttr(atirAttrName);
      Attribute newAttr;

      if (auto iAttr = dyn_cast<IntegerAttr>(existingAttr)) {
        if (auto v = std::get_if<int64_t>(&tfVal)) {
          newAttr = IntegerAttr::get(iAttr.getType(), *v);
        } else if (auto v = std::get_if<bool>(&tfVal)) {
          newAttr = IntegerAttr::get(iAttr.getType(), *v ? 1 : 0);
        } else if (auto v = std::get_if<double>(&tfVal)) {
          newAttr = IntegerAttr::get(iAttr.getType(),
                                      static_cast<int64_t>(*v));
        }
      } else if (auto fAttr = dyn_cast<FloatAttr>(existingAttr)) {
        double d;
        if (auto v = std::get_if<double>(&tfVal)) {
          d = *v;
        } else if (auto v = std::get_if<int64_t>(&tfVal)) {
          d = static_cast<double>(*v);
        } else {
          d = 0.0;
          continue;
        }
        newAttr = FloatAttr::get(fAttr.getType(), d);
      } else if (auto bAttr = dyn_cast<BoolAttr>(existingAttr)) {
        if (auto v = std::get_if<bool>(&tfVal)) {
          newAttr = BoolAttr::get(ctx, *v);
        } else if (auto v = std::get_if<int64_t>(&tfVal)) {
          newAttr = BoolAttr::get(ctx, *v != 0);
        } else {
          continue;
        }
      } else if (auto sAttr = dyn_cast<StringAttr>(existingAttr)) {
        if (auto v = std::get_if<std::string>(&tfVal)) {
          newAttr = StringAttr::get(ctx, *v);
        } else if (auto v = std::get_if<int64_t>(&tfVal)) {
          newAttr = StringAttr::get(ctx, std::to_string(*v));
        } else {
          continue;
        }
      } else if (auto denseI64 = dyn_cast<DenseI64ArrayAttr>(existingAttr)) {
        (void)denseI64;
        if (auto v = std::get_if<std::vector<int64_t>>(&tfVal)) {
          newAttr = DenseI64ArrayAttr::get(ctx, *v);
        }
      } else if (auto arrAttr = dyn_cast<ArrayAttr>(existingAttr)) {
        if (!arrAttr || arrAttr.size() == 0)
          continue;
        Attribute first = arrAttr[0];

        if (dyn_cast<IntegerAttr>(first)) {
          if (auto v = std::get_if<std::vector<int64_t>>(&tfVal)) {
            auto ty = dyn_cast<IntegerAttr>(first).getType();
            llvm::SmallVector<Attribute> elems;
            elems.reserve(v->size());
            for (auto x : *v)
              elems.push_back(IntegerAttr::get(ty, x));
            newAttr = ArrayAttr::get(ctx, elems);
          }
        } else if (dyn_cast<FloatAttr>(first)) {
          if (auto v = std::get_if<std::vector<double>>(&tfVal)) {
            auto ty = dyn_cast<FloatAttr>(first).getType();
            llvm::SmallVector<Attribute> elems;
            elems.reserve(v->size());
            for (auto x : *v)
              elems.push_back(FloatAttr::get(ty, x));
            newAttr = ArrayAttr::get(ctx, elems);
          }
        } else if (dyn_cast<BoolAttr>(first)) {
          if (auto v = std::get_if<std::vector<bool>>(&tfVal)) {
            llvm::SmallVector<Attribute> elems;
            elems.reserve(v->size());
            for (auto x : *v)
              elems.push_back(BoolAttr::get(ctx, x));
            newAttr = ArrayAttr::get(ctx, elems);
          }
        } else if (dyn_cast<StringAttr>(first)) {
          if (auto v = std::get_if<std::vector<std::string>>(&tfVal)) {
            llvm::SmallVector<Attribute> elems;
            elems.reserve(v->size());
            for (auto x : *v)
              elems.push_back(StringAttr::get(ctx, x));
            newAttr = ArrayAttr::get(ctx, elems);
          }
        }
      }

      if (newAttr)
        createdOp->setAttr(atirAttrName, newAttr);
    }
  };

  Operation *createdOp = nullptr;
  if (it != table.end()) {
    (this->*it->second)(node, outs, ins);
  } else {
    createCustomizeNode(node, outs, ins);
  }
  for (const auto &out : node.outputs) {
    auto itVal = tensorValues_.find(out.name);
    if (itVal == tensorValues_.end())
      continue;
    if (!itVal->second)
      continue;
    if (auto op = itVal->second.getDefiningOp()) {
      createdOp = op;
      break;
    }
  }
  applyTfAttrsToCreatedOp(createdOp);
}

atir::TensorType MLIRBuilder::getTensorType(const std::string& name,
                                            const std::string& dtype,
                                            const std::vector<int64_t>& shape) {
  std::vector<int64_t> tensorShape = shape;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == -1) tensorShape[i] = ShapedType::kDynamic;
  }
  mlir::Attribute encoding;
  if (dtype == "float32")
    return atir::TensorType::get(tensorShape, builder_.getF32Type(),
                                 builder_.getStringAttr(name), encoding);
  if (dtype == "float64")
    return atir::TensorType::get(tensorShape, builder_.getF64Type(),
                                 builder_.getStringAttr(name), encoding);
  if (dtype == "float16")
    return atir::TensorType::get(tensorShape, builder_.getF16Type(),
                                 builder_.getStringAttr(name), encoding);
  if (dtype == "bfloat16")
    return atir::TensorType::get(tensorShape,
                                 BFloat16Type::get(builder_.getContext()),
                                 builder_.getStringAttr(name), encoding);
  if (dtype == "uint8")
    return atir::TensorType::get(tensorShape, builder_.getIntegerType(8),
                                 builder_.getStringAttr(name), encoding);
  if (dtype == "uint16")
    return atir::TensorType::get(tensorShape, builder_.getIntegerType(16),
                                 builder_.getStringAttr(name), encoding);
  if (dtype == "uint32")
    return atir::TensorType::get(tensorShape, builder_.getIntegerType(32),
                                 builder_.getStringAttr(name), encoding);
  if (dtype == "uint64")
    return atir::TensorType::get(tensorShape, builder_.getIntegerType(64),
                                 builder_.getStringAttr(name), encoding);
  if (dtype == "int8")
    return atir::TensorType::get(tensorShape, builder_.getI8Type(),
                                 builder_.getStringAttr(name), encoding);
  if (dtype == "int16")
    return atir::TensorType::get(tensorShape, builder_.getI16Type(),
                                 builder_.getStringAttr(name), encoding);
  if (dtype == "int32")
    return atir::TensorType::get(tensorShape, builder_.getI32Type(),
                                 builder_.getStringAttr(name), encoding);
  if (dtype == "int64")
    return atir::TensorType::get(tensorShape, builder_.getI64Type(),
                                 builder_.getStringAttr(name), encoding);
  // todo 类型处理
  if (dtype == "bool") {
    encoding = builder_.getStringAttr("bool");
    return atir::TensorType::get(tensorShape, builder_.getI32Type(),
                                 builder_.getStringAttr(name), encoding);
  }
  if (dtype == "string") {
    return atir::TensorType::get(
        tensorShape,
        ComplexType::get(builder_.getF32Type()),
        builder_.getStringAttr(name), builder_.getStringAttr("string"));
  }
  if (dtype == "complex64") {
    return atir::TensorType::get(
        tensorShape, ComplexType::get(builder_.getF32Type()),
        builder_.getStringAttr(name), encoding);
  }
  if (dtype == "complex128") {
    return atir::TensorType::get(
        tensorShape, ComplexType::get(builder_.getF64Type()),
        builder_.getStringAttr(name), encoding);
  }
  llvm::report_fatal_error("Unsupported data type");
}

void MLIRBuilder::createAddNode(
    const NodeInfo& node,
    ArrayRef<Type> outs,
    ArrayRef<Value> ins) {

  auto loc = getLoc(builder_.getContext(), node.name);

  if (ins.empty()) {
    llvm::report_fatal_error("Add requires at least one input");
  }

  bool do_relu = false;
  float relu_limit = -1.0f;
  FloatAttr scalar = FloatAttr();

  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = builder_.create<atir::BufferOp>(loc, outputType);

  auto add = builder_.create<atir::AddOp>(
      loc,
      outs[0],
      outputBuffer.getResult(),
      ins,
      builder_.getBoolAttr(do_relu),
      builder_.getF32FloatAttr(relu_limit),
      scalar
  );

  std::string outName = node.outputs[0].name;
  tensorValues_[outName] = add.getResult();
}

#define SINGLE_OUT(OP_CREATE) do { \
  auto loc = getLoc(builder_.getContext(), node.name); \
  auto op = (OP_CREATE); \
  tensorValues_[node.outputs[0].name] = op.getResult(); \
} while (0)

void MLIRBuilder::createMulNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins) {
  auto loc = getLoc(builder_.getContext(), node.name);
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = builder_.create<atir::BufferOp>(loc, outputType);
  SINGLE_OUT(builder_.create<atir::MulOp>(loc, outs[0], outputBuffer.getResult(), ins[0], ins[1]));
}
void MLIRBuilder::createSubNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins) {
  auto loc = getLoc(builder_.getContext(), node.name);
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = builder_.create<atir::BufferOp>(loc, outputType);
  SINGLE_OUT(builder_.create<atir::SubOp>(loc, outs[0], outputBuffer.getResult(), ins[0], ins[1]));
}
void MLIRBuilder::createRealDivNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins) {
  auto loc = getLoc(builder_.getContext(), node.name);
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = builder_.create<atir::BufferOp>(loc, outputType);
  SINGLE_OUT(builder_.create<atir::RealDivOp>(loc, outs[0], outputBuffer.getResult(), ins[0], ins[1]));
}
void MLIRBuilder::createNotEqualNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins) {
  SINGLE_OUT(builder_.create<atir::CompareOp>(loc, outs[0], ins[0], ins[1],
                                              builder_.getStringAttr("NE")));
}
void MLIRBuilder::createLessNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins) {
  SINGLE_OUT(builder_.create<atir::CompareOp>(loc, outs[0], ins[0], ins[1],
                                              builder_.getStringAttr("LT")));
}
void MLIRBuilder::createGreaterNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins) {
  SINGLE_OUT(builder_.create<atir::CompareOp>(loc, outs[0], ins[0], ins[1],
                                              builder_.getStringAttr("GT")));
}
void MLIRBuilder::createGreaterEqualNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins) {
  SINGLE_OUT(builder_.create<atir::CompareOp>(loc, outs[0], ins[0], ins[1],
                                              builder_.getStringAttr("GE")));
}
void MLIRBuilder::createMaximumNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins) {
  auto loc = getLoc(builder_.getContext(), node.name);
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = builder_.create<atir::BufferOp>(loc, outputType);
  SINGLE_OUT(builder_.create<atir::MaximumOp>(loc, outs[0], outputBuffer.getResult(), ins[0], ins[1]));
}
void MLIRBuilder::createMinimumNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins) {
  auto loc = getLoc(builder_.getContext(), node.name);
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = builder_.create<atir::BufferOp>(loc, outputType);
  SINGLE_OUT(builder_.create<atir::MinimumOp>(loc, outs[0], outputBuffer.getResult(), ins[0], ins[1]));
}
void MLIRBuilder::createConcatNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins) {
  auto loc = getLoc(builder_.getContext(), node.name);
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = builder_.create<atir::BufferOp>(loc, outputType);
  auto op = builder_.create<atir::ConcatOp>(loc, outs[0], outputBuffer.getResult(), ins, builder_.getI32IntegerAttr(0),
      builder_.getBoolAttr(false), builder_.getF32FloatAttr(-1.0f), builder_.getI32IntegerAttr(0), BoolAttr());
  tensorValues_[node.outputs[0].name] = op.getResult();
}
void MLIRBuilder::createConcatV2Node(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins) {
  auto loc = getLoc(builder_.getContext(), node.name);
  // ConcatV2: 最后一个输入是 axis，前面的是 values
  if (ins.size() < 2) {
    llvm::report_fatal_error("ConcatV2 requires at least two inputs (values and axis)");
  }
  // 分离 values 和 axis
  llvm::SmallVector<Value> values(ins.begin(), ins.end() - 1);
  Value axis = ins.back();
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = builder_.create<atir::BufferOp>(loc, outputType);
  auto op = builder_.create<atir::ConcatV2Op>(loc, outs[0], outputBuffer.getResult(), values, axis);
  tensorValues_[node.outputs[0].name] = op.getResult();
}
void MLIRBuilder::createPackNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins) {
  auto loc = getLoc(builder_.getContext(), node.name);
  if (ins.empty()) { createUnsupportedNode(node, outs, ins); return; }
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = builder_.create<atir::BufferOp>(loc, outputType);
  auto op = builder_.create<atir::PackOp>(loc, outs[0], outputBuffer.getResult(), ins, builder_.getI64IntegerAttr(0));
  tensorValues_[node.outputs[0].name] = op.getResult();
}
void MLIRBuilder::createMergeNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins) {
  if (ins.empty() || outs.size() < 2) { createUnsupportedNode(node, outs, ins); return; }
  auto loc = getLoc(builder_.getContext(), node.name);
  auto mergeOp = builder_.create<atir::MergeOp>(loc, TypeRange{outs[0], outs[1]}, ins);
  if (node.outputs.size() >= 2) {
    tensorValues_[node.outputs[0].name] = mergeOp.getOutput();
    tensorValues_[node.outputs[1].name] = mergeOp.getValueIndex();
  } else if (node.outputs.size() == 1) {
    tensorValues_[node.outputs[0].name] = mergeOp.getOutput();
  }
}
void MLIRBuilder::createDynamicPartitionNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins) {
  if (ins.size() < 2) { createUnsupportedNode(node, outs, ins); return; }
  auto loc = getLoc(builder_.getContext(), node.name);
  Value data = ins[0];
  Value partitions = ins[1];
  int32_t numPartitions = static_cast<int32_t>(outs.size());
  auto partitionOp = builder_.create<atir::DynamicPartitionOp>(loc, outs, data, partitions, 
                                                                builder_.getI32IntegerAttr(numPartitions));
  for (size_t i = 0; i < node.outputs.size() && i < outs.size(); ++i) {
    Value outVal = partitionOp.getOutputs()[i];
    tensorValues_[node.outputs[i].name] = outVal;
    if (i == 0) {
      tensorValues_[node.name] = outVal;
    } else {
      tensorValues_[node.name + ":" + std::to_string(i)] = outVal;
    }
  }
}
void MLIRBuilder::createParallelDynamicStitchNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins) {
  if (ins.size() < 2) { createUnsupportedNode(node, outs, ins); return; }
  auto loc = getLoc(builder_.getContext(), node.name);
  size_t half = ins.size() / 2;
  llvm::SmallVector<Value, 4> indices(ins.begin(), ins.begin() + half);
  llvm::SmallVector<Value, 4> data(ins.begin() + half, ins.end());
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = builder_.create<atir::BufferOp>(loc, outputType);
  auto stitchOp = builder_.create<atir::ParallelDynamicStitchOp>(loc, outs[0], outputBuffer.getResult(), indices, data);
  tensorValues_[node.outputs[0].name] = stitchOp.getResult();
}
void MLIRBuilder::createWhereNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins) {
  if (ins.size() != 1 && ins.size() != 3) {
    createUnsupportedNode(node, outs, ins);
    return;
  }
  auto loc = getLoc(builder_.getContext(), node.name);
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = builder_.create<atir::BufferOp>(loc, outputType);
  SINGLE_OUT(builder_.create<atir::WhereOp>(loc, outs[0], outputBuffer.getResult(), ins));
}

}  // namespace annc