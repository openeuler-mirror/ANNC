#include "Builder/Builder.h"
#include "mlir/IR/Verifier.h"

using json = nlohmann::json;
using namespace mlir;
namespace annc {

void MLIRBuilder::jsonConvertor(const json& graph) {
  auto unknownLoc = UnknownLoc::get(module_.getContext());
  mainFunc_ = builder_.create<func::FuncOp>(
      unknownLoc, graph.at("name").template get<std::string>(),
      builder_.getFunctionType({}, {}));
  module_.push_back(mainFunc_);

  auto entryBlock = mainFunc_.addEntryBlock();
  builder_.setInsertionPointToStart(entryBlock);

  // noneValue_ = builder_.create<atir::NoneOp>(
  //     unknownLoc, NoneType::get(builder_.getContext()));
  std::vector<Value> inputs = addGraphInput(graph.at("inputs"));
  for (const auto& node : graph.at("nodes")) {
    addNode(node);
  }
  std::vector<Value> outputs = addGraphOutput(graph.at("outputs"));

  llvm::SmallVector<Type> inputTypes{};
  std::for_each(inputs.begin(), inputs.end(),
                [&](Value val) { inputTypes.push_back(val.getType()); });
  llvm::SmallVector<Type> outputTypes{};
  std::for_each(outputs.begin(), outputs.end(),
                [&](Value val) { outputTypes.push_back(val.getType()); });
  mainFunc_.setType(builder_.getFunctionType(inputTypes, outputTypes));
  builder_.create<func::ReturnOp>(unknownLoc, outputs);
}

std::vector<Value> MLIRBuilder::addGraphInput(const json& inputs) {
  std::vector<Value> inputValues;
  for (int i = 0; i < inputs.size(); i++) {
    const std::string name = inputs[i].at("name").template get<std::string>();
    const std::string dtype = inputs[i].at("dtype").template get<std::string>();
    const std::vector<int64_t> shape =
        inputs[i].at("shape").template get<std::vector<int64_t>>();
    atir::TensorType inType = getTensorType(name, dtype, shape);

    Block* entryBlock = &mainFunc_.getBody().back();
    auto argVal =
        entryBlock->addArgument(inType, getLoc(builder_.getContext(), name));
    inputValues.push_back(argVal);
    tensorValues_[name] = argVal;
  }
  return inputValues;
}

std::vector<Value> MLIRBuilder::addGraphOutput(const json& outputs) {
  std::vector<Value> output_vals;
  for (int i = 0; i < outputs.size(); i++) {
    const std::string out =
        outputs.at(i).at("name").template get<std::string>();
    Value val = tensorValues_[out];
    if (val == nullptr)
      llvm::report_fatal_error(llvm::StringRef("Unknown output value: " + out));
    output_vals.emplace_back(val);
  }
  return output_vals;
}

#define CREATE_CONSTAN(T, PREC)                                          \
  {                                                                      \
    const std::vector<T> data =                                          \
        node.at("data").template get<std::vector<T>>();                  \
    auto elems = DenseElementsAttr::get(                                 \
        RankedTensorType::get(shape, builder_.PREC), ArrayRef<T>(data)); \
    auto type = atir::TensorType::get(shape, builder_.PREC, elems);      \
    auto gOp = builder_.create<atir::ConstantOp>(                        \
        getLoc(builder_.getContext(), name), type,                       \
        builder_.getStringAttr(name), builder_.getStringAttr("public")); \
    tensorValues_[name] = gOp.getResult();                               \
    return tensorValues_[name];                                          \
  }

mlir::Value MLIRBuilder::addConstantNode(const json& node) {
  const std::string name = node.at("name").template get<std::string>();
  const std::string dtype = node.at("dtype").template get<std::string>();
  const std::vector<int64_t> shape =
      node.at("shape").template get<std::vector<int64_t>>();
  atir::TensorType inType = getTensorType(name, dtype, shape);
  int64_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                 std::multiplies<int64_t>());
  if (dtype == "float32")
    CREATE_CONSTAN(float_t, getF32Type())
  else if (dtype == "int64")
    CREATE_CONSTAN(float_t, getI64Type())
  else if (dtype == "int32")
    CREATE_CONSTAN(float_t, getI32Type())
  llvm::report_fatal_error("Unsupported data type");
}

#define CREATE_NODE(T, FUNC) \
  if (type == T) {           \
    FUNC(node);              \
    return;                  \
  }

#define CREATE_COMMON_NODE(T, inNum)                                       \
  if (type == #T) {                                                        \
    if (node.at("inputs").size() < inNum) {                                \
      for (size_t i = node.at("inputs").size(); i <= inNum; i++)           \
        ins.emplace_back(noneValue_);                                      \
    }                                                                      \
    mlir::Operation* op =                                                  \
        builder_                                                           \
            .create<atir::T##Op>(                                          \
                getLoc(ctx, node.at("name").template get<std::string>()),  \
                outs, ins, attrs)                                          \
            .getOperation();                                               \
    for (int i = 0; i < node.at("outputs").size(); i++) {                  \
      std::string outName =                                                \
          node.at("outputs").at(i).at("name").template get<std::string>(); \
      tensorValues_[outName] = op->getResult(i);                           \
    }                                                                      \
    return;                                                                \
  }

void MLIRBuilder::createMatMulOp(const json& node,
  ArrayRef<Type> outs,
  ArrayRef<Value> ins,
  ArrayRef<NamedAttribute> attrs) {
    auto ctx = builder_.getContext();
    Location loc = getLoc(ctx, node.at("name").template get<std::string>());
    Value lhs = ins[0];
    Value rhs = ins[1];

    Type outputTensorType = outs[0];
    Value C = builder_.create<atir::BufferOp>(loc, outputTensorType);

    // === Step 2:  bias () ===
    Value bias = nullptr;
    bool hasBias = (ins.size() >= 3);
    if (hasBias) {
      bias = ins[2];
    }

    bool right_transpose =  false;
    bool left_transpose = false;
    bool output_transpose = false;
    bool do_relu = false;
    float relu_limit = -1.0f;
    std::optional<int64_t>  m_start, n_start, k_start, m_size, n_size, k_size;

    // TODO: attrs
    
    auto matmul = builder_.create<atir::MatMulOp>(
      loc,
      outs[0], lhs, rhs, C, hasBias? bias: Value{},
      builder_.getBoolAttr(hasBias),
      builder_.getBoolAttr(right_transpose),
      builder_.getBoolAttr(left_transpose),
      builder_.getBoolAttr(output_transpose),
      builder_.getBoolAttr(do_relu),
      builder_.getF32FloatAttr(relu_limit),
      m_start ? builder_.getI32IntegerAttr(*m_start) : IntegerAttr(),
      n_start ? builder_.getI32IntegerAttr(*n_start) : IntegerAttr(),
      k_start ? builder_.getI32IntegerAttr(*k_start) : IntegerAttr(),
      m_size ? builder_.getI32IntegerAttr(*m_size) : IntegerAttr(),
      n_size ? builder_.getI32IntegerAttr(*n_size) : IntegerAttr(),
      k_size ? builder_.getI32IntegerAttr(*k_size) : IntegerAttr());
    std::string outName = node.at("outputs")[0].at("name").get<std::string>();
    tensorValues_[outName] = matmul.getResult();
    return ;
}

void MLIRBuilder::createAddOp(
    const json& node,
    ArrayRef<Type> outs,
    ArrayRef<Value> ins,
    ArrayRef<NamedAttribute> attrs) {

  auto loc = getLoc(builder_.getContext(), node.at("name").get<std::string>());

  if (ins.empty()) {
    llvm::report_fatal_error("Add requires at least one input");
  }

  bool do_relu = false;
  float relu_limit = -1.0f;
  FloatAttr scalar = FloatAttr();

  auto add = builder_.create<atir::AddOp>(
      loc,
      outs[0],
      ins, // variadic inputs
      builder_.getBoolAttr(do_relu),
      builder_.getF32FloatAttr(relu_limit),
      scalar
  );

  std::string outName = node.at("outputs")[0].at("name").get<std::string>();
  tensorValues_[outName] = add.getResult();
  return ;
}

void MLIRBuilder::addNode(const json& node) {
  const std::string& type = node.at("type").template get<std::string>();
  CREATE_NODE("Constant", addConstantNode)

  std::vector<Type> outs;
  for (int i = 0; i < node.at("outputs").size(); i++) {
    const json& out = node.at("outputs").at(i);
    const std::string name = out.at("name").template get<std::string>();
    const std::string dtype = out.at("dtype").template get<std::string>();
    const std::vector<int64_t> shape =
        out.at("shape").template get<std::vector<int64_t>>();
    outs.push_back(getTensorType(name, dtype, shape));
  }
  std::vector<Value> ins;
  const json& inputs = node.at("inputs");
  for (int i = 0; i < inputs.size(); i++) {
    const std::string inName = inputs.at(i).template get<std::string>();
    Value val = tensorValues_[inName];
    if (val == nullptr) {
      llvm::report_fatal_error(
          llvm::StringRef("Unknown input tensor: " + inName));
    } else if (val.getDefiningOp() == nullptr) {
      // function argument
      ins.push_back(val);
    } else {
      // node output
      ins.push_back(val);
    }
  }
  std::vector<NamedAttribute> attrs;
  for (int i = 0; i < node.at("attributes").size(); i++) {
    const json& attr = node.at("attributes").at(i);
    const std::string name = attr.at("name").template get<std::string>();
    const std::string dtype = attr.at("dtype").template get<std::string>();
    if (dtype == "int64") {
      const auto value = attr.at("value").template get<std::vector<int64_t>>();
      attrs.push_back(NamedAttribute(builder_.getStringAttr(name),
                                     builder_.getI64ArrayAttr(value)));
    } else if (dtype == "float32") {
      const auto value = attr.at("value").template get<std::vector<float_t>>();
      attrs.push_back(NamedAttribute(builder_.getStringAttr(name),
                                     builder_.getF32ArrayAttr(value)));
    } else if (dtype == "string") {
      const std::string value = attr.at("value").template get<std::string>();
      attrs.push_back(NamedAttribute(builder_.getStringAttr(name),
                                     builder_.getStringAttr(value)));
    } else {
      llvm::report_fatal_error(
          llvm::StringRef("Unsupport attribute type: " + dtype));
    }
  }

  auto ctx = builder_.getContext();
  if (type == "MatMul") {
    createMatMulOp(node, outs, ins, attrs);
    return ;
  } else if (type == "Add") {
    createAddOp(node, outs, ins, attrs);
    return ;
  }

  CREATE_COMMON_NODE(Relu, 1)
  // TODO: support more op types
  llvm::report_fatal_error(llvm::StringRef("Unknown op type: " + type));
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
  if (dtype == "uint8")
    return atir::TensorType::get(tensorShape, builder_.getIntegerType(8),
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
  if (dtype == "float16")
    return atir::TensorType::get(tensorShape, builder_.getF16Type(),
                                 builder_.getStringAttr(name), encoding);
  llvm::report_fatal_error("Unsupported data type");
}
}  // namespace annc