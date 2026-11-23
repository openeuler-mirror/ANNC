#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "src/Dialect/ONNX/ONNXTypes.hpp"

using namespace mlir;

namespace {

class AnncFrameworkDialect : public Dialect {
 public:
  explicit AnncFrameworkDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<AnncFrameworkDialect>()) {
    allowUnknownOperations();
  }
  static StringRef getDialectNamespace() { return "annc_framework"; }
};

bool isString(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  return shaped && isa<ONNXStringType>(shaped.getElementType());
}

bool containsString(Type type) {
  bool found = false;
  type.walk([&](Type nested) { found |= isa<ONNXStringType>(nested); });
  return found;
}

Type carrierType(Type type) {
  if (auto ranked = dyn_cast<RankedTensorType>(type); ranked && isString(type))
    return RankedTensorType::get(ranked.getShape(),
                                 IntegerType::get(type.getContext(), 64));
  return type;
}

std::string hexBytes(StringRef value) {
  const char *digits = "0123456789abcdef";
  std::string result;
  result.reserve(value.size() * 2);
  for (unsigned char byte : value.bytes()) {
    result.push_back(digits[byte >> 4]);
    result.push_back(digits[byte & 15]);
  }
  return result;
}

class OnnxStringToFrameworkCustomCallPass
    : public PassWrapper<OnnxStringToFrameworkCustomCallPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      OnnxStringToFrameworkCustomCallPass)
  StringRef getArgument() const final {
    return "onnx-string-to-framework-custom-call";
  }
  StringRef getDescription() const final {
    return "Replace ONNX string Constant/Expand with framework custom calls";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AnncFrameworkDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> strings;
    bool unsupported = false;
    module.walk([&](Operation *op) {
      auto check = [&](Type type) {
        if (containsString(type) &&
            (!isa<RankedTensorType>(type) || !isString(type))) {
          op->emitError("framework string boundary requires ranked string "
                        "tensors; unranked or nested strings are unsupported");
          unsupported = true;
        }
      };
      for (Type type : op->getOperandTypes()) check(type);
      for (Type type : op->getResultTypes()) check(type);
      if (auto function = dyn_cast<func::FuncOp>(op)) {
        for (Type type : function.getArgumentTypes()) check(type);
        for (Type type : function.getResultTypes()) check(type);
      }
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument arg : block.getArguments()) check(arg.getType());
      bool usesString = llvm::any_of(op->getResultTypes(), isString) ||
                        llvm::any_of(op->getOperandTypes(), isString);
      if (!usesString || isa<func::ReturnOp>(op)) return;
      StringRef name = op->getName().getStringRef();
      if (name != "onnx.Constant" && name != "onnx.Expand") {
        op->emitError("unsupported string operation at framework boundary");
        unsupported = true;
      } else {
        strings.push_back(op);
      }
    });
    if (unsupported) {
      signalPassFailure();
      return;
    }

    IRRewriter builder(&getContext());
    for (Operation *op : strings) {
      bool constant = op->getName().getStringRef() == "onnx.Constant";
      auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!resultType || !isString(resultType)) {
        op->emitError("string custom call requires a ranked string result");
        signalPassFailure();
        return;
      }
      llvm::json::Object config{
          {"schema", "annc.onnx.string/1"},
          {"source_op", op->getName().getStringRef().str()},
          {"element_type", "string"},
          {"carrier", "opaque-i64-handles"}};
      if (auto node = op->getAttrOfType<StringAttr>("onnx_node_name"))
        config["source_node"] = node.getValue().str();
      llvm::json::Array shape;
      for (int64_t extent : resultType.getShape())
        shape.push_back(extent == ShapedType::kDynamic ? -1 : extent);
      config["result_shape"] = std::move(shape);
      if (constant) {
        llvm::json::Array values;
        if (auto dense = op->getAttrOfType<DenseStringElementsAttr>("value")) {
          for (StringRef value : dense.getValues<StringRef>())
            values.push_back(hexBytes(value));
        } else if (auto value = op->getAttrOfType<StringAttr>("value_string")) {
          values.push_back(hexBytes(value.getValue()));
        } else {
          op->emitError("string Constant has no losslessly representable data");
          signalPassFailure();
          return;
        }
        if (!resultType.hasStaticShape() ||
            static_cast<int64_t>(values.size()) != resultType.getNumElements()) {
          op->emitError("string Constant data count does not match its shape");
          signalPassFailure();
          return;
        }
        config["string_data_hex"] = std::move(values);
      }

      builder.setInsertionPoint(op);
      OperationState state(op->getLoc(), "annc_framework.custom_call");
      state.addOperands(op->getOperands());
      state.addTypes(carrierType(resultType));
      state.addAttribute("call_target_name",
                         builder.getStringAttr(constant
                             ? "annc.onnx.string.constant"
                             : "annc.onnx.string.expand"));
      state.addAttribute("backend_config", builder.getStringAttr(
          llvm::formatv("{0}", llvm::json::Value(std::move(config))).str()));
      state.addAttribute("has_side_effect", builder.getBoolAttr(false));
      if (auto ids = op->getAttr("source_ids"))
        state.addAttribute("source_ids", ids);
      if (auto node = op->getAttr("onnx_node_name"))
        state.addAttribute("onnx_node_name", node);
      builder.replaceOp(op, builder.create(state)->getResults());
    }
    module.walk([&](func::FuncOp function) {
      auto oldType = function.getFunctionType();
      SmallVector<Type> inputs, results;
      for (auto [index, type] : llvm::enumerate(oldType.getInputs())) {
        inputs.push_back(carrierType(type));
        if (isString(type))
          function.setArgAttr(index, "annc.original_element_type",
                              builder.getStringAttr("string"));
      }
      for (auto [index, type] : llvm::enumerate(oldType.getResults())) {
        results.push_back(carrierType(type));
        if (isString(type))
          function.setResultAttr(index, "annc.original_element_type",
                                 builder.getStringAttr("string"));
      }
      function.setFunctionType(FunctionType::get(&getContext(), inputs, results));
      if (!function.isExternal())
        for (BlockArgument arg : function.front().getArguments())
          arg.setType(carrierType(arg.getType()));
    });
  }
};

}  // namespace

void registerOnnxStringToFrameworkCustomCallPass() {
  PassRegistration<OnnxStringToFrameworkCustomCallPass>();
}

void registerAnncFrameworkDialect(DialectRegistry &registry) {
  registry.insert<AnncFrameworkDialect>();
}
