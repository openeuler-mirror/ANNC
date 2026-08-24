#include <exception>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "Dialect/Atir/AtirTypes.h.inc"
#include "Dialect/Atir/Interfaces/Interfaces.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "nlohmann/json.hpp"

using namespace llvm;
using namespace mlir;

namespace atir {
namespace {

TensorType cloneWithShape(TensorType type, ArrayRef<int64_t> shape) {
  return TensorType::get(
      shape, type.getElementType(), type.getName(), type.getEncoding(),
      type.getStride(), type.getLayout(), type.getMemType(), type.getAddress(),
      type.getDeviceParallel(), type.getOnchipParallel(), type.getCacheData());
}

bool hasDynamicShape(Type type) {
  auto tensorType = dyn_cast<TensorType>(type);
  return tensorType &&
         llvm::any_of(tensorType.getShape(), ShapedType::isDynamic);
}

void specializeDestinationBuffers(func::FuncOp function) {
  function.walk([](BufferOp buffer) {
    auto bufferType = dyn_cast<TensorType>(buffer.getOutput().getType());
    if (!bufferType || !hasDynamicShape(bufferType) ||
        !buffer.getOutput().hasOneUse()) {
      return;
    }

    Operation *user = *buffer.getOutput().getUsers().begin();
    if (user->getNumOperands() == 0 ||
        user->getOperand(0) != buffer.getOutput() ||
        user->getNumResults() != 1) {
      return;
    }
    auto resultType = dyn_cast<TensorType>(user->getResult(0).getType());
    if (!resultType || hasDynamicShape(resultType)) return;
    buffer.getOutput().setType(
        cloneWithShape(bufferType, resultType.getShape()));
  });
}

struct RuntimeArgumentShape {
  unsigned index;
  std::vector<int64_t> shape;
};

FailureOr<std::vector<RuntimeArgumentShape>> readShapeSpec(
    ModuleOp module, StringRef path, std::string &kernelName) {
  std::ifstream input(path.str());
  if (!input.is_open()) {
    module.emitError() << "cannot open runtime shape spec '" << path << "'";
    return failure();
  }

  nlohmann::json json;
  try {
    input >> json;
    kernelName = json.at("kernel_name").get<std::string>();
    std::vector<RuntimeArgumentShape> arguments;
    for (const auto &argument : json.at("arguments")) {
      int64_t index = argument.at("index").get<int64_t>();
      if (index < 0) {
        module.emitError("runtime shape argument index must be non-negative");
        return failure();
      }
      arguments.push_back(RuntimeArgumentShape{
          static_cast<unsigned>(index),
          argument.at("shape").get<std::vector<int64_t>>()});
    }
    return arguments;
  } catch (const std::exception &error) {
    module.emitError() << "invalid runtime shape spec '" << path
                       << "': " << error.what();
    return failure();
  }
}

class AtirSpecializeShapesPass
    : public AtirSpecializeShapesBase<AtirSpecializeShapesPass> {
 public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (shapeSpec.empty()) {
      module.emitError("atir-specialize-shapes requires shape-spec");
      signalPassFailure();
      return;
    }

    std::string kernelName;
    FailureOr<std::vector<RuntimeArgumentShape>> runtimeShapes =
        readShapeSpec(module, shapeSpec, kernelName);
    if (failed(runtimeShapes)) {
      signalPassFailure();
      return;
    }

    func::FuncOp function = module.lookupSymbol<func::FuncOp>(kernelName);
    if (!function || !function->hasAttr("annc.kernel")) {
      module.emitError() << "shape spec references unknown ANNC kernel '"
                         << kernelName << "'";
      signalPassFailure();
      return;
    }
    if (runtimeShapes->size() != function.getNumArguments()) {
      function.emitError() << "runtime shape spec has " << runtimeShapes->size()
                           << " arguments, expected "
                           << function.getNumArguments();
      signalPassFailure();
      return;
    }

    std::unordered_set<unsigned> seen;
    SmallVector<Type> inputTypes(function.getArgumentTypes());
    for (const RuntimeArgumentShape &runtimeShape : *runtimeShapes) {
      if (runtimeShape.index >= function.getNumArguments() ||
          !seen.insert(runtimeShape.index).second) {
        function.emitError() << "invalid or duplicate runtime argument index "
                             << runtimeShape.index;
        signalPassFailure();
        return;
      }

      auto tensorType = dyn_cast<TensorType>(
          function.getArgument(runtimeShape.index).getType());
      if (!tensorType) {
        function.emitError()
            << "argument " << runtimeShape.index << " is not an ATIR tensor";
        signalPassFailure();
        return;
      }
      if (tensorType.getShape().size() != runtimeShape.shape.size()) {
        function.emitError()
            << "argument " << runtimeShape.index << " rank mismatch";
        signalPassFailure();
        return;
      }
      for (auto [declared, actual] :
           llvm::zip_equal(tensorType.getShape(), runtimeShape.shape)) {
        if (actual <= 0 ||
            (!ShapedType::isDynamic(declared) && declared != actual)) {
          function.emitError() << "argument " << runtimeShape.index
                               << " has incompatible runtime shape";
          signalPassFailure();
          return;
        }
      }

      Type specialized = cloneWithShape(tensorType, runtimeShape.shape);
      function.getArgument(runtimeShape.index).setType(specialized);
      inputTypes[runtimeShape.index] = specialized;
    }
    function.setType(FunctionType::get(function.getContext(), inputTypes,
                                       function.getResultTypes()));

    function.walk([](ShapeInferIntfc interface) { interface.inferShape(); });
    specializeDestinationBuffers(function);

    bool dynamicTypeFound = false;
    function.walk([&](Operation *operation) {
      if (llvm::any_of(operation->getOperandTypes(), hasDynamicShape) ||
          llvm::any_of(operation->getResultTypes(), hasDynamicShape)) {
        operation->emitError(
            "runtime specialization left a dynamic ATIR tensor");
        dynamicTypeFound = true;
      }
    });
    if (dynamicTypeFound) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createAtirSpecializeShapesPass() {
  return std::make_unique<AtirSpecializeShapesPass>();
}

}  // namespace atir
