#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace {

bool isFrameworkValueType(Type type) {
  if (auto tuple = dyn_cast<TupleType>(type))
    return llvm::all_of(tuple.getTypes(), isFrameworkValueType);
  if (isa<stablehlo::TokenType>(type)) return true;
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor) return false;
  Type element = tensor.getElementType();
  if (auto complex = dyn_cast<ComplexType>(element))
    return isa<FloatType>(complex.getElementType());
  return isa<IntegerType, FloatType, quant::QuantizedType>(element);
}

class FrameworkVerifyPass
    : public PassWrapper<FrameworkVerifyPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FrameworkVerifyPass)
  StringRef getArgument() const final { return "verify-framework-stablehlo"; }
  StringRef getDescription() const final {
    return "Reject operations and types outside the StableHLO output contract";
  }
  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      auto checkType = [&](Type type) {
        if (!isFrameworkValueType(type)) {
          op->emitError("type is outside the framework StableHLO contract: ")
              << type;
          signalPassFailure();
        }
      };
      for (Type type : op->getOperandTypes()) checkType(type);
      for (Type type : op->getResultTypes()) checkType(type);
      if (auto function = dyn_cast<func::FuncOp>(op)) {
        for (Type type : function.getArgumentTypes()) checkType(type);
        for (Type type : function.getResultTypes()) checkType(type);
      }
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument arg : block.getArguments()) checkType(arg.getType());
      if (isa<ModuleOp, func::FuncOp, func::ReturnOp, func::CallOp>(op)) return;
      if (op->getName().getDialectNamespace() != "stablehlo") {
        op->emitError("framework output must contain only StableHLO operations "
                       "and module/function structure");
        signalPassFailure();
      }
    });
  }
};

}  // namespace

void registerFrameworkVerifyPass() { PassRegistration<FrameworkVerifyPass>(); }
