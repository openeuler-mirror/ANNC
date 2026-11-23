#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace {

class FrameworkCustomCallToStablehloPass
    : public PassWrapper<FrameworkCustomCallToStablehloPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      FrameworkCustomCallToStablehloPass)
  StringRef getArgument() const final {
    return "framework-custom-call-to-stablehlo";
  }
  StringRef getDescription() const final {
    return "Lower framework custom-call placeholders to StableHLO";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<stablehlo::StablehloDialect>();
  }
  void runOnOperation() override {
    SmallVector<Operation *> markers;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "annc_framework.custom_call")
        markers.push_back(op);
    });
    IRRewriter builder(&getContext());
    for (Operation *op : markers) {
      builder.setInsertionPoint(op);
      OperationState state(op->getLoc(), "stablehlo.custom_call");
      state.addOperands(op->getOperands());
      state.addTypes(op->getResultTypes());
      state.addAttributes(op->getAttrs());
      builder.replaceOp(op, builder.create(state)->getResults());
    }
  }
};

}  // namespace

void registerFrameworkCustomCallToStablehloPass() {
  PassRegistration<FrameworkCustomCallToStablehloPass>();
}
