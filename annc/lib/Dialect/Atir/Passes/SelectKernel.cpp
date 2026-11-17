#include "Dialect/Atir/Passes/Passes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace atir {
namespace {

class AnncSelectKernelPass : public AnncSelectKernelBase<AnncSelectKernelPass> {
 public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (kernelName.empty()) {
      module.emitError("annc-select-kernel requires kernel-name");
      signalPassFailure();
      return;
    }

    func::FuncOp selected = module.lookupSymbol<func::FuncOp>(kernelName);
    if (!selected || !selected->hasAttr("annc.kernel")) {
      module.emitError() << "cannot find ANNC kernel function '" << kernelName
                         << "'";
      signalPassFailure();
      return;
    }

    for (func::FuncOp function :
         llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
      if (function != selected) function.erase();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createAnncSelectKernelPass() {
  return std::make_unique<AnncSelectKernelPass>();
}

}  // namespace atir
