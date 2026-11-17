#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/FusionV2/Patterns.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Support/Log.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace atir {

namespace {
class AtirOpFusionV2Pass
    : public PassWrapper<AtirOpFusionV2Pass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AtirOpFusionV2Pass)
  llvm::StringRef getArgument() const override { return "atir-op-fusion-v2"; }
  llvm::StringRef getDescription() const override {
    return "operator fusion over the stablehlo+atir_v2 mixed carrier (v2)";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    func::FuncOp main = module.lookupSymbol<func::FuncOp>("main");
    if (!main) {
      ANNC_LOG_DEBUG("op-fusion-v2") << "no @main; nothing to probe\n";
      return;
    }

    // Cascade ports live exactly one runOnOperation; the probes run
    // main-first so a later site's boundary reads this run's entries.
    fusionv2::CallPortTable portTable;
    RewritePatternSet patterns(&getContext());
    populateAtirOpFusionV2Patterns(patterns, &portTable);

    // tf.If passthrough branch funcs (_functionalize_if_*_branch) hold
    // sites too (presort measurement); kernel funcs are skipped - their
    // members are already absorbed.
    SmallVector<func::FuncOp> probes;
    for (func::FuncOp fn : module.getOps<func::FuncOp>()) {
      if (fn->hasAttr("fusion.metadata")) continue;
      probes.push_back(fn);
    }
    llvm::stable_sort(probes, [](func::FuncOp a, func::FuncOp b) {
      return (a.getName() == "main") > (b.getName() == "main");
    });
    FrozenRewritePatternSet frozen(std::move(patterns));
    for (func::FuncOp fn : probes) {
      if (failed(applyPatternsAndFoldGreedily(fn, frozen))) {
        signalPassFailure();
        return;
      }
    }

    int kernels = 0;
    for (func::FuncOp fn : module.getOps<func::FuncOp>())
      if (fn->hasAttr("fusion.metadata")) ++kernels;
    ANNC_LOG_INFO("op-fusion-v2")
        << "fusion kernels in module: " << kernels << "\n";
  }
};
}  // namespace

void populateAtirOpFusionV2Patterns(RewritePatternSet &patterns,
                                    fusionv2::CallPortTable *portTable) {
  addKpGatherV2Pattern(patterns, portTable);
}

std::unique_ptr<OperationPass<ModuleOp>> createAtirOpFusionV2Pass() {
  return std::make_unique<AtirOpFusionV2Pass>();
}

void registerAtirOpFusionV2Pass() {
  PassRegistration<AtirOpFusionV2Pass>();
}

}  // namespace atir
