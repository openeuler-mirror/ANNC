#include <cstdlib>

#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistry.h"
#include "Support/LLVMCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;

namespace atir {

class AtirFastCodegenPass : public AtirFastCodegenBase<AtirFastCodegenPass> {
 public:
  AtirFastCodegenPass() = default;
  AtirFastCodegenPass(const AtirFastCodegenPass &) = default;

  void runOnOperation() override {
    auto m = getOperation();
    auto ctx = m.getContext();

    // Set annc.enable_kdnn module attribute from the --enable-kdnn option.
    // This flag controls whether KDNN optimized kernels are preferred
    // over aarch64 defaults during kernel resolution.
    if (enableKdnn) {
      m->setAttr("annc.enable_kdnn", BoolAttr::get(ctx, true));
    }

    GreedyRewriteConfig config;
    annc::setRegionSimplificationLevel(config, GreedySimplifyRegionLevel::Disabled);

    RewritePatternSet patterns(ctx);
    SmallVector<std::string> environmentEnabledCustomOps;
    appendCustomOpTypesFromEnv(kEnableCustomOpsEnv,
                               environmentEnabledCustomOps);
    SmallVector<std::string> effectiveEnabledCustomOps;
    if (enabledCustomOps.empty()) {
      effectiveEnabledCustomOps = std::move(environmentEnabledCustomOps);
    } else {
      effectiveEnabledCustomOps.append((*enabledCustomOps).begin(),
                                       (*enabledCustomOps).end());
    }

    SmallVector<std::string> effectiveDisabledCustomOps;
    appendCustomOpTypesFromEnv(kDisableCustomOpsEnv,
                               effectiveDisabledCustomOps);
    effectiveDisabledCustomOps.append((*disabledCustomOps).begin(),
                                      (*disabledCustomOps).end());

    CustomOpTypeFilter customOpFilter(effectiveEnabledCustomOps,
                                      effectiveDisabledCustomOps);
    PatternRegistry::instance().populatePatterns(patterns, customOpFilter);
    (void)applyPatternsGreedily(m, std::move(patterns), config);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAtirFastCodegenPass() {
  return std::make_unique<AtirFastCodegenPass>();
}
}  // namespace atir
