#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistry.h"

using namespace llvm;
using namespace mlir;

namespace atir {



    class AtirFastCodegenPass : public AtirFastCodegenBase<AtirFastCodegenPass> {
    public:
        AtirFastCodegenPass() = default;

        void runOnOperation() override {
            llvm::dbgs() << "this is PimpFastCodegenPass\n";
            auto m = getOperation();
            auto ctx = m.getContext();
            GreedyRewriteConfig config;
            config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);

            RewritePatternSet patterns(ctx);
            PatternRegistry::instance().populatePatterns(patterns);
            (void)applyPatternsGreedily(m, std::move(patterns), config);

        }
    };

    std::unique_ptr<OperationPass<ModuleOp>> createAtirFastCodegenPass() {
        llvm::dbgs() << "this is createPimpFastCodegenPass\n";
        return std::make_unique<AtirFastCodegenPass>();
    }
}  // namespace atir