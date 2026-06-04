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
        AtirFastCodegenPass(const AtirFastCodegenPass &) = default;

        void runOnOperation() override {
            auto m = getOperation();
            auto ctx = m.getContext();

            // Set annc.enable_kdnn module attribute from the --enable-kdnn option.
            // This flag controls whether KDNN optimized kernels are preferred
            // over aarch64 defaults during kernel resolution.
            if (enableKdnn) {
                m->setAttr("annc.enable_kdnn", BoolAttr::get(ctx, true));
                llvm::dbgs() << "ANNC: KDNN optimization enabled\n";
            }

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
