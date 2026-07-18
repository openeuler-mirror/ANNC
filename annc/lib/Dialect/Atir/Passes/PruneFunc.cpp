#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/AtirTypes.h.inc"
#include "Dialect/Atir/Passes/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "Helper.h"

using namespace llvm;
using namespace mlir;

namespace atir {

    class AtirPruneFuncPass : public AtirPruneFuncBase<AtirPruneFuncPass> {
    public:
        AtirPruneFuncPass() = default;

        void runOnOperation() override {
            auto m = getOperation();
            for (auto func : llvm::make_early_inc_range(
                    m.getOps<func::FuncOp>())) {
                if (!func->hasAttr("fusion.pattern")) {
                    func.erase();
                }
            }
        }
    };

    std::unique_ptr<OperationPass<ModuleOp>> createAtirPruneFuncPass() {
        return std::make_unique<AtirPruneFuncPass>();
    }
}  // namespace atir