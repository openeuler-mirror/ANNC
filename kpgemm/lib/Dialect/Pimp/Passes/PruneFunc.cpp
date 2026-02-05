#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Pimp/PimpOps.h"
#include "Dialect/Pimp/PimpTypes.h.inc"
#include "Dialect/Pimp/Passes/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "Helper.h"
#include "iostream"

using namespace llvm;
using namespace mlir;

namespace pimp {

    class PimpPruneFuncPass : public PimpPruneFuncBase<PimpPruneFuncPass> {
    public:
        PimpPruneFuncPass() = default;

        void runOnOperation() override {
            std::cout << "this is PimpPruneFuncPass" << std::endl;

            auto m = getOperation();
            for (auto func : llvm::make_early_inc_range(
                    m.getOps<func::FuncOp>())) {
                if (!func->hasAttr("fusion.pattern")) {
                    llvm::errs() << "Erase func: " << func.getName() << "\n";
                    func.erase();
                }
            }
        }
    };

    std::unique_ptr<OperationPass<ModuleOp>> createPimpPruneFuncPass() {
        std::cout << "this is createPimpPruneFuncPass" << std::endl;
        return std::make_unique<PimpPruneFuncPass>();
    }
}  // namespace pimp