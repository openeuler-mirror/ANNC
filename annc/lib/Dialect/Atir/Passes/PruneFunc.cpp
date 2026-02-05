#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/AtirTypes.h.inc"
#include "Dialect/Atir/Passes/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "Helper.h"
#include "iostream"

using namespace llvm;
using namespace mlir;

namespace atir {

    class AtirPruneFuncPass : public AtirPruneFuncBase<AtirPruneFuncPass> {
    public:
        AtirPruneFuncPass() = default;

        void runOnOperation() override {
            std::cout << "this is AtirPruneFuncPass" << std::endl;

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

    std::unique_ptr<OperationPass<ModuleOp>> createAtirPruneFuncPass() {
        std::cout << "this is createAtirPruneFuncPass" << std::endl;
        return std::make_unique<AtirPruneFuncPass>();
    }
}  // namespace atir