#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Pimp/PimpOps.h"
#include "Dialect/Pimp/Passes/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "iostream"

using namespace llvm;
using namespace mlir;

namespace pimp {
    class PimpEltwiseFusionPass : public PimpEltwiseFusionBase<PimpEltwiseFusionPass> {
    public:
        PimpEltwiseFusionPass() = default;

        void runOnOperation() override {
            std::cout << "this is PimpEltwiseFusionPass" << std::endl;
            //todo ,addrelu
            //todo
            // 1.loweringaffineaddrelufunc
            //  1.1 func
            //  1.2 func
            // 2.func lowingaffine
            //  2.1 affine, linalg, scf

        }
    };

    std::unique_ptr<OperationPass<ModuleOp>> createPimpEltwiseFusionPass() {
        std::cout << "this is createPimpEltwiseFusionPass" << std::endl;
        return std::make_unique<PimpEltwiseFusionPass>();
    }
}  // namespace pimp