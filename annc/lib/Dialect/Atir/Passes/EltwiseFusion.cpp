#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "iostream"

using namespace llvm;
using namespace mlir;

namespace atir {
    class AtirEltwiseFusionPass : public AtirEltwiseFusionBase<AtirEltwiseFusionPass> {
    public:
        AtirEltwiseFusionPass() = default;

        void runOnOperation() override {
            std::cout << "this is AtirEltwiseFusionPass" << std::endl;
            //todo ,addrelu
            //todo
            // 1.loweringaffineaddrelufunc
            //  1.1 func
            //  1.2 func
            // 2.func lowingaffine
            //  2.1 affine, linalg, scf

        }
    };

    std::unique_ptr<OperationPass<ModuleOp>> createAtirEltwiseFusionPass() {
        std::cout << "this is createAtirEltwiseFusionPass" << std::endl;
        return std::make_unique<AtirEltwiseFusionPass>();
    }
}  // namespace atir