#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "Support/Log.h"

using namespace llvm;
using namespace mlir;

namespace atir {
    class AtirEltwiseFusionPass : public AtirEltwiseFusionBase<AtirEltwiseFusionPass> {
    public:
        AtirEltwiseFusionPass() = default;

        void runOnOperation() override {
            ANNC_LOG_DEBUG("eltwise-fusion") << "this is AtirEltwiseFusionPass\n";
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
        ANNC_LOG_DEBUG("eltwise-fusion") << "this is createAtirEltwiseFusionPass\n";
        return std::make_unique<AtirEltwiseFusionPass>();
    }
}  // namespace atir
