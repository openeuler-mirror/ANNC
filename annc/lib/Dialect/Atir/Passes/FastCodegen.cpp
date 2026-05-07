#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "iostream"

using namespace llvm;
using namespace mlir;

namespace atir {
    class AtirFastCodegenPass : public AtirFastCodegenBase<AtirFastCodegenPass> {
    public:
        AtirFastCodegenPass() = default;

        void runOnOperation() override {
            std::cout << "this is AtirFastCodegenPass" << std::endl;
            auto m = getOperation();
            //todo c++
            //todo 
            //1.mlir
            //2.pass?

            //todo
            // 1.
            // 2.c++matmul-add-relu

        }
    };

    std::unique_ptr<OperationPass<ModuleOp>> createAtirFastCodegenPass() {
        std::cout << "this is createAtirFastCodegenPass" << std::endl;
        return std::make_unique<AtirFastCodegenPass>();
    }
}  // namespace atir