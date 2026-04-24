#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Pimp/PimpOps.h"
#include "Dialect/Pimp/Passes/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "iostream"

using namespace llvm;
using namespace mlir;

namespace pimp {
    class PimpFastCodegenPass : public PimpFastCodegenBase<PimpFastCodegenPass> {
    public:
        PimpFastCodegenPass() = default;

        void runOnOperation() override {
            std::cout << "this is PimpFastCodegenPass" << std::endl;
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

    std::unique_ptr<OperationPass<ModuleOp>> createPimpFastCodegenPass() {
        std::cout << "this is createPimpFastCodegenPass" << std::endl;
        return std::make_unique<PimpFastCodegenPass>();
    }
}  // namespace pimp