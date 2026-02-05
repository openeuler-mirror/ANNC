#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "iostream"

using namespace llvm;
using namespace mlir;

namespace atir {

    atir::AddOp findFollowingAddOp(atir::MatMulOp matmulOp) {
        auto result = matmulOp.getResult();
        // travel MatMul
        for (auto* user : result.getUsers()) {
            if (auto addOp = llvm::dyn_cast<atir::AddOp>(user)) {
                return addOp;
            }
        }
        return nullptr;
    }

    atir::ReluOp findFollowingReluOp(atir::AddOp addOp) {
        auto result = addOp.getResult();
        // travel Add
        for (auto* user : result.getUsers()) {
            if (auto reluOp = llvm::dyn_cast<atir::ReluOp>(user)) {
                return reluOp;
            }
        }
        return nullptr;
    }

    bool isSupportOpenxla (ModuleOp m)
    {
        //todo openxla,
        return false;
    }

    bool isSupportKPGemm (mlir::ModuleOp m)
    {
        //todo annc (eltwise)
        bool foundPattern = false;
        m->walk([&](func::FuncOp funcOp) {
            auto* block = &funcOp.front();
            //travel
            for (auto& op : *block) {
                //find matmul
                if (auto matmulOp = llvm::dyn_cast<atir::MatMulOp>(op)) {
                    // find add
                    if (auto addOp = findFollowingAddOp(matmulOp)) {
                        // find relu
                        if (auto reluOp = findFollowingReluOp(addOp)) {
                            foundPattern = true; // 
                            std::cout << "found matmul add relu pattern" << std::endl;
                            return WalkResult::interrupt();
                        }
                    }
                }
            }
            return WalkResult::advance();
        });
        return foundPattern;
    }

    bool isSupportLLM (ModuleOp m)
    {
        //todo LLM
        return false;
    }

    class AtirDistributePass : public AtirDistributeBase<AtirDistributePass> {
    public:
        AtirDistributePass() = default;

        void runOnOperation() override {
            std::cout << "this is AtirDistributePass start" << std::endl;
            auto m = getOperation();

            //todo 

            if (isSupportKPGemm(m))
            {
                std::cout << "start KPGemm" << std::endl;
                mlir::PassManager pm(m.getContext(),PassManager::getAnyOpAnchorName(),mlir::OpPassManager::Nesting::Implicit);
                pm.addPass(createAtirTilingPass());
                pm.addPass(createAtirCanonicalizePass());
                pm.addPass(createAtirFastCodegenPass());
                if (pm.run(m.getOperation()).failed())
                {
                    llvm::report_fatal_error("run KPGemm fusion error");
                }
                return;
            }

            if (isSupportOpenxla(m))
            {
                mlir::PassManager pm(m.getContext(),PassManager::getAnyOpAnchorName(),mlir::OpPassManager::Nesting::Implicit);
                //todo openxlapass
                //pm.addPass(createAtirTilingPass());
                if (pm.run(m.getOperation()).failed())
                    llvm::report_fatal_error("run Openxla fusion error");
                return;
            }

            if (isSupportLLM(m))
            {
                mlir::PassManager pm(m.getContext(),PassManager::getAnyOpAnchorName(),mlir::OpPassManager::Nesting::Implicit);
                //todo llmpass
                //pm.addPass(createAtirTilingPass());
                if (pm.run(m.getOperation()).failed())
                {
                    llvm::report_fatal_error("run LLM fusion error");
                }
                return;
            }

            std::cout << "this is AtirDistributePass end" << std::endl;

            return;
        }
    };

    std::unique_ptr<OperationPass<ModuleOp>> createAtirDistributePass() {
        std::cout << "this is createAtirDistributePass" << std::endl;
        return std::make_unique<AtirDistributePass>();
    }
}  // namespace atir