#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Pimp/PimpOps.h"
#include "Dialect/Pimp/Passes/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "iostream"

using namespace llvm;
using namespace mlir;

namespace pimp {

    pimp::AddOp findFollowingAddOp(pimp::MatMulOp matmulOp) {
        auto result = matmulOp.getResult();
        // travel MatMul
        for (auto* user : result.getUsers()) {
            if (auto addOp = llvm::dyn_cast<pimp::AddOp>(user)) {
                return addOp;
            }
        }
        return nullptr;
    }

    pimp::ReluOp findFollowingReluOp(pimp::AddOp addOp) {
        auto result = addOp.getResult();
        // travel Add
        for (auto* user : result.getUsers()) {
            if (auto reluOp = llvm::dyn_cast<pimp::ReluOp>(user)) {
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
        //todo kpgemm (eltwise)
        bool foundPattern = false;
        m->walk([&](func::FuncOp funcOp) {
            auto* block = &funcOp.front();
            //travel
            for (auto& op : *block) {
                //find matmul
                if (auto matmulOp = llvm::dyn_cast<pimp::MatMulOp>(op)) {
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

    class PimpDistributePass : public PimpDistributeBase<PimpDistributePass> {
    public:
        PimpDistributePass() = default;

        void runOnOperation() override {
            std::cout << "this is PimpDistributePass start" << std::endl;
            auto m = getOperation();

            //todo 

            if (isSupportKPGemm(m))
            {
                std::cout << "start KPGemm" << std::endl;
                mlir::PassManager pm(m.getContext(),PassManager::getAnyOpAnchorName(),mlir::OpPassManager::Nesting::Implicit);
                pm.addPass(createPimpTilingPass());
                pm.addPass(createPimpCanonicalizePass());
                pm.addPass(createPimpFastCodegenPass());
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
                //pm.addPass(createPimpTilingPass());
                if (pm.run(m.getOperation()).failed())
                    llvm::report_fatal_error("run Openxla fusion error");
                return;
            }

            if (isSupportLLM(m))
            {
                mlir::PassManager pm(m.getContext(),PassManager::getAnyOpAnchorName(),mlir::OpPassManager::Nesting::Implicit);
                //todo llmpass
                //pm.addPass(createPimpTilingPass());
                if (pm.run(m.getOperation()).failed())
                {
                    llvm::report_fatal_error("run LLM fusion error");
                }
                return;
            }

            std::cout << "this is PimpDistributePass end" << std::endl;

            return;
        }
    };

    std::unique_ptr<OperationPass<ModuleOp>> createPimpDistributePass() {
        std::cout << "this is createPimpDistributePass" << std::endl;
        return std::make_unique<PimpDistributePass>();
    }
}  // namespace pimp