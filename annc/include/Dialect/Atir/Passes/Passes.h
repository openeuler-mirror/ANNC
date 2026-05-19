#ifndef ATIR_PASS_H
#define ATIR_PASS_H
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir/AtirOps.h"

using namespace mlir;
namespace atir {
#define GEN_PASS_DECL
#include "Dialect/Atir/Passes/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createAtirTilingPass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirOpFusionPass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirCanonicalizePass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirFastCodegenPass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirDistributePass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirBlockFusionPass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirUnrollPass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirEltwiseFusionPass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirLLMCodeGenPass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirPruneFuncPass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirSelectLoweringStrategyPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "Dialect/Atir/Passes/Passes.h.inc"

void registerAllAtirPasses();
}  // namespace atir
#endif // ATIR_PASS_H