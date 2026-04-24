#ifndef PIMP_PASS_H
#define PIMP_PASS_H
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Pimp/PimpOps.h"

using namespace mlir;
namespace pimp {
#define GEN_PASS_DECL
#include "Dialect/Pimp/Passes/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createPimpTilingPass();
std::unique_ptr<OperationPass<ModuleOp>> createPimpOpFusionPass();
std::unique_ptr<OperationPass<ModuleOp>> createPimpCanonicalizePass();
std::unique_ptr<OperationPass<ModuleOp>> createPimpFastCodegenPass();
std::unique_ptr<OperationPass<ModuleOp>> createPimpDistributePass();
std::unique_ptr<OperationPass<ModuleOp>> createPimpBlockFusionPass();
std::unique_ptr<OperationPass<ModuleOp>> createPimpUnrollPass();
std::unique_ptr<OperationPass<ModuleOp>> createPimpEltwiseFusionPass();
std::unique_ptr<OperationPass<ModuleOp>> createPimpLLMCodeGenPass();
std::unique_ptr<OperationPass<ModuleOp>> createPimpPruneFuncPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "Dialect/Pimp/Passes/Passes.h.inc"

void registerAllPimpPasses();
}  // namespace pimp
#endif // PIMP_PASS_H