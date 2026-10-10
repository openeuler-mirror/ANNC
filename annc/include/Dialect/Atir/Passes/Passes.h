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

std::unique_ptr<OperationPass<ModuleOp>> createAtirOpFusionPass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirConfigFusionPass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirIdentityCanonicalizePass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirGemmEpilogueFusionPass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirFoldBatchNormPass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirRankInferencePass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirFastCodegenPass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirPruneFuncPass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirSelectKernelPass();
std::unique_ptr<OperationPass<ModuleOp>> createAtirSpecializeShapesPass();
void populateAtirOpFusionPatterns(RewritePatternSet &patterns);

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "Dialect/Atir/Passes/Passes.h.inc"

void registerAllAtirPasses();
}  // namespace atir
#endif // ATIR_PASS_H
