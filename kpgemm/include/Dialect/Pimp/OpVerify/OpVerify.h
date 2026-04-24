#ifndef PIMP_OPVERIFY_H
#define PIMP_OPVERIFY_H
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Pimp/PimpOps.h"

using namespace mlir;
namespace pimp {
#define GEN_PASS_DECL
#include "Dialect/Pimp/OpVerify/OpVerify.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createPimpOpVerifyPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "Dialect/Pimp/OpVerify/OpVerify.h.inc"

}  // namespace pimp
#endif // PIMP_OPVERIFY_H