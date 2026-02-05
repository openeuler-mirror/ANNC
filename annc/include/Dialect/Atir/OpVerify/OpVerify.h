#ifndef ATIR_OPVERIFY_H
#define ATIR_OPVERIFY_H
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir/AtirOps.h"

using namespace mlir;
namespace atir {
#define GEN_PASS_DECL
#include "Dialect/Atir/OpVerify/OpVerify.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createAtirOpVerifyPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "Dialect/Atir/OpVerify/OpVerify.h.inc"

}  // namespace atir
#endif // ATIR_OPVERIFY_H