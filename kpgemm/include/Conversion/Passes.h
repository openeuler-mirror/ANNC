#ifndef PIMP_MLIR_CONVERSION_PASSES_H
#define PIMP_MLIR_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "Common/PimpLowering.h"
#include "Dialect/Pimp/PimpOps.h"

using namespace mlir;
namespace pimp {

std::unique_ptr<mlir::Pass> createConvertPimpToAffine();
std::unique_ptr<mlir::Pass> createConvertPimpToLinalg();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "Conversion/Passes.h.inc"

}  // namespace pimp
#endif //PIMP_MLIR_CONVERSION_PASSES_H
