#ifndef ATIR_MLIR_CONVERSION_PASSES_H
#define ATIR_MLIR_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "Common/AtirLowering.h"
#include "Dialect/Atir/AtirOps.h"

using namespace mlir;
namespace atir {

std::unique_ptr<mlir::Pass> createConvertAtirToAffine();
std::unique_ptr<mlir::Pass> createConvertAtirToLinalg();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "Conversion/Passes.h.inc"

}  // namespace atir
#endif //ATIR_MLIR_CONVERSION_PASSES_H
