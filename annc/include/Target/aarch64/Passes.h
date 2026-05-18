#ifndef PIMP_MLIR_AARCH64_CODEGEN_PASSES_H
#define PIMP_MLIR_AARCH64_CODEGEN_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
namespace annc {

std::unique_ptr<mlir::Pass> createKPGemmOneShotBufferize();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "Target/aarch64/Passes.h.inc"

void buildAArch64CodegenPipeline(PassManager& passManager);
}
#endif // PIMP_MLIR_AARCH64_CODEGEN_PASSES_H
