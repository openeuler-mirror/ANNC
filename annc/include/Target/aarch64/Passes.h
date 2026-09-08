#ifndef ATIR_MLIR_AARCH64_CODEGEN_PASSES_H
#define ATIR_MLIR_AARCH64_CODEGEN_PASSES_H

#include "llvm/ADT/StringRef.h"
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
#include "mlir/Dialect/Affine/IR/AffineOps.h"

using namespace mlir;
namespace annc {

std::unique_ptr<mlir::Pass> createKPGemmOneShotBufferize();
std::unique_ptr<mlir::Pass> createAArch64GemmPrepackRhs();
std::unique_ptr<mlir::Pass> createAArch64GemmPrepackRhs(
    llvm::StringRef packedCPath);
std::unique_ptr<mlir::Pass> createAArch64ResolveGemmPlan();
std::unique_ptr<mlir::Pass> createAArch64SelectGemmStrategy();
std::unique_ptr<mlir::Pass> createAArch64SelectGemmStrategy(
    llvm::StringRef configPath, bool enablePrepack, int64_t intraThreadCount);
std::unique_ptr<mlir::Pass> createAArch64AutotuneGemmPlan();
std::unique_ptr<mlir::Pass> createAArch64FinalizeGemmPlan();
std::unique_ptr<mlir::Pass> createAArch64GemmCacheBlocking();
std::unique_ptr<mlir::Pass> createAArch64GemmKernelTiling();
std::unique_ptr<mlir::Pass> createAArch64GemmLeafMaterialization();
std::unique_ptr<mlir::Pass> createAArch64GemmMicrokernelLowering();
std::unique_ptr<mlir::Pass> createAArch64GemmABILowering();
std::unique_ptr<mlir::Pass> createAArch64GemmEpilogueLowering();
std::unique_ptr<mlir::Pass> createAArch64VerifyGemmSchedule();
std::unique_ptr<mlir::Pass> createCacheParallel();
std::unique_ptr<mlir::Pass> createMatmulPackAffine();
std::unique_ptr<mlir::Pass> createCacheReduction();
std::unique_ptr<mlir::Pass> createVectorCommonParallel();
std::unique_ptr<mlir::Pass> createVectorReduction();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "Target/aarch64/Passes.h.inc"

void buildAArch64CodegenPipeline(OpPassManager& passManager);
void registerAArch64CodegenPipeline();
}
#endif // PIMP_MLIR_AARCH64_CODEGEN_PASSES_H
