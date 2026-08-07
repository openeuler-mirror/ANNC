#ifndef ANNC_LIB_TARGET_AARCH64_GEMM_GEMMTILINGUTILS_H
#define ANNC_LIB_TARGET_AARCH64_GEMM_GEMMTILINGUTILS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

namespace annc::aarch64::gemm {

void disableLoopUnrolling(mlir::OpBuilder &builder, mlir::scf::ForOp loop);
mlir::LogicalResult validateGemmGeneric(mlir::linalg::GenericOp generic);
mlir::FailureOr<mlir::linalg::GenericOp> cloneTiledGemmGeneric(
    mlir::OpBuilder &builder, mlir::linalg::GenericOp generic,
    mlir::ArrayRef<mlir::OpFoldResult> iterationOffsets,
    mlir::ArrayRef<mlir::OpFoldResult> iterationSizes);

}  // namespace annc::aarch64::gemm

#endif  // ANNC_LIB_TARGET_AARCH64_GEMM_GEMMTILINGUTILS_H
