#ifndef ANNC_LIB_TARGET_AARCH64_GEMM_GEMMTILINGUTILS_H
#define ANNC_LIB_TARGET_AARCH64_GEMM_GEMMTILINGUTILS_H

#include "GemmStrategy.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

namespace annc::aarch64::gemm {

struct MemRefBaseAndOffset {
  mlir::Value base;
  mlir::Value offset;
};

struct GemmTileShape {
  int64_t m;
  int64_t n;
  int64_t k;
};

mlir::FailureOr<GemmTileShape> readGemmTileShape(mlir::Operation *op);

void disableLoopUnrolling(mlir::OpBuilder &builder, mlir::scf::ForOp loop);
mlir::LogicalResult materializeGemmTileRun(
    mlir::OpBuilder &builder, mlir::Location loc, const GemmTileRun &run,
    llvm::function_ref<mlir::LogicalResult(mlir::Value, int64_t)> emitTile);
mlir::LogicalResult materializeGemmPartition(
    mlir::OpBuilder &builder, mlir::Location loc,
    const GemmPartition &partition,
    llvm::function_ref<mlir::LogicalResult(mlir::Value, int64_t)> emitTile);

mlir::FailureOr<MemRefBaseAndOffset> getMemRefBaseAndOffset(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value memref);
mlir::Value castToUnrankedF32MemRef(mlir::OpBuilder &builder,
                                    mlir::Location loc, mlir::Value memref);
mlir::LogicalResult validateGemmGeneric(mlir::linalg::GenericOp generic);
mlir::FailureOr<mlir::linalg::GenericOp> cloneTiledGemmGeneric(
    mlir::OpBuilder &builder, mlir::linalg::GenericOp generic,
    mlir::ArrayRef<mlir::OpFoldResult> iterationOffsets,
    mlir::ArrayRef<mlir::OpFoldResult> iterationSizes);
// Clones `generic` into a statically sized (mSize x nSize x kSize) tile of
// the GEMM iteration space at the given index offsets, tagging the result
// with `stage`.
mlir::FailureOr<mlir::linalg::GenericOp> createTiledGemmGeneric(
    mlir::OpBuilder &builder, mlir::linalg::GenericOp generic,
    mlir::Value mOffset, mlir::Value nOffset, mlir::Value kOffset,
    int64_t mSize, int64_t nSize, int64_t kSize, mlir::StringRef stage);
// Creates a statically sized (mSize x nSize x kSize) row-major matmul tile
// over (lhs, rhs, out) at the given index offsets, carrying `attributes` to
// the tiled op. The caller tags the stage.
mlir::linalg::MatmulOp createTiledMatmul(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
    mlir::Value rhs, mlir::Value out, mlir::Value mOffset, mlir::Value nOffset,
    mlir::Value kOffset, int64_t mSize, int64_t nSize, int64_t kSize,
    llvm::ArrayRef<mlir::NamedAttribute> attributes);

}  // namespace annc::aarch64::gemm

#endif  // ANNC_LIB_TARGET_AARCH64_GEMM_GEMMTILINGUTILS_H
