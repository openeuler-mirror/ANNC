#include "GemmTilingUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

namespace annc::aarch64::gemm {
namespace {

bool matchesProjectedMap(mlir::AffineMap map,
                         llvm::ArrayRef<unsigned> dimensions) {
  if (map.getNumDims() != 3 || map.getNumSymbols() != 0 ||
      map.getNumResults() != dimensions.size()) {
    return false;
  }
  for (auto [expression, dimension] :
       llvm::zip_equal(map.getResults(), dimensions)) {
    auto dim = llvm::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dim || dim.getPosition() != dimension) return false;
  }
  return true;
}

mlir::FailureOr<mlir::Value> createProjectedSubview(
    mlir::OpBuilder &builder, mlir::linalg::GenericOp generic,
    mlir::Value operand, mlir::AffineMap map,
    llvm::ArrayRef<mlir::OpFoldResult> iterationOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> iterationSizes) {
  auto type = llvm::dyn_cast<mlir::MemRefType>(operand.getType());
  if (!type || type.getRank() != map.getNumResults()) {
    generic.emitOpError(
        "requires memref operands whose ranks match their indexing maps");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::OpFoldResult> offsets;
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  llvm::SmallVector<mlir::OpFoldResult> strides;
  offsets.reserve(type.getRank());
  sizes.reserve(type.getRank());
  strides.reserve(type.getRank());
  for (mlir::AffineExpr expression : map.getResults()) {
    auto dim = llvm::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dim || dim.getPosition() >= iterationOffsets.size()) {
      generic.emitOpError(
          "requires projected M/N/K indexing maps for every operand");
      return mlir::failure();
    }
    offsets.push_back(iterationOffsets[dim.getPosition()]);
    sizes.push_back(iterationSizes[dim.getPosition()]);
    strides.push_back(builder.getIndexAttr(1));
  }
  return builder
      .create<mlir::memref::SubViewOp>(generic.getLoc(), operand, offsets,
                                       sizes, strides)
      .getResult();
}

}  // namespace

void disableLoopUnrolling(mlir::OpBuilder &builder, mlir::scf::ForOp loop) {
  mlir::MLIRContext *context = builder.getContext();
  auto unroll = mlir::LLVM::LoopUnrollAttr::get(
      context, builder.getBoolAttr(true), /*count=*/{},
      /*runtimeDisable=*/{}, /*full=*/{}, /*followupUnrolled=*/{},
      /*followupRemainder=*/{}, /*followupAll=*/{});
  auto annotation = mlir::LLVM::LoopAnnotationAttr::get(
      context, /*disableNonforced=*/{}, /*vectorize=*/{}, /*interleave=*/{},
      unroll, /*unrollAndJam=*/{}, /*licm=*/{}, /*distribute=*/{},
      /*pipeline=*/{}, /*peeled=*/{}, /*unswitch=*/{}, /*mustProgress=*/{},
      /*isVectorized=*/{}, /*startLoc=*/{}, /*endLoc=*/{},
      /*parallelAccesses=*/{});
  // ControlFlowToLLVM forwards this name into llvm.br's inherent property.
  loop->setAttr("loop_annotation", annotation);
}

mlir::LogicalResult validateGemmGeneric(mlir::linalg::GenericOp generic) {
  if (generic->getNumResults() != 0) {
    return generic.emitOpError(
        "AArch64 GEMM requires buffer-semantics linalg.generic");
  }
  if (generic.getInputs().size() < 2 || generic.getOutputs().size() != 1) {
    return generic.emitOpError(
        "annc.gemm requires A/B inputs and exactly one C output");
  }

  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes =
      generic.getIteratorTypesArray();
  if (iteratorTypes.size() != 3 ||
      iteratorTypes[0] != mlir::utils::IteratorType::parallel ||
      iteratorTypes[1] != mlir::utils::IteratorType::parallel ||
      iteratorTypes[2] != mlir::utils::IteratorType::reduction) {
    return generic.emitOpError(
        "annc.gemm requires M/N/K iterator types "
        "parallel/parallel/reduction");
  }

  llvm::SmallVector<mlir::AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != generic.getInputs().size() + generic.getOutputs().size() ||
      !matchesProjectedMap(maps[0], {0, 2}) ||
      !matchesProjectedMap(maps[1], {2, 1}) ||
      !matchesProjectedMap(maps.back(), {0, 1})) {
    return generic.emitOpError(
        "annc.gemm requires A=(M,K), B=(K,N), and C=(M,N) indexing maps");
  }
  return mlir::success();
}

mlir::FailureOr<mlir::linalg::GenericOp> cloneTiledGemmGeneric(
    mlir::OpBuilder &builder, mlir::linalg::GenericOp generic,
    mlir::ArrayRef<mlir::OpFoldResult> iterationOffsets,
    mlir::ArrayRef<mlir::OpFoldResult> iterationSizes) {
  if (iterationOffsets.size() != 3 || iterationSizes.size() != 3 ||
      mlir::failed(validateGemmGeneric(generic))) {
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value> originalOperands;
  llvm::SmallVector<mlir::Value> tiledOperands;
  llvm::SmallVector<mlir::AffineMap> maps = generic.getIndexingMapsArray();
  originalOperands.append(generic.getInputs().begin(),
                          generic.getInputs().end());
  originalOperands.append(generic.getOutputs().begin(),
                          generic.getOutputs().end());
  tiledOperands.reserve(originalOperands.size());
  for (auto [operand, map] : llvm::zip_equal(originalOperands, maps)) {
    mlir::FailureOr<mlir::Value> tile = createProjectedSubview(
        builder, generic, operand, map, iterationOffsets, iterationSizes);
    if (mlir::failed(tile)) return mlir::failure();
    tiledOperands.push_back(*tile);
  }

  mlir::IRMapping mapping;
  for (auto [original, tiled] :
       llvm::zip_equal(originalOperands, tiledOperands)) {
    mapping.map(original, tiled);
  }
  return llvm::cast<mlir::linalg::GenericOp>(
      builder.clone(*generic.getOperation(), mapping));
}

}  // namespace annc::aarch64::gemm
