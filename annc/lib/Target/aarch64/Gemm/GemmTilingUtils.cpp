#include "GemmTilingUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

namespace annc::aarch64::gemm {
namespace {

mlir::Value materializeIndex(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::OpFoldResult value) {
  if (auto dynamicValue = llvm::dyn_cast<mlir::Value>(value))
    return dynamicValue;
  auto attr = llvm::cast<mlir::IntegerAttr>(llvm::cast<mlir::Attribute>(value));
  return builder.create<mlir::arith::ConstantIndexOp>(loc, attr.getInt());
}

mlir::FailureOr<MemRefBaseAndOffset> getBaseAndOffsetImpl(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value memref) {
  if (auto castOp = memref.getDefiningOp<mlir::memref::CastOp>())
    return getBaseAndOffsetImpl(builder, loc, castOp.getSource());

  auto subview = memref.getDefiningOp<mlir::memref::SubViewOp>();
  if (!subview) {
    auto type = llvm::dyn_cast<mlir::MemRefType>(memref.getType());
    if (!type || !type.getLayout().isIdentity()) return mlir::failure();
    mlir::Value zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    return MemRefBaseAndOffset{memref, zero};
  }

  mlir::FailureOr<MemRefBaseAndOffset> parent =
      getBaseAndOffsetImpl(builder, loc, subview.getSource());
  if (mlir::failed(parent)) return mlir::failure();

  auto sourceType =
      llvm::dyn_cast<mlir::MemRefType>(subview.getSource().getType());
  llvm::SmallVector<int64_t> strides;
  int64_t staticOffset = 0;
  if (!sourceType ||
      mlir::failed(sourceType.getStridesAndOffset(strides, staticOffset))) {
    return mlir::failure();
  }

  std::optional<mlir::memref::ExtractStridedMetadataOp> metadata;
  if (llvm::is_contained(strides, mlir::ShapedType::kDynamic))
    metadata = builder.create<mlir::memref::ExtractStridedMetadataOp>(
        loc, subview.getSource());

  mlir::Value offset = parent->offset;
  for (auto [dimension, mixedOffset, stride] :
       llvm::enumerate(subview.getMixedOffsets(), strides)) {
    mlir::Value coordinate = materializeIndex(builder, loc, mixedOffset);
    if (stride != 1) {
      mlir::Value strideValue;
      if (stride == mlir::ShapedType::kDynamic)
        strideValue = metadata->getStrides()[dimension];
      else
        strideValue = builder.create<mlir::arith::ConstantIndexOp>(loc, stride);
      coordinate =
          builder.create<mlir::arith::MulIOp>(loc, coordinate, strideValue);
    }
    offset = builder.create<mlir::arith::AddIOp>(loc, offset, coordinate);
  }
  return MemRefBaseAndOffset{parent->base, offset};
}

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

mlir::FailureOr<MemRefBaseAndOffset> getMemRefBaseAndOffset(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value memref) {
  return getBaseAndOffsetImpl(builder, loc, memref);
}

mlir::Value castToUnrankedF32MemRef(mlir::OpBuilder &builder,
                                    mlir::Location loc, mlir::Value memref) {
  auto type = mlir::UnrankedMemRefType::get(builder.getF32Type(), 0);
  return builder.create<mlir::memref::CastOp>(loc, type, memref);
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
