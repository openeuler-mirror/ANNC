#include "GemmTilingUtils.h"

#include "GemmPlan.h"
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

  if (llvm::isa<mlir::UnrankedMemRefType>(memref.getType())) {
    mlir::Value zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    return MemRefBaseAndOffset{memref, zero};
  }

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
    if (auto constant = llvm::dyn_cast<mlir::AffineConstantExpr>(expression);
        constant && constant.getValue() == 0) {
      offsets.push_back(builder.getIndexAttr(0));
      sizes.push_back(builder.getIndexAttr(1));
      strides.push_back(builder.getIndexAttr(1));
      continue;
    }
    if (!dim || dim.getPosition() >= iterationOffsets.size()) {
      generic.emitOpError("requires projected or constant-zero indexing maps");
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

mlir::FailureOr<GemmTileShape> readGemmTileShape(mlir::Operation *op) {
  auto lhs = llvm::dyn_cast<mlir::MemRefType>(getGemmInput(op, 0).getType());
  auto rhs = llvm::dyn_cast<mlir::MemRefType>(getGemmInput(op, 1).getType());
  auto out = llvm::dyn_cast<mlir::MemRefType>(getGemmOutput(op, 0).getType());
  for (auto type : {lhs, rhs, out}) {
    if (!type || type.getRank() != 2 || !type.hasStaticShape() ||
        !type.getElementType().isF32()) {
      op->emitOpError("requires static rank-2 f32 GEMM memrefs");
      return mlir::failure();
    }
  }
  GemmTileShape shape{lhs.getDimSize(0), rhs.getDimSize(1), lhs.getDimSize(1)};
  if (shape.m <= 0 || shape.n <= 0 || shape.k <= 0 ||
      rhs.getDimSize(0) != shape.k || out.getDimSize(0) != shape.m ||
      out.getDimSize(1) != shape.n) {
    op->emitOpError("has inconsistent GEMM tile dimensions");
    return mlir::failure();
  }
  return shape;
}

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

mlir::LogicalResult materializeGemmTileRun(
    mlir::OpBuilder &builder, mlir::Location loc, const GemmTileRun &run,
    llvm::function_ref<mlir::LogicalResult(mlir::Value, int64_t)> emitTile) {
  auto index = [&](int64_t value) {
    return builder.create<mlir::arith::ConstantIndexOp>(loc, value);
  };
  if (run.count == 1) return emitTile(index(run.offset), run.extent);
  mlir::Value lower = index(run.offset);
  mlir::Value upper = index(run.offset + run.extent * run.count);
  mlir::Value step = index(run.extent);
  auto loop = builder.create<mlir::scf::ForOp>(loc, lower, upper, step);
  disableLoopUnrolling(builder, loop);
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(loop.getBody());
  return emitTile(loop.getInductionVar(), run.extent);
}

mlir::LogicalResult materializeGemmPartition(
    mlir::OpBuilder &builder, mlir::Location loc,
    const GemmPartition &partition,
    llvm::function_ref<mlir::LogicalResult(mlir::Value, int64_t)> emitTile) {
  for (const auto &run : partition)
    if (mlir::failed(materializeGemmTileRun(builder, loc, run, emitTile)))
      return mlir::failure();
  return mlir::success();
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

mlir::FailureOr<mlir::linalg::GenericOp> createTiledGemmGeneric(
    mlir::OpBuilder &builder, mlir::linalg::GenericOp generic,
    mlir::Value mOffset, mlir::Value nOffset, mlir::Value kOffset,
    int64_t mSize, int64_t nSize, int64_t kSize, mlir::StringRef stage) {
  llvm::SmallVector<mlir::OpFoldResult> offsets = {mOffset, nOffset, kOffset};
  llvm::SmallVector<mlir::OpFoldResult> sizes = {builder.getIndexAttr(mSize),
                                                 builder.getIndexAttr(nSize),
                                                 builder.getIndexAttr(kSize)};
  mlir::FailureOr<mlir::linalg::GenericOp> tiled =
      cloneTiledGemmGeneric(builder, generic, offsets, sizes);
  if (mlir::failed(tiled)) return mlir::failure();
  setStage(*tiled, builder, stage);
  return tiled;
}

mlir::linalg::MatmulOp createTiledMatmul(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
    mlir::Value rhs, mlir::Value out, mlir::Value mOffset, mlir::Value nOffset,
    mlir::Value kOffset, int64_t mSize, int64_t nSize, int64_t kSize,
    llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  llvm::SmallVector<mlir::OpFoldResult> strides = {builder.getIndexAttr(1),
                                                   builder.getIndexAttr(1)};
  auto tile = [&](mlir::Value base, mlir::Value row, mlir::Value column,
                  int64_t rowSize, int64_t columnSize) {
    return builder.create<mlir::memref::SubViewOp>(
        loc, base, llvm::SmallVector<mlir::OpFoldResult>{row, column},
        llvm::SmallVector<mlir::OpFoldResult>{builder.getIndexAttr(rowSize),
                                              builder.getIndexAttr(columnSize)},
        strides);
  };
  mlir::Value lhsTile = tile(lhs, mOffset, kOffset, mSize, kSize);
  mlir::Value rhsTile = tile(rhs, kOffset, nOffset, kSize, nSize);
  mlir::Value outTile = tile(out, mOffset, nOffset, mSize, nSize);
  auto matmul = builder.create<mlir::linalg::MatmulOp>(
      loc, mlir::ValueRange{lhsTile, rhsTile}, mlir::ValueRange{outTile});
  matmul->setAttrs(attributes);
  return matmul;
}

}  // namespace annc::aarch64::gemm
