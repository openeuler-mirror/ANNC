#include "Conversion/AtirToLinalg/OpLowering.h"

#include "Conversion/AtirToLinalg/AtirTypeConverter.h"
#include "Conversion/Common/AtirLowering.h"
#include "Conversion/Common/CustomizeCallLowering.h"
#include "Dialect/Atir/Passes/GemmEpilogueCandidate.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace {
constexpr llvm::StringLiteral kGemmAttrName = "annc.gemm";

Value asTensor(PatternRewriter& rewriter, Location loc, Value value) {
  auto type = llvm::dyn_cast<MemRefType>(value.getType());
  if (!type) return value;
  return rewriter.create<bufferization::ToTensorOp>(
      loc, RankedTensorType::get(type.getShape(), type.getElementType()), value,
      /*restrict=*/true, /*writable=*/true);
}

FailureOr<AffineMap> addBroadcastMap(RankedTensorType input,
                                     RankedTensorType output, AffineExpr m,
                                     AffineExpr n, unsigned dimensions,
                                     MLIRContext* context) {
  if (input.getElementType() != output.getElementType()) return failure();
  auto map = [&](ArrayRef<AffineExpr> results) {
    return AffineMap::get(dimensions, 0, results, context);
  };
  if (input.getShape() == output.getShape()) return map({m, n});
  auto broadcast =
      atir::classifyEpilogueBroadcastShape(output.getShape(), input.getShape());
  if (!broadcast) return failure();
  auto zero = getAffineConstantExpr(0, context);
  switch (*broadcast) {
    case atir::BroadcastKind::kScalar:
      if (input.getRank() == 0) return map({});
      return input.getRank() == 1 ? map({zero}) : map({zero, zero});
    case atir::BroadcastKind::kM:
      return map({m, zero});
    case atir::BroadcastKind::kN:
      return input.getRank() == 1 ? map({n}) : map({zero, n});
    case atir::BroadcastKind::kMatrix:
      return map({m, n});
  }
  llvm_unreachable("unknown epilogue broadcast");
}

void preserveRhsCheckpointName(Operation* target, atir::MatMulOp source) {
  auto rhsType = llvm::dyn_cast<atir::TensorType>(source.getRhs().getType());
  if (!rhsType) return;
  auto name = rhsType.getName();
  if (!name || name.getValue().empty()) return;
  // The AArch64 prepack pass runs after bufferization, where TensorType
  // metadata is no longer available. Preserve the checkpoint tensor name on
  // the lowered GEMM operation.
  target->setDiscardableAttr("annc.aarch64.rhs_name", name);
}

FailureOr<Value> createFusedGemmGeneric(PatternRewriter& rewriter,
                                        atir::MatMulOp op, Value lhs, Value rhs,
                                        Value output,
                                        llvm::ArrayRef<Value> epilogueInputs,
                                        const atir::EpilogueProgram& epilogue) {
  auto lhsType = llvm::dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = llvm::dyn_cast<RankedTensorType>(rhs.getType());
  auto outputType = llvm::dyn_cast<RankedTensorType>(output.getType());
  if (!lhsType || !rhsType || !outputType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || outputType.getRank() != 2)
    return op.emitOpError(
               "fused linalg.generic lowering requires rank-2 tensors"),
           failure();
  int64_t reductionSize = lhsType.getDimSize(1);
  int64_t rhsReductionSize = rhsType.getDimSize(0);
  if (ShapedType::isDynamic(reductionSize) || reductionSize <= 0)
    return op.emitOpError(
               "fused linalg.generic lowering requires a static positive K "
               "dimension"),
           failure();
  if (!ShapedType::isDynamic(rhsReductionSize) &&
      rhsReductionSize != reductionSize)
    return op.emitOpError("has inconsistent lhs/rhs K dimensions"), failure();
  Type elementType = outputType.getElementType();
  if (!llvm::isa<FloatType>(elementType))
    return op.emitOpError(
               "fused linalg.generic lowering requires a floating-point "
               "element type"),
           failure();
  MLIRContext* ctx = rewriter.getContext();
  auto m = rewriter.getAffineDimExpr(0);
  auto n = rewriter.getAffineDimExpr(1);
  unsigned binaryInputCount = 0;
  for (const atir::EpilogueStep& step : epilogue.steps)
    if (atir::isBinaryEpilogueOpcode(step.opcode)) ++binaryInputCount;
  if (epilogueInputs.size() != binaryInputCount) {
    return op.emitOpError(
               "fused epilogue inputs do not match the ordered binary steps"),
           failure();
  }
  unsigned inputIndex = 0;
  SmallVector<AffineMap> epilogueMaps;
  for (const atir::EpilogueStep& step : epilogue.steps) {
    if (!atir::isBinaryEpilogueOpcode(step.opcode)) continue;
    auto biasType =
        llvm::dyn_cast<RankedTensorType>(epilogueInputs[inputIndex].getType());
    if (!biasType)
      return op.emitOpError("epilogue input must be a ranked tensor"),
             failure();
    auto map = addBroadcastMap(biasType, outputType, m, n, 3, ctx);
    if (failed(map))
      return op.emitOpError("epilogue input has an invalid broadcast shape"),
             failure();
    epilogueMaps.push_back(*map);
    ++inputIndex;
  }
  auto k = rewriter.getAffineDimExpr(2);
  SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(3, 0, ArrayRef<AffineExpr>{m, k}, ctx),
      AffineMap::get(3, 0, ArrayRef<AffineExpr>{k, n}, ctx)};
  SmallVector<Value> inputs = {lhs, rhs};
  inputs.append(epilogueInputs.begin(), epilogueInputs.end());
  indexingMaps.append(epilogueMaps.begin(), epilogueMaps.end());
  indexingMaps.push_back(AffineMap::get(3, 0, ArrayRef<AffineExpr>{m, n}, ctx));
  SmallVector<utils::IteratorType> iteratorTypes = {
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::reduction};
  const unsigned accumulatorIndex = 2 + binaryInputCount;
  auto generic = rewriter.create<linalg::GenericOp>(
      op.getLoc(), TypeRange{output.getType()}, inputs, ValueRange{output},
      indexingMaps, iteratorTypes,
      [accumulatorIndex](OpBuilder& builder, Location loc, ValueRange args) {
        Value product = builder.create<arith::MulFOp>(loc, args[0], args[1]);
        Value sum =
            builder.create<arith::AddFOp>(loc, args[accumulatorIndex], product);
        builder.create<linalg::YieldOp>(loc, sum);
      });
  for (NamedAttribute attr : op->getAttrs())
    generic->setAttr(attr.getName(), attr.getValue());
  generic->setAttr(kGemmAttrName, UnitAttr::get(ctx));
  preserveRhsCheckpointName(generic, op);
  return generic.getResult(0);
}
}  // namespace
namespace atir {
void populateAtirToLinalgConversionPatterns(TypeConverter& inputTypeConverter,
                                            TypeConverter& atirTypeConverter,
                                            RewritePatternSet& patterns) {
  patterns.add<ConstantLoweringToLinalg>(atirTypeConverter,
                                         patterns.getContext());
  patterns.add<MatMulLoweringToLinalg>(atirTypeConverter,
                                       patterns.getContext());
  patterns.add<AddLoweringToLinalg>(atirTypeConverter, patterns.getContext());
  patterns.add<ReluLoweringToLinalg>(atirTypeConverter, patterns.getContext());
  patterns.add<BufferLoweringToLinalg>(atirTypeConverter,
                                       patterns.getContext());
  patterns.add<ReshapeLoweringToLinalg>(atirTypeConverter,
                                        patterns.getContext());
  patterns.add<CustomizeLoweringToLinalg>(inputTypeConverter,
                                          patterns.getContext());
  patterns.add<FuncReturnOpLowering>(inputTypeConverter, patterns.getContext());
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      patterns, inputTypeConverter);
}

bool isSupportedInsertUnitDimensionBeforeN(atir::ReshapeOp op) {
  auto inputType = llvm::dyn_cast<atir::TensorType>(op.getInput().getType());
  auto resultType = llvm::dyn_cast<atir::TensorType>(op.getResult().getType());
  auto targetShapeType =
      llvm::dyn_cast<atir::TensorType>(op.getTargetShape().getType());
  if (!inputType || !resultType || inputType.getShape().size() != 2 ||
      resultType.getShape().size() != 3 ||
      inputType.getElementType() != resultType.getElementType() ||
      ShapedType::isDynamicShape(inputType.getShape()) ||
      ShapedType::isDynamicShape(resultType.getShape()) ||
      resultType.getShape()[0] != inputType.getShape()[0] ||
      resultType.getShape()[1] != 1 ||
      resultType.getShape()[2] != inputType.getShape()[1] || !targetShapeType ||
      !llvm::isa<IntegerType, IndexType>(targetShapeType.getElementType()) ||
      !targetShapeType.getCacheData() ||
      !llvm::isa<IntegerType, IndexType>(
          targetShapeType.getCacheData().getElementType()) ||
      targetShapeType.getCacheData().getNumElements() != 3)
    return false;
  llvm::SmallVector<int64_t> targetDimensions;
  for (const APInt& dimension :
       targetShapeType.getCacheData().getValues<APInt>())
    targetDimensions.push_back(dimension.getSExtValue());
  return targetDimensions ==
         llvm::SmallVector<int64_t>{inputType.getShape()[0], 1,
                                    inputType.getShape()[1]};
}

mlir::LogicalResult ReshapeLoweringToLinalg::matchAndRewrite(
    atir::ReshapeOp op, atir::ReshapeOp::Adaptor adaptor,
    mlir::ConversionPatternRewriter& rewriter) const {
  if (!isSupportedInsertUnitDimensionBeforeN(op)) return mlir::failure();
  Operation* targetShapeDef = op.getTargetShape().getDefiningOp();
  auto inputType =
      llvm::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  auto resultType = llvm::dyn_cast_or_null<RankedTensorType>(
      getTypeConverter()->convertType(op.getResult().getType()));
  if (!inputType || !resultType)
    return rewriter.notifyMatchFailure(op, "requires ranked tensor types");
  llvm::SmallVector<ReassociationIndices> reassociation = {{0}, {1, 2}};
  auto expanded = rewriter.create<tensor::ExpandShapeOp>(
      op.getLoc(), resultType, adaptor.getInput(), reassociation);
  rewriter.create<bufferization::MaterializeInDestinationOp>(
      op.getLoc(), expanded.getResult(), adaptor.getOutput());
  rewriter.eraseOp(op);
  if (targetShapeDef && targetShapeDef->use_empty())
    rewriter.eraseOp(targetShapeDef);
  return mlir::success();
}

LogicalResult BufferLoweringToLinalg::matchAndRewrite(
    BufferOp op, BufferOp::Adaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto convertedType =
      getTypeConverter()->convertType(op.getOutput().getType());
  auto rankedType = llvm::dyn_cast_or_null<RankedTensorType>(convertedType);
  if (!rankedType || !rankedType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "BufferOp lowering requires a statically shaped tensor");
  auto memrefType =
      MemRefType::get(rankedType.getShape(), rankedType.getElementType());
  auto alloc = rewriter.create<memref::AllocaOp>(op.getLoc(), memrefType);
  auto tensor = rewriter.create<bufferization::ToTensorOp>(
      op.getLoc(), rankedType, alloc, /*restrict=*/true, /*writable=*/true);
  rewriter.replaceOp(op, tensor.getResult());
  return success();
}

void ConstantLoweringToLinalg::Lowering(PatternRewriter& rewriter,
                                        ConstantOpAdaptor adaptor,
                                        ConstantOp op) const {
  auto type = llvm::dyn_cast<atir::TensorType>(op.getData().getType());
  if (!type || !type.getCacheData()) {
    if (!op->use_empty())
      op->emitOpError(
          "constant without cache data cannot be lowered while it has uses");
    else
      rewriter.eraseOp(op);
    return;
  }
  auto tensorType =
      RankedTensorType::get(type.getShape(), type.getElementType());
  rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, tensorType,
                                                 type.getCacheData());
}

void AddLoweringToLinalg::Lowering(PatternRewriter& rewriter,
                                   AddOpAdaptor adaptor, AddOp op) const {
  auto outputType = llvm::dyn_cast<RankedTensorType>(
      getTypeConverter()->convertType(op.getResult().getType()));
  if (!outputType || outputType.getRank() != 2) {
    op.emitOpError("Linalg Add lowering requires a ranked rank-2 output");
    return;
  }

  MLIRContext* ctx = rewriter.getContext();
  auto m = rewriter.getAffineDimExpr(0);
  auto n = rewriter.getAffineDimExpr(1);
  SmallVector<Value> inputs;
  for (Value input : adaptor.getInputs())
    inputs.push_back(asTensor(rewriter, op.getLoc(), input));
  Value output = asTensor(rewriter, op.getLoc(), adaptor.getOutput());
  SmallVector<AffineMap> indexingMaps;
  for (Value input : inputs) {
    auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      op.emitOpError("Linalg Add inputs must be ranked tensors");
      return;
    }
    auto map = addBroadcastMap(inputType, outputType, m, n, 2, ctx);
    if (failed(map)) {
      op.emitOpError(
          "Linalg Add input is not an exact or statically provable "
          "scalar, M, N, or matrix broadcast");
      return;
    }
    indexingMaps.push_back(*map);
  }
  indexingMaps.push_back(AffineMap::get(2, 0, ArrayRef<AffineExpr>{m, n}, ctx));
  SmallVector<utils::IteratorType> iteratorTypes = {
      utils::IteratorType::parallel, utils::IteratorType::parallel};
  auto generic = rewriter.create<linalg::GenericOp>(
      op.getLoc(), TypeRange{outputType}, inputs, ValueRange{output},
      indexingMaps, iteratorTypes,
      [&](OpBuilder& builder, Location loc, ValueRange args) {
        Value value = builder.create<arith::ConstantOp>(
            loc, FloatAttr::get(outputType.getElementType(), 0.0));
        for (Value arg : args.drop_back())
          value = builder.create<arith::AddFOp>(loc, value, arg);
        if (op.getScalarAttr())
          value = builder.create<arith::AddFOp>(
              loc, value,
              builder.create<arith::ConstantOp>(loc, op.getScalarAttr()));
        if (op.getDoRelu()) {
          Value zero = builder.create<arith::ConstantOp>(
              loc, FloatAttr::get(outputType.getElementType(), 0.0));
          value = builder.create<arith::MaxNumFOp>(loc, value, zero);
          const float reluLimit = op.getReluLimit().convertToFloat();
          if (reluLimit >= 0.0f) {
            Value limit = builder.create<arith::ConstantOp>(
                loc, FloatAttr::get(outputType.getElementType(), reluLimit));
            value = builder.create<arith::MinNumFOp>(loc, value, limit);
          }
        }
        builder.create<linalg::YieldOp>(loc, value);
      });
  rewriter.replaceOp(op, generic.getResult(0));
}

LogicalResult ReluLoweringToLinalg::matchAndRewrite(
    atir::ReluOp op, atir::ReluOp::Adaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto output = asTensor(rewriter, op.getLoc(), adaptor.getOutput());
  auto input = asTensor(rewriter, op.getLoc(), adaptor.getInput());
  auto type = llvm::dyn_cast<RankedTensorType>(input.getType());
  if (!type || type.getRank() != 2)
    return rewriter.notifyMatchFailure(op, "requires a ranked rank-2 tensor");
  MLIRContext* ctx = rewriter.getContext();
  auto map = AffineMap::getMultiDimIdentityMap(2, ctx);
  auto generic = rewriter.create<linalg::GenericOp>(
      op.getLoc(), TypeRange{output.getType()}, ValueRange{input},
      ValueRange{output}, ArrayRef<AffineMap>{map, map},
      SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                       utils::IteratorType::parallel},
      [&](OpBuilder& builder, Location loc, ValueRange args) {
        Value zero = builder.create<arith::ConstantOp>(
            loc, FloatAttr::get(type.getElementType(), 0.0));
        Value value = builder.create<arith::MaxNumFOp>(loc, args[0], zero);
        if (op.getReluLimitAttr().getValueAsDouble() >= 0.0)
          value = builder.create<arith::MinNumFOp>(
              loc, value,
              builder.create<arith::ConstantOp>(
                  loc,
                  FloatAttr::get(type.getElementType(),
                                 op.getReluLimitAttr().getValueAsDouble())));
        builder.create<linalg::YieldOp>(loc, value);
      });
  rewriter.replaceOp(op, generic.getResult(0));
  return success();
}

void MatMulLoweringToLinalg::Lowering(PatternRewriter& rewriter,
                                      MatMulOpAdaptor adaptor,
                                      MatMulOp op) const {
  auto loc = op.getLoc();
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  auto lhs_type = mlir::cast<RankedTensorType>(lhs.getType());
  auto rhs_type = mlir::cast<RankedTensorType>(rhs.getType());
  auto lhs_rank = lhs_type.getRank();
  auto rhs_rank = rhs_type.getRank();
  Value c = adaptor.getC();
  if (lhs_rank == 2 && rhs_rank == 2) {
    if (op.getLeftTranspose() || op.getRightTranspose() ||
        op.getOutputTranspose()) {
      op.emitOpError("Linalg MatMul lowering does not support transpose");
      return;
    }
    auto attr = op->getAttrOfType<DictionaryAttr>(atir::kGemmEpilogueAttr);
    llvm::SmallVector<Value> epilogueInputs(adaptor.getEpilogueInputs().begin(),
                                            adaptor.getEpilogueInputs().end());
    if (attr) {
      FailureOr<atir::EpilogueProgram> epilogue =
          atir::parseEpilogueCandidatePlan(attr);
      if (failed(epilogue)) return;
      FailureOr<Value> fused = createFusedGemmGeneric(
          rewriter, op, lhs, rhs, c, epilogueInputs, *epilogue);
      if (failed(fused)) return;
      rewriter.replaceOp(op, *fused);
      return;
    }
    auto linalgMatmul = rewriter.create<linalg::MatmulOp>(
        loc, c.getType(), ValueRange{lhs, rhs}, c, op->getAttrs());
    preserveRhsCheckpointName(linalgMatmul, op);
    rewriter.replaceOp(op, linalgMatmul.getResult(0));
  }
}

void CustomizeLoweringToLinalg::Lowering(mlir::PatternRewriter& rewriter,
                                         atir::CustomizeOpAdaptor adaptor,
                                         atir::CustomizeOp op) const {
  lowerCustomizeOpToFuncCall(rewriter, adaptor, op);
}

}  // namespace atir
