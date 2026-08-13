#include "Conversion/AtirToLinalg/OpLowering.h"
#include "Conversion/Common/AtirLowering.h"
#include "Conversion/Common/CustomizeCallLowering.h"
#include "Conversion/AtirToLinalg/AtirTypeConverter.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace {
constexpr llvm::StringLiteral kGemmEpilogueAttrName = "annc.gemm.epilogue";
constexpr llvm::StringLiteral kGemmAttrName = "annc.gemm";

struct GemmEpilogue {
  bool hasBias = false;
  bool hasRelu = false;
  FloatAttr reluLimit;
};

FailureOr<GemmEpilogue> getEpilogue(atir::MatMulOp op) {
  auto attr = op->getAttrOfType<ArrayAttr>(kGemmEpilogueAttrName);
  if (!attr) return GemmEpilogue{};
  GemmEpilogue epilogue;
  bool sawRelu = false;
  for (Attribute attribute : attr) {
    auto step = llvm::dyn_cast<DictionaryAttr>(attribute);
    auto kind = step ? step.getAs<StringAttr>("kind") : StringAttr();
    if (!kind)
      return op.emitOpError("expects each epilogue step to have a kind"),
             failure();
    if (kind.getValue() == "bias_add") {
      auto input = step.getAs<IntegerAttr>("input");
      auto broadcast = step.getAs<StringAttr>("broadcast");
      if (epilogue.hasBias || sawRelu || !input || input.getInt() != 0 ||
          !broadcast || broadcast.getValue() != "n")
        return op.emitOpError(
                   "only a single leading N-axis bias_add epilogue is "
                   "supported"),
               failure();
      epilogue.hasBias = true;
    } else if (kind.getValue() == "relu") {
      auto limit = step.getAs<FloatAttr>("limit");
      if (epilogue.hasRelu || !limit)
        return op.emitOpError("expects one relu epilogue with a limit"),
               failure();
      epilogue.hasRelu = true;
      epilogue.reluLimit = limit;
      sawRelu = true;
    } else {
      return op.emitOpError() << "does not support epilogue step '"
                              << kind.getValue() << "'",
             failure();
    }
  }
  if (!epilogue.hasBias && !epilogue.hasRelu)
    return op.emitOpError("requires a non-empty gemm epilogue"), failure();
  return epilogue;
}

FailureOr<Value> createFusedGemmGeneric(PatternRewriter& rewriter,
                                        atir::MatMulOp op, Value lhs, Value rhs,
                                        Value output, Value bias,
                                        const GemmEpilogue& epilogue) {
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
  if (epilogue.hasBias) {
    auto biasType = llvm::dyn_cast<RankedTensorType>(bias.getType());
    if (!biasType || biasType.getRank() != 1 ||
        biasType.getElementType() != elementType)
      return op.emitOpError(
                 "bias_add epilogue requires a rank-1 bias with the output "
                 "element type"),
             failure();
  }
  MLIRContext* ctx = rewriter.getContext();
  auto m = rewriter.getAffineDimExpr(0);
  auto n = rewriter.getAffineDimExpr(1);
  auto k = rewriter.getAffineDimExpr(2);
  SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(3, 0, ArrayRef<AffineExpr>{m, k}, ctx),
      AffineMap::get(3, 0, ArrayRef<AffineExpr>{k, n}, ctx)};
  SmallVector<Value> inputs = {lhs, rhs};
  if (epilogue.hasBias) {
    inputs.push_back(bias);
    indexingMaps.push_back(AffineMap::get(3, 0, n, ctx));
  }
  indexingMaps.push_back(AffineMap::get(3, 0, ArrayRef<AffineExpr>{m, n}, ctx));
  SmallVector<utils::IteratorType> iteratorTypes = {
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::reduction};
  auto generic = rewriter.create<linalg::GenericOp>(
      op.getLoc(), TypeRange{output.getType()}, inputs, ValueRange{output},
      indexingMaps, iteratorTypes,
      [hasBias = epilogue.hasBias, hasRelu = epilogue.hasRelu,
       reluLimitAttr = epilogue.reluLimit, elementType,
       lastReductionIndex = reductionSize - 1](OpBuilder& builder, Location loc,
                                               ValueRange args) {
        Value product = builder.create<arith::MulFOp>(loc, args[0], args[1]);
        Value sum =
            builder.create<arith::AddFOp>(loc, args[hasBias ? 3 : 2], product);
        Value epilogueValue = sum;
        if (hasBias)
          epilogueValue =
              builder.create<arith::AddFOp>(loc, epilogueValue, args[2]);
        if (hasRelu) {
          auto zeroAttr = FloatAttr::get(elementType, 0.0);
          Value zero = builder.create<arith::ConstantOp>(loc, zeroAttr);
          epilogueValue =
              builder.create<arith::MaxNumFOp>(loc, epilogueValue, zero);
          if (reluLimitAttr.getValueAsDouble() >= 0.0) {
            auto limitAttr =
                FloatAttr::get(elementType, reluLimitAttr.getValueAsDouble());
            Value limit = builder.create<arith::ConstantOp>(loc, limitAttr);
            epilogueValue =
                builder.create<arith::MinNumFOp>(loc, epilogueValue, limit);
          }
        }
        Value reductionIndex = builder.create<linalg::IndexOp>(loc, 2);
        Value lastIndex =
            builder.create<arith::ConstantIndexOp>(loc, lastReductionIndex);
        Value isLastReductionIteration = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, reductionIndex, lastIndex);
        Value result = builder.create<arith::SelectOp>(
            loc, isLastReductionIteration, epilogueValue, sum);
        builder.create<linalg::YieldOp>(loc, result);
      });
  for (NamedAttribute attr : op->getAttrs())
    generic->setAttr(attr.getName(), attr.getValue());
  generic->setAttr(kGemmAttrName, UnitAttr::get(ctx));
  return generic.getResult(0);
}

}
namespace atir
{
void populateAtirToLinalgConversionPatterns(TypeConverter& inputTypeConverter, TypeConverter& atirTypeConverter, RewritePatternSet& patterns)
{
    // patterns.add<NoneLoweringToLinalg>(typeConverter, patterns.getContext());
    // patterns.add<ConstantLoweringToLinalg>(typeConverter, patterns.getContext());
    // patterns.add<ReluLoweringToLinalg>(typeConverter, patterns.getContext());
    // patterns.add<LoadLoweringToLinalg>(typeConverter, patterns.getContext());
    // patterns.add<AddLoweringToLinalg>(typeConverter, patterns.getContext());
    // patterns.add<ConcatLoweringToLinalg>(typeConverter, patterns.getContext());
    patterns.add<MatMulLoweringToLinalg>(atirTypeConverter, patterns.getContext());
    patterns.add<BufferLoweringToLinalg>(atirTypeConverter, patterns.getContext());
    // patterns.add<ReturnLoweringToLinalg>(inputTypeConverter, patterns.getContext());
    patterns.add<CustomizeLoweringToLinalg>(inputTypeConverter, patterns.getContext());
    patterns.add<FuncReturnOpLowering>(inputTypeConverter, patterns.getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, inputTypeConverter);
}

LogicalResult BufferLoweringToLinalg::matchAndRewrite(
    BufferOp op, BufferOp::Adaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto convertedType = getTypeConverter()->convertType(op.getOutput().getType());
  auto rankedType = llvm::dyn_cast_or_null<RankedTensorType>(convertedType);
  if (!rankedType || !rankedType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "BufferOp lowering requires a statically shaped tensor");
  auto alloc = rewriter.create<bufferization::AllocTensorOp>(
      op.getLoc(), rankedType, ValueRange{});
  rewriter.replaceOp(op, alloc.getResult());
  return success();
}

void NoneLoweringToLinalg::Lowering(PatternRewriter& rewriter, NoneOpAdaptor adaptor, NoneOp op) const
{
    rewriter.eraseOp(op);
}

void ConstantLoweringToLinalg::Lowering(PatternRewriter& rewriter, ConstantOpAdaptor adaptor, ConstantOp op) const
{

}

void ReluLoweringToLinalg::Lowering(PatternRewriter &rewriter, ReluOpAdaptor adaptor, ReluOp op) const {

}

void LoadLoweringToLinalg::Lowering(PatternRewriter &rewriter, LoadOpAdaptor adaptor, LoadOp op) const {

}

void AddLoweringToLinalg::Lowering(PatternRewriter& rewriter, AddOpAdaptor adaptor, AddOp op) const
{

}

void ConcatLoweringToLinalg::Lowering(PatternRewriter& rewriter, ConcatOpAdaptor adaptor, ConcatOp op) const
{

}

void MatMulLoweringToLinalg::Lowering(PatternRewriter& rewriter, MatMulOpAdaptor adaptor, MatMulOp op) const
{
    auto loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto lhs_type = mlir::cast<RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<RankedTensorType>(rhs.getType());
    auto lhs_rank = lhs_type.getRank();
    auto rhs_rank = rhs_type.getRank();
    Value c = adaptor.getC();
    if (lhs_rank == 2 && rhs_rank == 2)
    {
        if (op.getLeftTranspose() || op.getRightTranspose() ||
            op.getOutputTranspose()) {
            op.emitOpError("Linalg MatMul lowering does not support transpose");
            return;
        }
        FailureOr<GemmEpilogue> epilogue = getEpilogue(op);
        if (failed(epilogue)) return;
        Value bias = adaptor.getBias();
        if (bias && !epilogue->hasBias) {
            op.emitOpError("a bias operand requires a bias_add gemm epilogue");
            return;
        }
        if (epilogue->hasBias && !bias) {
            op.emitOpError("bias_add epilogue requires a bias operand");
            return;
        }
        if (epilogue->hasBias || epilogue->hasRelu) {
            FailureOr<Value> fused =
                createFusedGemmGeneric(rewriter, op, lhs, rhs, c, bias,
                                       *epilogue);
            if (failed(fused)) return;
            rewriter.replaceOp(op, *fused);
            return;
        }
        auto linalgMatmul = rewriter.create<linalg::MatmulOp>(
            loc, c.getType(), ValueRange{lhs, rhs}, c, op->getAttrs());
        rewriter.replaceOp(op, linalgMatmul.getResult(0));
    }
}

void CustomizeLoweringToLinalg::Lowering(mlir::PatternRewriter &rewriter, atir::CustomizeOpAdaptor adaptor,
                                         atir::CustomizeOp op) const {
  lowerCustomizeOpToFuncCall(rewriter, adaptor, op);
}


void ReturnLoweringToLinalg::Lowering(PatternRewriter &rewriter, ReturnOpAdaptor adaptor, ReturnOp op) const {
    auto ret = adaptor.getResults();
    auto funcReturn = rewriter.create<func::ReturnOp>(op.getLoc(), ret);
    rewriter.replaceOp(op, funcReturn);
}
}
