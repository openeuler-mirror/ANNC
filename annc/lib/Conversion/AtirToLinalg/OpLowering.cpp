#include "Conversion/AtirToLinalg/OpLowering.h"
#include "Conversion/Common/AtirLowering.h"
#include "Conversion/Common/CustomizeCallLowering.h"
#include "Conversion/AtirToLinalg/AtirTypeConverter.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace {
    std::vector<AffineMap> getTransposeAttr(MLIRContext *ctx, ArrayRef<bool> transpose)
    {
        std::vector<AffineMap> res;
        if (transpose[0] || transpose[1] || transpose[2])
        {
            auto d0 = mlir::getAffineDimExpr(0, ctx);
            auto d1 = mlir::getAffineDimExpr(1, ctx);
            auto d2 = mlir::getAffineDimExpr(2, ctx);
            AffineMap left, right, output;
            if (transpose[0])
            {
                left = AffineMap::get(3, 0, {d2, d0}, ctx);
            } else {
                right = AffineMap::get(3, 0, {d0, d2}, ctx);
            }
            if (transpose[1])
            {
                right = AffineMap::get(3, 0, {d1, d2}, ctx);
            } else {
                right = AffineMap::get(3, 0, {d2, d1}, ctx);
            }
            if (transpose[2])
            {
                output = AffineMap::get(3, 0, {d1, d0}, ctx);
            } else {
                output = AffineMap::get(3, 0, {d0, d1}, ctx);
            }
            res.push_back(left);
            res.push_back(right);
            res.push_back(output);
        }

        return res;
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
    // patterns.add<ReturnLoweringToLinalg>(inputTypeConverter, patterns.getContext());
    patterns.add<CustomizeLoweringToLinalg>(inputTypeConverter, patterns.getContext());
    patterns.add<FuncReturnOpLowering>(inputTypeConverter, patterns.getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, inputTypeConverter);
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
    MLIRContext *ctx = getContext();
    auto loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto lhs_type = mlir::cast<RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<RankedTensorType>(rhs.getType());
    auto lhs_rank = lhs_type.getRank();
    auto rhs_rank = rhs_type.getRank();
    Value c = adaptor.getC();
    auto attrs = op->getAttrs();
    if (lhs_rank == 2 && rhs_rank == 2)
    {
        auto linalgMatmul = rewriter.create<linalg::MatmulOp>(loc, c.getType(),ValueRange{lhs, rhs}, c, attrs);
        std::vector<AffineMap> transposeMap = getTransposeAttr(ctx,
            {op.getLeftTranspose(), op.getRightTranspose(), op.getOutputTranspose()});
        if (!transposeMap.empty())
        {
            linalgMatmul->setAttr("indexing_maps", rewriter.getAffineMapArrayAttr(transposeMap));
        }
        Value matmul = linalgMatmul.getResult(0);
        rewriter.replaceOp(op, matmul);
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
