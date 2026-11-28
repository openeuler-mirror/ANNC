#include "Conversion/Common/AtirLowering.h"
#include "Dialect/Atir/AtirOps.h"

namespace atir
{
void populateAtirToLinalgConversionPatterns(TypeConverter &inputTypeConverter, TypeConverter &atirTypeConverter, RewritePatternSet &patterns);
void populateAtirV2ToLinalgConversionPatterns(RewritePatternSet &patterns);

#define OpLowering(OP)                                                         \
  struct OP##LoweringToLinalg : public AtirLowering<atir::OP##Op> {      \
  using AtirLowering<OP##Op>::AtirLowering;                             \
  void Lowering(PatternRewriter &rewriter, OP##OpAdaptor adaptor, OP##Op op) const override;   \
  };

OpLowering(Constant) OpLowering(Add) OpLowering(MatMul) OpLowering(Customize)
#undef OpLowering

struct FusedMatMulLoweringToLinalg
    : public mlir::OpConversionPattern<atir::FusedMatMulOp> {
  using OpConversionPattern<atir::FusedMatMulOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      atir::FusedMatMulOp op, atir::FusedMatMulOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override;
};

struct V2FuncReturnLoweringToLinalg
    : public mlir::OpConversionPattern<mlir::func::ReturnOp> {
  using OpConversionPattern<mlir::func::ReturnOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::func::ReturnOp op, mlir::func::ReturnOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override;
};

struct ContextCustomizeLoweringToLinalg
    : public mlir::OpRewritePattern<atir::CustomizeOp> {
  using OpRewritePattern<atir::CustomizeOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(atir::CustomizeOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

struct ReluLoweringToLinalg : public mlir::OpConversionPattern<atir::ReluOp> {
  using OpConversionPattern<atir::ReluOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      atir::ReluOp op, atir::ReluOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override;
};

struct DotLoweringToLinalg : public mlir::OpConversionPattern<atir::DotOp> {
  using OpConversionPattern<atir::DotOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      atir::DotOp op, atir::DotOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override;
};

struct BufferLoweringToLinalg
    : public mlir::OpConversionPattern<atir::BufferOp> {
  using OpConversionPattern<atir::BufferOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      atir::BufferOp op, atir::BufferOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override;
};

struct ReshapeLoweringToLinalg
    : public mlir::OpConversionPattern<atir::ReshapeOp> {
  using OpConversionPattern<atir::ReshapeOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      atir::ReshapeOp op, atir::ReshapeOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override;
};

bool isSupportedInsertUnitDimensionBeforeN(atir::ReshapeOp op);
}
