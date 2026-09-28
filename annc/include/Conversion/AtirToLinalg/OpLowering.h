#include "Conversion/Common/AtirLowering.h"
#include "Dialect/Atir/AtirOps.h"

namespace atir
{
void populateAtirToLinalgConversionPatterns(TypeConverter &inputTypeConverter, TypeConverter &atirTypeConverter, RewritePatternSet &patterns);

#define OpLowering(OP)                                                         \
  struct OP##LoweringToLinalg : public AtirLowering<atir::OP##Op> {      \
  using AtirLowering<OP##Op>::AtirLowering;                             \
  void Lowering(PatternRewriter &rewriter, OP##OpAdaptor adaptor, OP##Op op) const override;   \
  };

OpLowering(Constant) OpLowering(Add) OpLowering(MatMul) OpLowering(Customize)
#undef OpLowering

struct ReluLoweringToLinalg : public mlir::OpConversionPattern<atir::ReluOp> {
  using OpConversionPattern<atir::ReluOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      atir::ReluOp op, atir::ReluOp::Adaptor adaptor,
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
