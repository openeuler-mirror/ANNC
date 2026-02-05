#include "Conversion/Common/PimpLowering.h"
#include "Dialect/Pimp/PimpOps.h"

namespace pimp
{
void populatePimpToAffineConversionPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns);

#define OpLowering(OP)                                                         \
  struct OP##LoweringToAffine : public PimpLowering<pimp::OP##Op> {      \
  using PimpLowering<OP##Op>::PimpLowering;                             \
  void Lowering(PatternRewriter &rewriter, OP##OpAdaptor adaptor, OP##Op op) const override;   \
  };

OpLowering(None)OpLowering(Constant)OpLowering(Relu)OpLowering(Add)OpLowering(Concat)OpLowering(MatMul)
  OpLowering(Return)OpLowering(Load)
}