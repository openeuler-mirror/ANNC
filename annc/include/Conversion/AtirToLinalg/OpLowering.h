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

OpLowering(None)OpLowering(Constant)OpLowering(Relu)OpLowering(Add)OpLowering(Concat)OpLowering(MatMul)
  OpLowering(Return)OpLowering(Load)
}