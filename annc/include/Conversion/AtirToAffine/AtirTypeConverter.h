#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace atir
{
class AtirTypeToAffineConverter : public mlir::TypeConverter
{
public:
    AtirTypeToAffineConverter();
};
} // namespace atir