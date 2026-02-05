#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace atir
{
class AtirTypeToLinalgConverter : public mlir::TypeConverter
{
public:
    AtirTypeToLinalgConverter();
};
} // namespace atir