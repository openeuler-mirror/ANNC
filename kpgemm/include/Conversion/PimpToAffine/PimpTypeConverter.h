#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace pimp
{
class PimpTypeToAffineConverter : public mlir::TypeConverter
{
public:
    PimpTypeToAffineConverter();
};
} // namespace pimp