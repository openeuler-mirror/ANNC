#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace pimp
{
class PimpTypeToLinalgConverter : public mlir::TypeConverter
{
public:
    PimpTypeToLinalgConverter();
};
} // namespace pimp