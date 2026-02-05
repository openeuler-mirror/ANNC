#ifndef INPUTTYPECONVERTER_H
#define INPUTTYPECONVERTER_H
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
namespace pimp
{
class InputTypeConverter : public mlir::TypeConverter
{
public:
    InputTypeConverter();
};
}
#endif //INPUTTYPECONVERTER_H
