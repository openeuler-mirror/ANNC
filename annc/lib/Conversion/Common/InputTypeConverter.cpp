#include "Conversion/Common/InputTypeConverter.h"

#include <mlir/Dialect/Bufferization/IR/Bufferization.h>

#include "Dialect/Atir/AtirOps.h"


using namespace atir;

namespace {

bool isStringTensor(atir::TensorType tensorType) {
    auto encoding = tensorType.getEncoding();
    auto stringAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(encoding);
    return stringAttr && stringAttr.getValue() == "string";
}

} // namespace

InputTypeConverter::InputTypeConverter()
{
    addConversion([](Type type) { return type; });
    addConversion([](atir::TensorType tensorType)
    {
        auto shape = tensorType.getShape();
        auto elemType = isStringTensor(tensorType)
                            ? mlir::IntegerType::get(tensorType.getContext(), 64)
                            : tensorType.getElementType();
        int64_t offset = 0; // todo:
        auto strides = tensorType.getValueOfStride();
        if (strides.empty())
        {
            return MemRefType::get(shape, elemType);
        }
        return MemRefType::get(shape, elemType, StridedLayoutAttr::get(tensorType.getContext(), offset, strides));
    });

    addTargetMaterialization(
    [](OpBuilder& builder, mlir::MemRefType memrefType, ValueRange inputs, Location loc){
        return builder.create<bufferization::ToBufferOp>(loc, memrefType, inputs[0]).getResult();
    });
}
