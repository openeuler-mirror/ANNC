#include "Conversion/Common/InputTypeConverter.h"

#include <mlir/Dialect/Bufferization/IR/Bufferization.h>

#include "Dialect/Pimp/PimpOps.h"


using namespace pimp;
InputTypeConverter::InputTypeConverter()
{
    addConversion([](Type type) { return type; });
    addConversion([](TensorType tensorType)
    {
        auto shape = tensorType.getShape();
        auto elemType = tensorType.getElementType();
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
