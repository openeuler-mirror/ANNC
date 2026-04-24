#include "Conversion/PimpToAffine/PimpTypeConverter.h"
#include "Dialect/Pimp/PimpOps.h"

using namespace pimp;

PimpTypeToAffineConverter::PimpTypeToAffineConverter()
{
    addConversion([](Type type) { return type; });
    addConversion([](TensorType tensorType)
    {
        // return RankedTensorType::get(tensorType.getShape(), tensorType.getElementType());
        auto shape = tensorType.getShape();
        // int64_t offset = ShapedType::kDynamic;
        // SmallVector<int64_t> dynStrides(shape.size(), ShapedType::kDynamic);
        // return MemRefType::get(shape, tensorType.getElementType(), StridedLayoutAttr::get(tensorType.getContext(), offset, dynStrides));
        return MemRefType::get(shape, tensorType.getElementType());
    });
}