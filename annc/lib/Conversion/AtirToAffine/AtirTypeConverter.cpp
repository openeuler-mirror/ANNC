#include "Conversion/AtirToAffine/AtirTypeConverter.h"
#include "Dialect/Atir/AtirOps.h"

using namespace atir;

AtirTypeToAffineConverter::AtirTypeToAffineConverter()
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