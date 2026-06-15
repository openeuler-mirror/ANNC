#include "Conversion/AtirToAffine/AtirTypeConverter.h"
#include "Dialect/Atir/AtirOps.h"

using namespace atir;

namespace {

bool isStringTensor(atir::TensorType tensorType) {
    auto encoding = tensorType.getEncoding();
    auto stringAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(encoding);
    return stringAttr && stringAttr.getValue() == "string";
}

} // namespace

AtirTypeToAffineConverter::AtirTypeToAffineConverter()
{
    addConversion([](Type type) { return type; });
    addConversion([](atir::TensorType tensorType)
    {
        // return RankedTensorType::get(tensorType.getShape(), tensorType.getElementType());
        auto shape = tensorType.getShape();
        auto elemType = isStringTensor(tensorType)
                            ? mlir::IntegerType::get(tensorType.getContext(), 64)
                            : tensorType.getElementType();
        // int64_t offset = ShapedType::kDynamic;
        // SmallVector<int64_t> dynStrides(shape.size(), ShapedType::kDynamic);
        // return MemRefType::get(shape, tensorType.getElementType(), StridedLayoutAttr::get(tensorType.getContext(), offset, dynStrides));
        return MemRefType::get(shape, elemType);
    });
}
