#include "Conversion/AtirToLinalg/AtirTypeConverter.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "Dialect/Atir/AtirOps.h"

using namespace atir;

AtirTypeToLinalgConverter::AtirTypeToLinalgConverter()
{
    addConversion([](Type type) { return type; });
    addConversion([](TensorType tensorType) -> Type {
        if (!tensorType.hasKnownRank() ||
            mlir::isa<atir::UnknownType>(tensorType.getElementType()))
            return Type();
        return RankedTensorType::get(tensorType.getShape(), tensorType.getElementType());
    });
    addConversion([](atir::ResourceType) -> Type { return Type(); });

    addTargetMaterialization(
        [](OpBuilder& builder, mlir::TensorType tensorType, ValueRange inputs, Location loc) {
        return builder.create<bufferization::ToTensorOp>(loc, tensorType, inputs[0], true, true).getResult();
    });
}
