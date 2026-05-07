#include "Conversion/AtirToLinalg/AtirTypeConverter.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "Dialect/Atir/AtirOps.h"

using namespace atir;

AtirTypeToLinalgConverter::AtirTypeToLinalgConverter()
{
    addConversion([](Type type) { return type; });
    addConversion([](TensorType tensorType) {
        return RankedTensorType::get(tensorType.getShape(), tensorType.getElementType());
    });

    addTargetMaterialization(
        [](OpBuilder& builder, mlir::TensorType tensorType, ValueRange inputs, Location loc) {
        return builder.create<bufferization::ToTensorOp>(loc, tensorType, inputs[0], true, true).getResult();
    });
}