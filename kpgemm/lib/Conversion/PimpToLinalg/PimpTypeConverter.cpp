#include "Conversion/PimpToLinalg/PimpTypeConverter.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "Dialect/Pimp/PimpOps.h"

using namespace pimp;

PimpTypeToLinalgConverter::PimpTypeToLinalgConverter()
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