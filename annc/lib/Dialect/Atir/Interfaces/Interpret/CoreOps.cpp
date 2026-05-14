#include "Common.h"
#include "Dialect/Atir/AtirOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace atir::interpret;

namespace atir {

void BufferOp::Interpret() {
  auto resultType = dyn_cast<atir::TensorType>(getResult().getType());
  if (!resultType) {
    emitOpError("buffer result must be atir::TensorType");
    return;
  }
  if (resultType.getCacheData()) {
    return;
  }
  auto shape = resultType.getShape();
  int64_t numElements = getElementCount(shape);
  if (numElements < 0) {
    return;
  }
  auto zeroAttr = getZeroElementAttr(resultType.getElementType(), getContext());
  if (!zeroAttr) {
    return;
  }
  auto standardTensorType =
      RankedTensorType::get(shape, resultType.getElementType());
  resultType.setCacheData(DenseElementsAttr::get(standardTensorType, zeroAttr));
}

void CustomizeOp::Interpret() {
  for (Value result : getResults()) {
    auto tensorType = dyn_cast<atir::TensorType>(result.getType());
    if (!tensorType) {
      emitOpError("Customize result must be atir::TensorType");
      return;
    }
    if (!tensorType.getCacheData()) {
      emitOpError(
          "CustomizeOp interpretation is unsupported without result cacheData");
      return;
    }
  }
}

void ConstantOp::Interpret() {
  auto resultType = dyn_cast<atir::TensorType>(getResult().getType());
  if (!resultType) {
    emitOpError("constant result must be atir::TensorType");
    return;
  }
  auto data = resultType.getCacheData();
  if (data) {
    return;
  }
  int64_t numElements = getElementCount(resultType.getShape());
  if (numElements != 0) {
    emitError("current constant op data is missing");
    return;
  }
  auto rankedType = RankedTensorType::get(resultType.getShape(), resultType.getElementType());
  auto elementType = resultType.getElementType();
  if (elementType.isF32()) {
    resultType.setCacheData(DenseElementsAttr::get(rankedType, ArrayRef<float>({})));
    return;
  }
  if (elementType.isF64()) {
    resultType.setCacheData(DenseElementsAttr::get(rankedType, ArrayRef<double>({})));
    return;
  }
  if (elementType.isInteger(32)) {
    resultType.setCacheData(DenseElementsAttr::get(rankedType, ArrayRef<int32_t>({})));
    return;
  }
  if (elementType.isInteger(64) || elementType.isIndex()) {
    resultType.setCacheData(DenseElementsAttr::get(rankedType, ArrayRef<int64_t>({})));
    return;
  }

  emitOpError("constant result element type is not supported");
}

void VariableOp::Interpret() {
  auto resultType = dyn_cast<atir::TensorType>(getOutput().getType());
  if (!resultType) {
    emitOpError("variable result must be atir::TensorType");
    return;
  }

  if (!resultType.getCacheData()) emitOpError("variable result is missing cacheData");
}

void ReturnOp::Interpret() {
  for (Value operand : getOperands()) {
    auto tensorType = dyn_cast<atir::TensorType>(operand.getType());
    if (!tensorType) continue;
    if (!tensorType.getCacheData()) {
      emitOpError("return operand is missing cacheData");
      return;
    }
  }
}

void IdentityOp::Interpret() {
  this->inferShape();
  atir::TensorType inputTensor;
  DenseElementsAttr inputAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(), "Identity input",
                                  inputTensor, inputAttr))) {
    return;
  }
  auto resultType = dyn_cast<atir::TensorType>(getResult().getType());
  if (!resultType) {
    emitOpError("Identity result must be atir::TensorType");
    return;
  }
  resultType.setCacheData(inputAttr);
}

void ShapeOp::Interpret() {
  this->inferShape();
  auto inputType = dyn_cast<atir::TensorType>(getInput().getType());
  auto resultType = dyn_cast<atir::TensorType>(getOutput().getType());
  if (!inputType || !resultType) {
    emitOpError("Shape expects tensor input and output");
    return;
  }
  SmallVector<int64_t> shapeValues(inputType.getShape().begin(), inputType.getShape().end());
  auto outputShape = resultType.getShape();
  if (failed(setDenseIntResult(resultType, outputShape, shapeValues))) {
    emitOpError("Shape output element type must be integer or index");
  }
}

void SizeOp::Interpret() {
  this->inferShape();
  atir::TensorType inputType;
  DenseElementsAttr inputAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(), "Size input",
                                  inputType, inputAttr)))  return;
  int64_t numElements = getElementCount(inputType.getShape());
  if (numElements < 0) {
    emitOpError("Size input shape must be static for interpretation");
    return;
  }
  auto resultType = dyn_cast<atir::TensorType>(getOutput().getType());
  if (!resultType) {
    emitOpError("Size expects tensor output");
    return;
  }
  SmallVector<int64_t> outputShape;
  std::vector<int64_t> value = {numElements};
  if (failed(setDenseIntResult(resultType, outputShape, value))) {
    emitOpError("Size output element type must be integer or index");
  }
}

void FillOp::Interpret() {
  inferShape();
  atir::TensorType shapeTy;
  atir::TensorType valueTy;
  DenseElementsAttr shapeAttr;
  DenseElementsAttr valueAttr;
  if (failed(getTensorTypeAndData(getOperation(), getShapeInput(),
                                  "Fill shapeInput", shapeTy, shapeAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getValueInput(),
                                  "Fill valueInput", valueTy, valueAttr))) {
    return;
  }

  auto dimsOr = getIntValues(shapeAttr);
  if (failed(dimsOr)) {
    emitOpError("Fill shapeInput must have numeric cacheData");
    return;
  }
  auto resultType = getResult().getType();
  llvm::SmallVector<int64_t> outputShape(dimsOr->begin(), dimsOr->end());
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("Fill output shape must be static for interpretation");
    return;
  }

  Type elemTy = resultType.getElementType();
  if (isa<IntegerType>(elemTy)) {
    auto intValOr = getIntValues(valueAttr);
    if (failed(intValOr) || intValOr->size() != 1) {
      emitOpError("Fill valueInput must be a scalar integer for integer output");
      return;
    }
    std::vector<int64_t> filled(outputSize, (*intValOr)[0]);
    if (failed(setDenseIntResult(resultType, outputShape, filled))) {
      emitOpError("Fill integer output element type is not supported");
    }
    return;
  }

  auto floatValOr = getFloatValues(valueAttr);
  if (failed(floatValOr) || floatValOr->size() != 1) {
    emitOpError("Fill valueInput must be a scalar");
    return;
  }
  std::vector<float> filled(outputSize, (*floatValOr)[0]);
  (void)setDenseResult(resultType, outputShape, filled);
}

void RangeOp::Interpret() {
  inferShape();
  atir::TensorType startTy;
  atir::TensorType limitTy;
  atir::TensorType deltaTy;
  DenseElementsAttr startAttr;
  DenseElementsAttr limitAttr;
  DenseElementsAttr deltaAttr;
  if (failed(getTensorTypeAndData(getOperation(), getStart(), "Range start",
                                  startTy, startAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getLimit(), "Range limit",
                                  limitTy, limitAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getDelta(), "Range delta",
                                  deltaTy, deltaAttr))) {
    return;
  }
  auto resultType = getResult().getType();
  auto outShape = resultType.getShape();
  if (outShape.size() != 1) {
    emitOpError("Range result must be rank-1");
    return;
  }
  Type elemTy = resultType.getElementType();
  if (isa<IntegerType>(elemTy)) {
    auto sOr = getIntValues(startAttr);
    auto lOr = getIntValues(limitAttr);
    auto dOr = getIntValues(deltaAttr);
    if (failed(sOr) || failed(lOr) || failed(dOr) || sOr->size() != 1 ||
        lOr->size() != 1 || dOr->size() != 1) {
      emitOpError("Range integer form expects scalar int start/limit/delta");
      return;
    }
    int64_t start = (*sOr)[0];
    int64_t limit = (*lOr)[0];
    int64_t delta = (*dOr)[0];
    if (delta == 0) {
      emitOpError("Range delta must be non-zero");
      return;
    }
    llvm::SmallVector<int64_t> values;
    if (delta > 0) {
      for (int64_t cur = start; cur < limit; cur += delta) {
        values.push_back(cur);
      }
    } else {
      for (int64_t cur = start; cur > limit; cur += delta) {
        values.push_back(cur);
      }
    }
    llvm::SmallVector<int64_t> staticShape{static_cast<int64_t>(values.size())};
    if (failed(setDenseIntResult(resultType, staticShape, values))) {
      emitOpError("Range integer output element type is not supported");
    }
    return;
  }
  auto sF = getFloatValues(startAttr);
  auto lF = getFloatValues(limitAttr);
  auto dF = getFloatValues(deltaAttr);
  if (failed(sF) || failed(lF) || failed(dF) || sF->size() != 1 ||
      lF->size() != 1 || dF->size() != 1) {
    emitOpError("Range floating form expects scalar start/limit/delta");
    return;
  }
  double start = static_cast<double>((*sF)[0]);
  double limit = static_cast<double>((*lF)[0]);
  double delta = static_cast<double>((*dF)[0]);
  if (delta == 0.0) {
    emitOpError("Range delta must be non-zero");
    return;
  }
  std::vector<float> values;
  if (delta > 0) {
    for (double cur = start; cur < limit; cur += delta) {
      values.push_back(static_cast<float>(cur));
    }
  } else {
    for (double cur = start; cur > limit; cur += delta) {
      values.push_back(static_cast<float>(cur));
    }
  }
  llvm::SmallVector<int64_t> staticShape{static_cast<int64_t>(values.size())};
  (void)setDenseResult(resultType, staticShape, values);
}

void ParameterOp::Interpret() {
  auto outTy = dyn_cast<atir::TensorType>(getOutput().getType());
  if (!outTy) {
    emitOpError("Parameter output must be atir::TensorType");
    return;
  }
  // If already materialized (cacheData present), nothing to do.
  if (outTy.getCacheData()) return;

  // Otherwise, try to copy cacheData from parent function arguments by
  // parameter_number (assumes parameter_number aligns with function-arg index).
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    emitOpError("ParameterOp missing cacheData and cannot find parent func");
    return;
  }

  int64_t pnum = getParameterNumber();
  if (pnum < 0 || pnum >= funcOp.getNumArguments()) {
    emitOpError("ParameterOp parameter_number is out of function-argument range");
    return;
  }

  auto argTy = dyn_cast<atir::TensorType>(funcOp.getArgument(static_cast<size_t>(pnum)).getType());
  if (!argTy || !argTy.getCacheData()) {
    emitOpError("ParameterOp cacheData source argument missing");
    return;
  }

  outTy.setCacheData(argTy.getCacheData());
  getOutput().setType(outTy);
}

void ScatterOp::Interpret() {}

}  // namespace atir
