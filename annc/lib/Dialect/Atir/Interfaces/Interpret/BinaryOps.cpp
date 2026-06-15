#include "Common.h"
#include "Dialect/Atir/AtirOps.h"

using namespace atir::interpret;

namespace atir {

void AddOp::Interpret() {
  this->inferShape();

  if (getNumOperands() == 0) {
    emitOpError("add op must have at least one input");
    return;
  }

  SmallVector<atir::TensorType> inputTypes;
  SmallVector<std::vector<float>> inputValues;
  for (size_t i = 1; i < getNumOperands(); ++i) {
    Value operand = getOperand(i);
    if (auto noneOp = dyn_cast_or_null<atir::NoneOp>(operand.getDefiningOp())) continue;

    atir::TensorType tensorType;
    DenseElementsAttr attr;
    if (failed(getTensorTypeAndData(getOperation(), operand, "Add input",
                                    tensorType, attr))) {
      return;
    }

    auto valuesOr = getFloatValues(attr);
    if (failed(valuesOr)) {
      emitOpError("Add only supports numeric input cacheData");
      return;
    }

    inputTypes.push_back(tensorType);
    inputValues.push_back(std::move(*valuesOr));
  }

  if (inputTypes.empty()) {
    emitOpError("Add has no materialized inputs");
    return;
  }

  auto resultType = this->getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outSize = getElementCount(outputShape);
  if (outSize < 0) {
    emitOpError("Add output shape must be static for interpretation");
    return;
  }

  // 获取addop的属性值
  float scalar = 0.0f;
  if (auto s = dyn_cast_or_null<FloatAttr>(getScalarAttr()))
    scalar = annc::apFloatToFloat(s.getValue());
  bool doRelu = getDoRelu();
  float reluLimit =
      annc::apFloatToFloat(cast<FloatAttr>(getReluLimitAttr()).getValue());

  // 初始化 result 为零
  std::vector<float> result(outSize, 0.0f);
  
  for (int64_t outIdx = 0; outIdx < outSize; ++outIdx) {
    auto outputIndex = getMultiIndex(outputShape, outIdx);
    float sum = result[outIdx] + scalar;

    // 遍历所有输入张量并进行加法运算
    for (size_t operandIdx = 0; operandIdx < inputTypes.size(); ++operandIdx) {
      auto inputShape = inputTypes[operandIdx].getShape();
      auto inputIndex =
          getBroadcastIndex(outputShape, inputShape, outputIndex);
      sum += inputValues[operandIdx][getFlatIndex(inputShape, inputIndex)];
    }

    result[outIdx] = applyReluLimit(sum, doRelu, reluLimit);
  }

  (void)setDenseResult(resultType, outputShape, result);
}

void SubOp::Interpret() {
  this->inferShape();

  atir::TensorType lhsType;
  atir::TensorType rhsType;
  DenseElementsAttr lhsAttr;
  DenseElementsAttr rhsAttr;
  if (failed(getTensorTypeAndData(getOperation(), getX(), "Sub lhs", lhsType,
                                  lhsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getY(), "Sub rhs", rhsType,
                                  rhsAttr))) {
    return;
  }

  auto lhsValsOr = getFloatValues(lhsAttr);
  auto rhsValsOr = getFloatValues(rhsAttr);
  if (failed(lhsValsOr) || failed(rhsValsOr)) {
    emitOpError("Sub only supports numeric input cacheData");
    return;
  }

  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("Sub output shape must be static for interpretation");
    return;
  }

  std::vector<float> result(outputSize, 0.0f);
  
  for (int64_t outIdx = 0; outIdx < outputSize; ++outIdx) {
    auto outputIndex = getMultiIndex(outputShape, outIdx);
    auto lhsIndex = getBroadcastIndex(outputShape, lhsType.getShape(), outputIndex);
    auto rhsIndex = getBroadcastIndex(outputShape, rhsType.getShape(), outputIndex);
    float lhsValue = (*lhsValsOr)[getFlatIndex(lhsType.getShape(), lhsIndex)];
    float rhsValue = (*rhsValsOr)[getFlatIndex(rhsType.getShape(), rhsIndex)];
    result[outIdx] = lhsValue - rhsValue;
  }

  (void)setDenseResult(resultType, outputShape, result);
}

void MulOp::Interpret() {
  this->inferShape();

  atir::TensorType lhsType;
  atir::TensorType rhsType;
  DenseElementsAttr lhsAttr;
  DenseElementsAttr rhsAttr;
  if (failed(getTensorTypeAndData(getOperation(), getX(), "Mul lhs", lhsType,
                                  lhsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getY(), "Mul rhs", rhsType,
                                  rhsAttr))) {
    return;
  }

  auto lhsValsOr = getFloatValues(lhsAttr);
  auto rhsValsOr = getFloatValues(rhsAttr);
  if (failed(lhsValsOr) || failed(rhsValsOr)) {
    emitOpError("Mul only supports numeric input cacheData");
    return;
  }

  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("Mul output shape must be static for interpretation");
    return;
  }

  std::vector<float> result(outputSize, 0.0f);
  
  for (int64_t outIdx = 0; outIdx < outputSize; ++outIdx) {
    auto outputIndex = getMultiIndex(outputShape, outIdx);
    auto lhsIndex = getBroadcastIndex(outputShape, lhsType.getShape(), outputIndex);
    auto rhsIndex = getBroadcastIndex(outputShape, rhsType.getShape(), outputIndex);
    float lhsValue = (*lhsValsOr)[getFlatIndex(lhsType.getShape(), lhsIndex)];
    float rhsValue = (*rhsValsOr)[getFlatIndex(rhsType.getShape(), rhsIndex)];
    result[outIdx] = lhsValue * rhsValue;
  }

  (void)setDenseResult(resultType, outputShape, result);
}

void RealDivOp::Interpret() {
  this->inferShape();

  atir::TensorType lhsType;
  atir::TensorType rhsType;
  DenseElementsAttr lhsAttr;
  DenseElementsAttr rhsAttr;
  if (failed(getTensorTypeAndData(getOperation(), getX(), "RealDiv lhs",
                                  lhsType, lhsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getY(), "RealDiv rhs",
                                  rhsType, rhsAttr))) {
    return;
  }

  auto lhsValsOr = getFloatValues(lhsAttr);
  auto rhsValsOr = getFloatValues(rhsAttr);
  if (failed(lhsValsOr) || failed(rhsValsOr)) {
    emitOpError("RealDiv only supports numeric input cacheData");
    return;
  }

  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("RealDiv output shape must be static for interpretation");
    return;
  }
  
  std::vector<float> result(outputSize, 0.0f);
  
  for (int64_t outIdx = 0; outIdx < outputSize; ++outIdx) {
    auto outputIndex = getMultiIndex(outputShape, outIdx);
    auto lhsIndex = getBroadcastIndex(outputShape, lhsType.getShape(), outputIndex);
    auto rhsIndex = getBroadcastIndex(outputShape, rhsType.getShape(), outputIndex);
    float lhsValue = (*lhsValsOr)[getFlatIndex(lhsType.getShape(), lhsIndex)];
    float rhsValue = (*rhsValsOr)[getFlatIndex(rhsType.getShape(), rhsIndex)];
    result[outIdx] = lhsValue / rhsValue;
  }

  (void)setDenseResult(resultType, outputShape, result);
}

void FloorModOp::Interpret() {
  this->inferShape();

  atir::TensorType lhsType;
  atir::TensorType rhsType;
  DenseElementsAttr lhsAttr;
  DenseElementsAttr rhsAttr;
  if (failed(getTensorTypeAndData(getOperation(), getX(), "FloorMod lhs",
                                  lhsType, lhsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getY(), "FloorMod rhs",
                                  rhsType, rhsAttr))) {
    return;
  }

  auto lhsValsOr = getFloatValues(lhsAttr);
  auto rhsValsOr = getFloatValues(rhsAttr);
  if (failed(lhsValsOr) || failed(rhsValsOr)) {
    emitOpError("FloorMod only supports numeric input cacheData");
    return;
  }

  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("FloorMod output shape must be static for interpretation");
    return;
  }

  std::vector<float> result(outputSize, 0.0f);
  for (int64_t outIdx = 0; outIdx < outputSize; ++outIdx) {
    auto outputIndex = getMultiIndex(outputShape, outIdx);
    auto lhsIndex = getBroadcastIndex(outputShape, lhsType.getShape(), outputIndex);
    auto rhsIndex = getBroadcastIndex(outputShape, rhsType.getShape(), outputIndex);
    float lhsValue = (*lhsValsOr)[getFlatIndex(lhsType.getShape(), lhsIndex)];
    float rhsValue = (*rhsValsOr)[getFlatIndex(rhsType.getShape(), rhsIndex)];
    result[outIdx] = std::fmod(std::fmod(lhsValue, rhsValue) + rhsValue, rhsValue);
  }

  (void)setDenseResult(resultType, outputShape, result);
}

void FloorDivOp::Interpret() {
  this->inferShape();

  atir::TensorType lhsType;
  atir::TensorType rhsType;
  DenseElementsAttr lhsAttr;
  DenseElementsAttr rhsAttr;
  if (failed(getTensorTypeAndData(getOperation(), getX(), "FloorDiv lhs",
                                  lhsType, lhsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getY(), "FloorDiv rhs",
                                  rhsType, rhsAttr))) {
    return;
  }

  auto lhsValsOr = getFloatValues(lhsAttr);
  auto rhsValsOr = getFloatValues(rhsAttr);
  if (failed(lhsValsOr) || failed(rhsValsOr)) {
    emitOpError("FloorDiv only supports numeric input cacheData");
    return;
  }

  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("FloorDiv output shape must be static for interpretation");
    return;
  }

  std::vector<float> result(outputSize, 0.0f);
  for (int64_t outIdx = 0; outIdx < outputSize; ++outIdx) {
    auto outputIndex = getMultiIndex(outputShape, outIdx);
    auto lhsIndex = getBroadcastIndex(outputShape, lhsType.getShape(), outputIndex);
    auto rhsIndex = getBroadcastIndex(outputShape, rhsType.getShape(), outputIndex);
    float lhsValue = (*lhsValsOr)[getFlatIndex(lhsType.getShape(), lhsIndex)];
    float rhsValue = (*rhsValsOr)[getFlatIndex(rhsType.getShape(), rhsIndex)];
    result[outIdx] = std::floor(lhsValue / rhsValue);
  }

  (void)setDenseResult(resultType, outputShape, result);
}

void DivideOp::Interpret() {
  this->inferShape();

  atir::TensorType lhsType;
  atir::TensorType rhsType;
  DenseElementsAttr lhsAttr;
  DenseElementsAttr rhsAttr;
  if (failed(getTensorTypeAndData(getOperation(), getLhs(), "Divide lhs",
                                  lhsType, lhsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getRhs(), "Divide rhs",
                                  rhsType, rhsAttr))) {
    return;
  }

  auto lhsValsOr = getFloatValues(lhsAttr);
  auto rhsValsOr = getFloatValues(rhsAttr);
  if (failed(lhsValsOr) || failed(rhsValsOr)) {
    emitOpError("Divide only supports numeric input cacheData");
    return;
  }

  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("Divide output shape must be static for interpretation");
    return;
  }

  std::vector<float> result(outputSize, 0.0f);
  for (int64_t outIdx = 0; outIdx < outputSize; ++outIdx) {
    auto outputIndex = getMultiIndex(outputShape, outIdx);
    auto lhsIndex =
        getBroadcastIndex(outputShape, lhsType.getShape(), outputIndex);
    auto rhsIndex =
        getBroadcastIndex(outputShape, rhsType.getShape(), outputIndex);
    float lhsValue = (*lhsValsOr)[getFlatIndex(lhsType.getShape(), lhsIndex)];
    float rhsValue = (*rhsValsOr)[getFlatIndex(rhsType.getShape(), rhsIndex)];
    result[outIdx] = lhsValue / rhsValue;
  }

  (void)setDenseResult(resultType, outputShape, result);
}

} // namespace atir
