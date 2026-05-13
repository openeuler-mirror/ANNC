#include "Common.h"
#include "Dialect/Atir/AtirOps.h"

using namespace atir::interpret;

namespace atir {

void NotEqualOp::Interpret() {
  this->inferShape();
  atir::TensorType lhsType;
  atir::TensorType rhsType;
  DenseElementsAttr lhsAttr;
  DenseElementsAttr rhsAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput1(), "NotEqual lhs",
                                  lhsType, lhsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getInput2(), "NotEqual rhs",
                                  rhsType, rhsAttr))) {
    return;
  }

  auto lhsValsOr = getFloatValues(lhsAttr);
  auto rhsValsOr = getFloatValues(rhsAttr);
  if (failed(lhsValsOr) || failed(rhsValsOr)) {
    emitOpError("NotEqual only supports numeric input cacheData");
    return;
  }

  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("NotEqual output shape must be static for interpretation");
    return;
  }

  std::vector<int64_t> result(outputSize, 0);
  for (int64_t outIdx = 0; outIdx < outputSize; ++outIdx) {
    auto outputIndex = getMultiIndex(outputShape, outIdx);
    auto lhsIndex = getBroadcastIndex(outputShape, lhsType.getShape(), outputIndex);
    auto rhsIndex = getBroadcastIndex(outputShape, rhsType.getShape(), outputIndex);
    float lhsValue = (*lhsValsOr)[getFlatIndex(lhsType.getShape(), lhsIndex)];
    float rhsValue = (*rhsValsOr)[getFlatIndex(rhsType.getShape(), rhsIndex)];
    result[outIdx] = lhsValue != rhsValue ? 1 : 0;
  }

  if (failed(setBooleanLikeResult(resultType, outputShape, result))) {
    emitOpError("NotEqual output element type is not supported");
  }
}

void LessOp::Interpret() {
  this->inferShape();
  atir::TensorType lhsType;
  atir::TensorType rhsType;
  DenseElementsAttr lhsAttr;
  DenseElementsAttr rhsAttr;
  if (failed(getTensorTypeAndData(getOperation(), getLhs(), "Less lhs", lhsType,
                                  lhsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getRhs(), "Less rhs", rhsType,
                                  rhsAttr))) {
    return;
  }

  auto lhsValsOr = getFloatValues(lhsAttr);
  auto rhsValsOr = getFloatValues(rhsAttr);
  if (failed(lhsValsOr) || failed(rhsValsOr)) {
    emitOpError("Less only supports numeric input cacheData");
    return;
  }

  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("Less output shape must be static for interpretation");
    return;
  }

  std::vector<int64_t> result(outputSize, 0);
  for (int64_t outIdx = 0; outIdx < outputSize; ++outIdx) {
    auto outputIndex = getMultiIndex(outputShape, outIdx);
    auto lhsIndex = getBroadcastIndex(outputShape, lhsType.getShape(), outputIndex);
    auto rhsIndex = getBroadcastIndex(outputShape, rhsType.getShape(), outputIndex);
    float lhsValue = (*lhsValsOr)[getFlatIndex(lhsType.getShape(), lhsIndex)];
    float rhsValue = (*rhsValsOr)[getFlatIndex(rhsType.getShape(), rhsIndex)];
    result[outIdx] = lhsValue < rhsValue ? 1 : 0;
  }

  if (failed(setBooleanLikeResult(resultType, outputShape, result))) {
    emitOpError("Less output element type is not supported");
  }
}

void GreaterEqualOp::Interpret() {
  this->inferShape();
  atir::TensorType lhsType;
  atir::TensorType rhsType;
  DenseElementsAttr lhsAttr;
  DenseElementsAttr rhsAttr;
  if (failed(getTensorTypeAndData(getOperation(), getLhs(),
                                  "GreaterEqual lhs", lhsType, lhsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getRhs(),
                                  "GreaterEqual rhs", rhsType, rhsAttr))) {
    return;
  }

  auto lhsValsOr = getFloatValues(lhsAttr);
  auto rhsValsOr = getFloatValues(rhsAttr);
  if (failed(lhsValsOr) || failed(rhsValsOr)) {
    emitOpError("GreaterEqual only supports numeric input cacheData");
    return;
  }

  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError(
        "GreaterEqual output shape must be static for interpretation");
    return;
  }

  std::vector<int64_t> result(outputSize, 0);
  for (int64_t outIdx = 0; outIdx < outputSize; ++outIdx) {
    auto outputIndex = getMultiIndex(outputShape, outIdx);
    auto lhsIndex = getBroadcastIndex(outputShape, lhsType.getShape(), outputIndex);
    auto rhsIndex = getBroadcastIndex(outputShape, rhsType.getShape(), outputIndex);
    float lhsValue = (*lhsValsOr)[getFlatIndex(lhsType.getShape(), lhsIndex)];
    float rhsValue = (*rhsValsOr)[getFlatIndex(rhsType.getShape(), rhsIndex)];
    result[outIdx] = lhsValue >= rhsValue ? 1 : 0;
  }

  if (failed(setBooleanLikeResult(resultType, outputShape, result))) {
    emitOpError("GreaterEqual output element type is not supported");
  }
}

void GreaterOp::Interpret() {
  this->inferShape();
  atir::TensorType lhsType;
  atir::TensorType rhsType;
  DenseElementsAttr lhsAttr;
  DenseElementsAttr rhsAttr;
  if (failed(getTensorTypeAndData(getOperation(), getLhs(), "Greater lhs",
                                  lhsType, lhsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getRhs(), "Greater rhs",
                                  rhsType, rhsAttr))) {
    return;
  }

  auto lhsValsOr = getFloatValues(lhsAttr);
  auto rhsValsOr = getFloatValues(rhsAttr);
  if (failed(lhsValsOr) || failed(rhsValsOr)) {
    emitOpError("Greater only supports numeric input cacheData");
    return;
  }

  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("Greater output shape must be static for interpretation");
    return;
  }

  std::vector<int64_t> result(outputSize, 0);
  for (int64_t outIdx = 0; outIdx < outputSize; ++outIdx) {
    auto outputIndex = getMultiIndex(outputShape, outIdx);
    auto lhsIndex = getBroadcastIndex(outputShape, lhsType.getShape(), outputIndex);
    auto rhsIndex = getBroadcastIndex(outputShape, rhsType.getShape(), outputIndex);
    float lhsValue = (*lhsValsOr)[getFlatIndex(lhsType.getShape(), lhsIndex)];
    float rhsValue = (*rhsValsOr)[getFlatIndex(rhsType.getShape(), rhsIndex)];
    result[outIdx] = lhsValue > rhsValue ? 1 : 0;
  }

  if (failed(setBooleanLikeResult(resultType, outputShape, result))) {
    emitOpError("Greater output element type is not supported");
  }
}

void CompareOp::Interpret() {
  this->inferShape();
  atir::TensorType lhsType;
  atir::TensorType rhsType;
  DenseElementsAttr lhsAttr;
  DenseElementsAttr rhsAttr;
  if (failed(getTensorTypeAndData(getOperation(), getLhs(), "Compare lhs",
                                  lhsType, lhsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getRhs(), "Compare rhs",
                                  rhsType, rhsAttr))) {
    return;
  }

  auto lhsValsOr = getFloatValues(lhsAttr);
  auto rhsValsOr = getFloatValues(rhsAttr);
  if (failed(lhsValsOr) || failed(rhsValsOr)) {
    emitOpError("Compare only supports numeric input cacheData");
    return;
  }

  StringRef dir = getComparisonDirection();
  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("Compare output shape must be static for interpretation");
    return;
  }

  std::vector<int64_t> result(outputSize, 0);
  for (int64_t outIdx = 0; outIdx < outputSize; ++outIdx) {
    auto outputIndex = getMultiIndex(outputShape, outIdx);
    auto lhsIndex =
        getBroadcastIndex(outputShape, lhsType.getShape(), outputIndex);
    auto rhsIndex =
        getBroadcastIndex(outputShape, rhsType.getShape(), outputIndex);
    float lhsValue = (*lhsValsOr)[getFlatIndex(lhsType.getShape(), lhsIndex)];
    float rhsValue = (*rhsValsOr)[getFlatIndex(rhsType.getShape(), rhsIndex)];

    bool pred = false;
    if (dir == "EQ") pred = (lhsValue == rhsValue);
    else if (dir == "NE") pred = (lhsValue != rhsValue);
    else if (dir == "LT") pred = (lhsValue < rhsValue);
    else if (dir == "LE") pred = (lhsValue <= rhsValue);
    else if (dir == "GT") pred = (lhsValue > rhsValue);
    else if (dir == "GE") pred = (lhsValue >= rhsValue);
    else {
      emitOpError("Compare comparisonDirection must be one of EQ/NE/LT/LE/GT/GE");
      return;
    }
    result[outIdx] = pred ? 1 : 0;
  }

  if (failed(setBooleanLikeResult(resultType, outputShape, result))) {
    emitOpError("Compare output element type is not supported");
  }
}

void AndOp::Interpret() {
  this->inferShape();

  atir::TensorType lhsType;
  atir::TensorType rhsType;
  DenseElementsAttr lhsAttr;
  DenseElementsAttr rhsAttr;
  if (failed(getTensorTypeAndData(getOperation(), getLhs(), "And lhs", lhsType,
                                  lhsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getRhs(), "And rhs", rhsType,
                                  rhsAttr))) {
    return;
  }
  
  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("And output shape must be static for interpretation");
    return;
  }

  auto lhsInts = getIntValues(lhsAttr);
  auto rhsInts = getIntValues(rhsAttr);
  if (succeeded(lhsInts) && succeeded(rhsInts)) {
    std::vector<int64_t> result(outputSize, 0);
    for (int64_t outIdx = 0; outIdx < outputSize; ++outIdx) {
      auto outputIndex = getMultiIndex(outputShape, outIdx);
      auto lhsIndex =
          getBroadcastIndex(outputShape, lhsType.getShape(), outputIndex);
      auto rhsIndex =
          getBroadcastIndex(outputShape, rhsType.getShape(), outputIndex);
      int64_t lv = (*lhsInts)[getFlatIndex(lhsType.getShape(), lhsIndex)];
      int64_t rv = (*rhsInts)[getFlatIndex(rhsType.getShape(), rhsIndex)];
      result[outIdx] = lv & rv;
    }
    if (failed(setDenseIntResult(resultType, outputShape, result))) {
      emitOpError("And integer output element type is not supported");
    }
    return;
  }

  auto lhsValsOr = getFloatValues(lhsAttr);
  auto rhsValsOr = getFloatValues(rhsAttr);
  if (failed(lhsValsOr) || failed(rhsValsOr)) {
    emitOpError("And supports integer or float input cacheData");
    return;
  }

  std::vector<float> result(outputSize, 0.0f);
  for (int64_t outIdx = 0; outIdx < outputSize; ++outIdx) {
    auto outputIndex = getMultiIndex(outputShape, outIdx);
    auto lhsIndex =
        getBroadcastIndex(outputShape, lhsType.getShape(), outputIndex);
    auto rhsIndex =
        getBroadcastIndex(outputShape, rhsType.getShape(), outputIndex);
    float lv = (*lhsValsOr)[getFlatIndex(lhsType.getShape(), lhsIndex)];
    float rv = (*rhsValsOr)[getFlatIndex(rhsType.getShape(), rhsIndex)];
    result[outIdx] = (lv != 0.0f && rv != 0.0f) ? 1.0f : 0.0f;
  }
  (void)setDenseResult(resultType, outputShape, result);
}

void MinimumOp::Interpret() {
  this->inferShape();

  atir::TensorType outputTensorType;
  DenseElementsAttr outputAttr;
  if (failed(getTensorTypeAndData(getOperation(), getOperand(0), "Minimum output",
                                   outputTensorType, outputAttr))) {
    return;
  }

  atir::TensorType lhsType;
  atir::TensorType rhsType;
  DenseElementsAttr lhsAttr;
  DenseElementsAttr rhsAttr;
  if (failed(getTensorTypeAndData(getOperation(), getX(), "Minimum lhs",
                                  lhsType, lhsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getY(), "Minimum rhs",
                                  rhsType, rhsAttr))) {
    return;
  }

  auto lhsValsOr = getFloatValues(lhsAttr);
  auto rhsValsOr = getFloatValues(rhsAttr);
  if (failed(lhsValsOr) || failed(rhsValsOr)) {
    emitOpError("Minimum only supports numeric input cacheData");
    return;
  }

  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("Minimum output shape must be static for interpretation");
    return;
  }

  std::vector<float> result;
  auto outputValsOr = getFloatValues(outputAttr);
  if (succeeded(outputValsOr)) {
    result = *outputValsOr;
  } else {
    result.assign(outputSize, 0.0f);
  }
  
  for (int64_t outIdx = 0; outIdx < outputSize; ++outIdx) {
    auto outputIndex = getMultiIndex(outputShape, outIdx);
    auto lhsIndex = getBroadcastIndex(outputShape, lhsType.getShape(), outputIndex);
    auto rhsIndex = getBroadcastIndex(outputShape, rhsType.getShape(), outputIndex);
    float lhsValue = (*lhsValsOr)[getFlatIndex(lhsType.getShape(), lhsIndex)];
    float rhsValue = (*rhsValsOr)[getFlatIndex(rhsType.getShape(), rhsIndex)];
    result[outIdx] = result[outIdx] + std::min(lhsValue, rhsValue);
  }

  (void)setDenseResult(resultType, outputShape, result);
}

void MaximumOp::Interpret() {
  this->inferShape();

  atir::TensorType outputTensorType;
  DenseElementsAttr outputAttr;
  if (failed(getTensorTypeAndData(getOperation(), getOutput(), "Maximum output",
                                   outputTensorType, outputAttr))) {
    return;
  }

  atir::TensorType lhsType;
  atir::TensorType rhsType;
  DenseElementsAttr lhsAttr;
  DenseElementsAttr rhsAttr;
  if (failed(getTensorTypeAndData(getOperation(), getLhs(), "Maximum lhs",
                                  lhsType, lhsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getRhs(), "Maximum rhs",
                                  rhsType, rhsAttr))) {
    return;
  }

  auto lhsValsOr = getFloatValues(lhsAttr);
  auto rhsValsOr = getFloatValues(rhsAttr);
  if (failed(lhsValsOr) || failed(rhsValsOr)) {
    emitOpError("Maximum only supports numeric input cacheData");
    return;
  }

  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("Maximum output shape must be static for interpretation");
    return;
  }

  std::vector<float> result;
  auto outputValsOr = getFloatValues(outputAttr);
  if (succeeded(outputValsOr)) {
    result = *outputValsOr;
  } else {
    result.assign(outputSize, 0.0f);
  }
  
  for (int64_t outIdx = 0; outIdx < outputSize; ++outIdx) {
    auto outputIndex = getMultiIndex(outputShape, outIdx);
    auto lhsIndex = getBroadcastIndex(outputShape, lhsType.getShape(), outputIndex);
    auto rhsIndex = getBroadcastIndex(outputShape, rhsType.getShape(), outputIndex);
    float lhsValue = (*lhsValsOr)[getFlatIndex(lhsType.getShape(), lhsIndex)];
    float rhsValue = (*rhsValsOr)[getFlatIndex(rhsType.getShape(), rhsIndex)];
    result[outIdx] = result[outIdx] + std::max(lhsValue, rhsValue);
  }

  (void)setDenseResult(resultType, outputShape, result);
}

} // namespace atir
