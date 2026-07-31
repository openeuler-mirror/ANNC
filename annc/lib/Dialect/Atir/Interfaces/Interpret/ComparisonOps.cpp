#include "Common.h"
#include "Dialect/Atir/AtirOps.h"

using namespace atir::interpret;

namespace atir {

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

  // String compare (dnn_embedding_hash_bucket): lhs is a string tensor, rhs is
  // the ignore-value constant. NE => "string is non-empty (not the ignore
  // value)"; produce a NUMERIC mask because downstream Where reads the
  // condition via getFloatValues. Mirrors the hand-written
  // kp_fused_embedding_hash_bucket's `if (!item.data || item.size<=0) continue`
  // empty-skip.
  if (isStringTensor(lhsType)) {
    auto lhsStringsOr = getStringValues(lhsAttr);
    if (failed(lhsStringsOr)) {
      emitOpError("Compare string lhs has no string cacheData");
      return;
    }
    StringRef dir = getComparisonDirection();
    std::vector<int64_t> mask;
    mask.reserve(lhsStringsOr->size());
    for (const std::string &s : *lhsStringsOr) {
      bool isEmpty = s.empty();
      bool pred;
      if (dir == "NE") pred = !isEmpty;
      else if (dir == "EQ") pred = isEmpty;
      else {
        emitOpError("string Compare only supports NE/EQ");
        return;
      }
      mask.push_back(pred ? 1 : 0);
    }
    auto resultType = getResult().getType();
    auto resolvedShape = interpret::resolveDynamicShape(
        resultType.getShape(), (int64_t)mask.size());
    // Emit the mask in the declared result element type (bool/i32 for
    // comparison ops, or float for legacy float-mask consumers).
    if (failed(setBooleanLikeResult(resultType, resolvedShape, mask))) {
      emitOpError("string Compare output element type is not supported");
    }
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
