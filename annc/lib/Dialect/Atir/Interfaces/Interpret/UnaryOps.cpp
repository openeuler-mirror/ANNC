#include "Common.h"
#include "Dialect/Atir/AtirOps.h"

using namespace atir::interpret;

namespace atir {

void ReluOp::Interpret() {
  this->inferShape();

  atir::TensorType inputTensor;
  DenseElementsAttr inputAttr;
  if (failed(
          getTensorTypeAndData(getOperation(), getInput(), "Relu input",
                               inputTensor, inputAttr))) {
    return;
  }

  auto inputValsOr = getFloatValues(inputAttr);
  if (failed(inputValsOr)) {
    emitOpError("Relu only supports numeric input cacheData");
    return;
  }

  const auto &inputVals = *inputValsOr;
  float reluLimit =
      annc::apFloatToFloat(cast<FloatAttr>(getReluLimitAttr()).getValue());
  
  std::vector<float> result(inputVals.size(), 0.0f);;
  
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = std::max(0.0f, inputVals[i]);
    if (reluLimit >= 0.0f) {
      result[i] = std::min(result[i], reluLimit);
    }
  }

  auto resultType = this->getResult().getType();
  auto outputShape = resultType.getShape();
  (void)setDenseResult(resultType, outputShape, result);
}

void LogisticOp::Interpret() {
  this->inferShape();

  atir::TensorType inputType;
  DenseElementsAttr inputAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(), "Logistic input",
                                  inputType, inputAttr))) {
    return;
  }

  auto valsOr = getFloatValues(inputAttr);
  if (failed(valsOr)) {
    emitOpError("Logistic only supports numeric input cacheData");
    return;
  }

  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t n = getElementCount(outputShape);
  if (n < 0) {
    emitOpError("Logistic output shape must be static for interpretation");
    return;
  }

  std::vector<float> result(n, 0.0f);

  for (int64_t i = 0; i < n; ++i) {
    float x = (*valsOr)[i];
    float y;
    if (x >= 0.0f) {
      float z = std::exp(-x);
      y = 1.0f / (1.0f + z);
    } else {
      float z = std::exp(x);
      y = z / (1.0f + z);
    }
    result[i] = result[i] + y;
  }
  (void)setDenseResult(resultType, outputShape, result);
}

void AbsOp::Interpret() {
  this->inferShape();

  atir::TensorType inputType;
  DenseElementsAttr inputAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(), "Abs input",
                                  inputType, inputAttr))) {
    return;
  }

  auto valsOr = getFloatValues(inputAttr);
  if (failed(valsOr)) {
    emitOpError("Abs only supports numeric input cacheData");
    return;
  }

  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t n = getElementCount(outputShape);
  if (n < 0) {
    emitOpError("Abs output shape must be static for interpretation");
    return;
  }

  std::vector<float> result(n, 0.0f);
  
  for (int64_t i = 0; i < n; ++i)
    result[i] = result[i] + std::fabs((*valsOr)[i]);
  (void)setDenseResult(resultType, outputShape, result);
}

void RsqrtOp::Interpret() {
  this->inferShape();

  atir::TensorType inputType;
  DenseElementsAttr inputAttr;
  if (failed(getTensorTypeAndData(getOperation(), getX(), "Rsqrt input",
                                  inputType, inputAttr))) {
    return;
  }

  auto inputValsOr = getFloatValues(inputAttr);
  if (failed(inputValsOr)) {
    emitOpError("Rsqrt only supports numeric input cacheData");
    return;
  }

  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("Rsqrt output shape must be static for interpretation");
    return;
  }

  std::vector<float> result(outputSize, 0.0f);
  
  for (int64_t i = 0; i < outputSize; ++i) {
    result[i] = result[i] + (1.0f / std::sqrt((*inputValsOr)[i]));
  }

  (void)setDenseResult(resultType, outputShape, result);
}

void ZerosLikeOp::Interpret() {
  this->inferShape();

  atir::TensorType inputType;
  DenseElementsAttr inputAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(), "ZerosLike input",
                                  inputType, inputAttr))) {
    return;
  }

  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("ZerosLike output shape must be static for interpretation");
    return;
  }

  auto elementType = resultType.getElementType();
  if (isa<FloatType>(elementType)) {
    std::vector<float> zeros(outputSize, 0.0f);
    (void)setDenseResult(resultType, outputShape, zeros);
    return;
  }

  std::vector<int64_t> zeros(outputSize, 0);
  if (failed(setDenseIntResult(resultType, outputShape, zeros))) {
    emitOpError("ZerosLike output element type is not supported");
  }
}
} // namespace atir
