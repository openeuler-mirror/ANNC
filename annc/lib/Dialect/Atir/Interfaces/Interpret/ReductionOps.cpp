#include "Common.h"
#include "Dialect/Atir/AtirOps.h"

using namespace atir::interpret;

namespace atir {

void SumOp::Interpret() {
  this->inferShape();
  atir::TensorType inputType;
  DenseElementsAttr inputAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(), "Sum input",
                                  inputType, inputAttr))) {
    return;
  }
  auto inputValsOr = getFloatValues(inputAttr);
  if (failed(inputValsOr)) {
    emitOpError("Sum only supports numeric input cacheData");
    return;
  }
  std::vector<int64_t> axes;
  if (auto indicesType = dyn_cast<atir::TensorType>(getIndices().getType())) {
    auto indicesAttr = indicesType.getCacheData();
    if (indicesAttr && !indicesAttr.empty()) {
      auto indicesValsOr = getIntValues(indicesAttr);
      if (failed(indicesValsOr)) {
        emitOpError("Sum indices must be integer-like");
        return;
      }
      axes.assign(indicesValsOr->begin(), indicesValsOr->end());
    }
  }
  auto inputShape = inputType.getShape();
  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  if (axes.empty()) axes = inferReductionAxesFromShapes(inputShape, outputShape);
  if (axes.empty()) {
    for (int64_t axis = 0; axis < static_cast<int64_t>(inputShape.size()); ++axis) {
      axes.push_back(axis);
    }
  }
  for (int64_t &axis : axes) {
    if (axis < 0) axis += static_cast<int64_t>(inputShape.size());
    if (axis < 0 || axis >= static_cast<int64_t>(inputShape.size())) {
      emitOpError("Sum axis out of range");
      return;
    }
  }
  std::sort(axes.begin(), axes.end());
  axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
  int64_t inputSize = getElementCount(inputShape);
  int64_t outputSize = getElementCount(outputShape);
  if (inputSize < 0 || outputSize < 0) {
    emitOpError("Sum shapes must be static for interpretation");
    return;
  }
  bool keepDims = outputShape.size() == inputShape.size() || getKeepDims();
  std::vector<float> result(outputSize, 0.0f);
  const auto &inputVals = *inputValsOr;
  for (int64_t inputFlat = 0; inputFlat < inputSize; ++inputFlat) {
    auto inputIndex = getMultiIndex(inputShape, inputFlat);
    SmallVector<int64_t> outputIndex;
    if (keepDims) {
      outputIndex.assign(inputShape.size(), 0);
    }
    for (size_t axis = 0; axis < inputShape.size(); ++axis) {
      bool reduced =
          std::binary_search(axes.begin(), axes.end(), static_cast<int64_t>(axis));
      if (keepDims) {
        outputIndex[axis] = reduced ? 0 : inputIndex[axis];
      } else if (!reduced) {
        outputIndex.push_back(inputIndex[axis]);
      }
    }
    result[getFlatIndex(outputShape, outputIndex)] += inputVals[inputFlat];
  }
  (void)setDenseResult(resultType, outputShape, result);
}

void ProdOp::Interpret() {
  this->inferShape();
  atir::TensorType inputType;
  DenseElementsAttr inputAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(), "Prod input",
                                  inputType, inputAttr)))  return;

  auto inputValsOr = getFloatValues(inputAttr);
  if (failed(inputValsOr)) {
    emitOpError("Prod only supports numeric input cacheData");
    return;
  }
  std::vector<int64_t> axes;
  if (auto indicesType = dyn_cast<atir::TensorType>(getIndices().getType())) {
    auto indicesAttr = indicesType.getCacheData();
    if (indicesAttr && !indicesAttr.empty()) {
      auto indicesValsOr = getIntValues(indicesAttr);
      if (failed(indicesValsOr)) {
        emitOpError("Prod indices must be integer-like");
        return;
      }
      axes.assign(indicesValsOr->begin(), indicesValsOr->end());
    }
  }
  auto inputShape = inputType.getShape();
  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  if (axes.empty()) axes = inferReductionAxesFromShapes(inputShape, outputShape);
  if (axes.empty()) {
    for (int64_t axis = 0; axis < static_cast<int64_t>(inputShape.size()); ++axis) {
      axes.push_back(axis);
    }
  }

  for (int64_t &axis : axes) {
    if (axis < 0)  axis += static_cast<int64_t>(inputShape.size());
    if (axis < 0 || axis >= static_cast<int64_t>(inputShape.size())) {
      emitOpError("Prod axis out of range");
      return;
    }
  }
  std::sort(axes.begin(), axes.end());
  axes.erase(std::unique(axes.begin(), axes.end()), axes.end());

  int64_t inputSize = getElementCount(inputShape);
  int64_t outputSize = getElementCount(outputShape);
  if (inputSize < 0 || outputSize < 0) {
    emitOpError("Prod shapes must be static for interpretation");
    return;
  }
  bool keepDims = outputShape.size() == inputShape.size() || getKeepDims();
  std::vector<float> result(outputSize, 1.0f);
  std::vector<bool> touched(outputSize, false);
  const auto &inputVals = *inputValsOr;

  for (int64_t inputFlat = 0; inputFlat < inputSize; ++inputFlat) {
    auto inputIndex = getMultiIndex(inputShape, inputFlat);
    SmallVector<int64_t> outputIndex;
    if (keepDims) outputIndex.assign(inputShape.size(), 0);
    for (size_t axis = 0; axis < inputShape.size(); ++axis) {
      bool reduced =
          std::binary_search(axes.begin(), axes.end(), static_cast<int64_t>(axis));
      if (keepDims) {
        outputIndex[axis] = reduced ? 0 : inputIndex[axis];
      } else if (!reduced) {
        outputIndex.push_back(inputIndex[axis]);
      }
    }
    int64_t outFlat = getFlatIndex(outputShape, outputIndex);
    result[outFlat] *= inputVals[inputFlat];
    touched[outFlat] = true;
  }
  for (int64_t outFlat = 0; outFlat < outputSize; ++outFlat) {
    if (!touched[outFlat]) {
      result[outFlat] = 1.0f;
    }
  }
  (void)setDenseResult(resultType, outputShape, result);
}

} // namespace atir
