#include "Common.h"
#include "Dialect/Atir/AtirOps.h"

using namespace atir::interpret;

namespace atir {

void GatherOp::Interpret() {
  this->inferShape();
  atir::TensorType paramsType;
  atir::TensorType indicesType;
  atir::TensorType axisType;
  DenseElementsAttr paramsAttr;
  DenseElementsAttr indicesAttr;
  DenseElementsAttr axisAttr;
  if (failed(getTensorTypeAndData(getOperation(), getParams(), "Gather params",
                                  paramsType, paramsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getIndices(),
                                  "Gather indices", indicesType,
                                  indicesAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getAxis(), "Gather axis",
                                  axisType, axisAttr))) {
    return;
  }
  auto paramsValsOr = getFloatValues(paramsAttr);
  auto indicesValsOr = getIntValues(indicesAttr);
  auto axisValsOr = getIntValues(axisAttr);
  if (failed(paramsValsOr) || failed(indicesValsOr) || failed(axisValsOr) ||
      axisValsOr->size() != 1) {
    emitOpError("Gather requires numeric params and scalar integer-like axis");
    return;
  }
  auto paramsShape = paramsType.getShape();
  auto indicesShape = indicesType.getShape();
  int64_t axis = (*axisValsOr)[0];
  int64_t paramsRank = static_cast<int64_t>(paramsShape.size());
  if (axis < 0) {
    axis += paramsRank;
  }
  if (axis < 0 || axis >= paramsRank) {
    emitOpError("Gather axis out of range");
    return;
  }
  int64_t batchDims = getBatchDims();
  int64_t indicesRank = static_cast<int64_t>(indicesShape.size());
  if (batchDims < 0) {
    emitOpError("Gather batch_dims must be non-negative");
    return;
  }
  if (indicesRank == 0) {
    if (batchDims != 0) {
      emitOpError("Gather with scalar indices requires batch_dims == 0");
      return;
    }
  } else if (!(batchDims < indicesRank)) {
    emitOpError("Gather batch_dims must be less than indices rank");
    return;
  }
  if (batchDims > axis) {
    emitOpError("Gather batch_dims must not exceed axis");
    return;
  }
  for (int64_t i = 0; i < batchDims; ++i) {
    int64_t ps = paramsShape[i];
    int64_t is = indicesShape[i];
    if (ps >= 0 && is >= 0 && ps != is) {
      emitOpError(
          "Gather params and indices must have identical shapes in the batch "
          "dimensions");
      return;
    }
  }
  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("Gather output shape must be static for interpretation");
    return;
  }
  const int64_t indicesTailRank = indicesRank - batchDims;
  const auto &paramsVals = *paramsValsOr;
  const auto &indicesVals = *indicesValsOr;
  std::vector<float> result(outputSize, 0.0f);
  for (int64_t outFlat = 0; outFlat < outputSize; ++outFlat) {
    auto outputIndex = getMultiIndex(outputShape, outFlat);
    SmallVector<int64_t> paramsIndex;
    paramsIndex.reserve(paramsShape.size());
    paramsIndex.append(outputIndex.begin(), outputIndex.begin() + axis);
    SmallVector<int64_t> indicesMulti(indicesRank, 0);
    for (int64_t i = 0; i < batchDims; ++i)
      indicesMulti[i] = outputIndex[i];
    for (int64_t i = 0; i < indicesTailRank; ++i)
      indicesMulti[batchDims + i] = outputIndex[axis + i];
    int64_t gathered = indicesVals[getFlatIndex(indicesShape, indicesMulti)];
    if (gathered < 0) {
      gathered += paramsShape[axis];
    }
    if (gathered < 0 || gathered >= paramsShape[axis]) {
      emitOpError("Gather index out of range");
      return;
    }
    paramsIndex.push_back(gathered);
    paramsIndex.append(outputIndex.begin() + axis + indicesTailRank,
                        outputIndex.end());
    result[outFlat] = paramsVals[getFlatIndex(paramsShape, paramsIndex)];
  }

  (void)setDenseResult(resultType, outputShape, result);
}

void GatherNdOp::Interpret() {
  this->inferShape();
  atir::TensorType valuesType;
  atir::TensorType indicesType;
  DenseElementsAttr valuesAttr;
  DenseElementsAttr indicesAttr;
  if (failed(getTensorTypeAndData(getOperation(), getValues(), "GatherNd values",
                                  valuesType, valuesAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getIndices(), "GatherNd indices",
                                  indicesType, indicesAttr))) {
    return;
  }
  auto valuesValsOr = getFloatValues(valuesAttr);
  auto indicesValsOr = getIntValues(indicesAttr);
  if (failed(valuesValsOr) || failed(indicesValsOr)) {
    emitOpError("GatherNd requires numeric values and integer-like indices");
    return;
  }
  auto valuesShape = valuesType.getShape();
  auto indicesShape = indicesType.getShape();
  if (indicesShape.empty()) {
    emitOpError("GatherNd indices must have at least one dimension");
    return;
  }
  int64_t indicesRank = static_cast<int64_t>(indicesShape.size());
  int64_t valuesRank = static_cast<int64_t>(valuesShape.size());
  int64_t lastIndexDim = indicesShape[indicesRank - 1];
  if (lastIndexDim > valuesRank) {
    emitOpError("GatherNd lastIndexDim must not exceed values rank");
    return;
  }
  SmallVector<int64_t> outputShape;
  for (int64_t i = 0; i < indicesRank - 1; ++i) {
    outputShape.push_back(indicesShape[i]);
  }
  for (int64_t i = lastIndexDim; i < valuesRank; ++i) {
    outputShape.push_back(valuesShape[i]);
  }
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("GatherNd output shape must be static for interpretation");
    return;
  }
  const auto &valuesVals = *valuesValsOr;
  const auto &indicesVals = *indicesValsOr;
  std::vector<float> result(outputSize, 0.0f);
  for (int64_t outFlat = 0; outFlat < outputSize; ++outFlat) {
    auto outputIndex = getMultiIndex(outputShape, outFlat);
    SmallVector<int64_t> indicesIndex;
    for (int64_t i = 0; i < indicesRank - 1; ++i) {
      indicesIndex.push_back(outputIndex[i]);
    }
    indicesIndex.push_back(0);
    SmallVector<int64_t> valuesIndex;
    for (int64_t i = 0; i < lastIndexDim; ++i) {
      indicesIndex[indicesRank - 1] = i;
      int64_t flatIndicesIdx = getFlatIndex(indicesShape, indicesIndex);
      if (flatIndicesIdx >= static_cast<int64_t>(indicesVals.size())) {
        emitOpError("GatherNd indices out of bounds");
        return;
      }
      int64_t indexValue = indicesVals[flatIndicesIdx];
      if (indexValue < 0) {
        indexValue += valuesShape[i];
      }
      if (indexValue < 0 || indexValue >= valuesShape[i]) {
        emitOpError("GatherNd index out of range");
        return;
      }
      valuesIndex.push_back(indexValue);
    }
    for (size_t i = indicesRank - 1; i < outputIndex.size(); ++i) {
      valuesIndex.push_back(outputIndex[i]);
    }
    int64_t valuesFlatIdx = getFlatIndex(valuesShape, valuesIndex);
    if (valuesFlatIdx >= static_cast<int64_t>(valuesVals.size())) {
      emitOpError("GatherNd computed index out of bounds");
      return;
    }
    result[outFlat] = valuesVals[valuesFlatIdx];
  }
  
  auto resultType = getResult().getType();
  (void)setDenseResult(resultType, outputShape, result);
}

void SliceOp::Interpret() {
  this->inferShape();
  atir::TensorType inputType;
  atir::TensorType beginType;
  atir::TensorType sizeType;
  DenseElementsAttr inputAttr;
  DenseElementsAttr beginAttr;
  DenseElementsAttr sizeAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(), "Slice input",
                                  inputType, inputAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getBegin(), "Slice begin",
                                  beginType, beginAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getSize(), "Slice size",
                                  sizeType, sizeAttr))) {
    return;
  }
  auto inputValsOr = getFloatValues(inputAttr);
  auto beginValsOr = getIntValues(beginAttr);
  auto sizeValsOr = getIntValues(sizeAttr);
  if (failed(inputValsOr) || failed(beginValsOr) || failed(sizeValsOr)) {
    emitOpError("Slice requires numeric input and integer begin/size");
    return;
  }
  auto inputShape = inputType.getShape();
  const auto &inputVals = *inputValsOr;
  const auto &beginVals = *beginValsOr;
  const auto &sizeVals = *sizeValsOr;
  if (beginVals.size() != inputShape.size() || sizeVals.size() != inputShape.size()) {
    emitOpError("Slice begin and size must match input rank");
    return;
  }
  SmallVector<int64_t> outputShape;
  SmallVector<int64_t> beginIndices;
  SmallVector<int64_t> endIndices;
  for (size_t i = 0; i < inputShape.size(); ++i) {
    int64_t begin = beginVals[i];
    int64_t size = sizeVals[i];
    int64_t dimSize = inputShape[i];
    if (begin < 0) {
      begin += dimSize;
    }
    if (begin < 0 || begin >= dimSize) {
      emitOpError("Slice begin index out of bounds");
      return;
    }
    if (size < 0) {
      size = dimSize - begin;
    }
    int64_t end = begin + size;
    if (end > dimSize) {
      emitOpError("Slice size exceeds dimension bounds");
      return;
    }
    outputShape.push_back(size);
    beginIndices.push_back(begin);
    endIndices.push_back(end);
  }
  int64_t outputSize = getElementCount(outputShape);
  std::vector<float> result(outputSize, 0.0f);
  for (int64_t outFlat = 0; outFlat < outputSize; ++outFlat) {
    auto outputIndex = getMultiIndex(outputShape, outFlat);
    SmallVector<int64_t> inputIndex;
    for (size_t i = 0; i < outputIndex.size(); ++i) {
      inputIndex.push_back(beginIndices[i] + outputIndex[i]);
    }
    int64_t inputFlatIdx = getFlatIndex(inputShape, inputIndex);
    if (inputFlatIdx >= static_cast<int64_t>(inputVals.size())) {
      emitOpError("Slice computed index out of bounds");
      return;
    }
    result[outFlat] = inputVals[inputFlatIdx];
  }
  auto resultType = getResult().getType();
  (void)setDenseResult(resultType, outputShape, result);
}
void LoadOp::Interpret() {
  this->inferShape();
  atir::TensorType inputType;
  DenseElementsAttr inputAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(), "Load input",
                                  inputType, inputAttr))) {
    return;
  }
  auto inputValsOr = getFloatValues(inputAttr);
  if (failed(inputValsOr)) {
    emitOpError("Load only supports numeric input cacheData");
    return;
  }
  auto inputShape = inputType.getShape();
  auto outputType = getOutput().getType();
  auto outputShape = outputType.getShape();
  auto axes = getIntArrayAttrValues(getAxes());
  auto starts = getIntArrayAttrValues(getStart());
  auto sizes = getIntArrayAttrValues(getSize());
  if (axes.size() != starts.size() || axes.size() != sizes.size()) {
    emitOpError("Load axes/start/size must have the same length");
    return;
  }
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("Load output shape must be static for interpretation");
    return;
  }
  SmallVector<int64_t> axisStart(inputShape.size(), 0);
  SmallVector<int64_t> axisSize(inputShape.begin(), inputShape.end());
  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t axis = axes[i];
    if (axis < 0 || axis >= static_cast<int64_t>(inputShape.size())) {
      emitOpError("Load axis out of range");
      return;
    }
    axisStart[axis] = starts[i];
    axisSize[axis] = sizes[i];
  }
  const auto &inputVals = *inputValsOr;
  std::vector<float> result(outputSize, 0.0f);
  for (int64_t outFlat = 0; outFlat < outputSize; ++outFlat) {
    auto outIndex = getMultiIndex(outputShape, outFlat);
    SmallVector<int64_t> inIndex(outIndex.begin(), outIndex.end());
    for (size_t axis = 0; axis < inIndex.size(); ++axis) {
      inIndex[axis] += axisStart[axis];
      if (inIndex[axis] < 0 || inIndex[axis] >= inputShape[axis]) {
        emitOpError("Load slice out of bounds");
        return;
      }
    }
    result[outFlat] = inputVals[getFlatIndex(inputShape, inIndex)];
  }
  (void)setDenseResult(outputType, outputShape, result);
}

void StridedSliceOp::Interpret() {
  this->inferShape();
  atir::TensorType inputType;
  DenseElementsAttr inputAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(),
                                  "StridedSlice input", inputType,
                                  inputAttr))) {
    return;
  }
  auto inputValsOr = getFloatValues(inputAttr);
  if (failed(inputValsOr)) {
    emitOpError("StridedSlice input cacheData type is not supported");
    return;
  }
  atir::TensorType beginType, endType, stridesType;
  DenseElementsAttr beginAttr, endAttr, stridesAttr;
  if (failed(getTensorTypeAndData(getOperation(), getBegin(), "StridedSlice begin",
                                  beginType, beginAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getEnd(), "StridedSlice end",
                                  endType, endAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getStrides(), "StridedSlice strides",
                                  stridesType, stridesAttr))) {
    return;
  }
  auto beginValsOr = getIntValues(beginAttr);
  auto endValsOr = getIntValues(endAttr);
  auto strideValsOr = getIntValues(stridesAttr);
  if (failed(beginValsOr) || failed(endValsOr) || failed(strideValsOr)) {
    emitOpError("StridedSlice begin/end/strides cacheData must be numeric");
    return;
  }
  auto inputShape = inputType.getShape();
  size_t rank = inputShape.size();
  const auto &beginVals = *beginValsOr;
  const auto &endVals = *endValsOr;
  const auto &strideVals = *strideValsOr;
  if (beginVals.size() != endVals.size() || beginVals.size() != strideVals.size()) {
    emitOpError("StridedSlice begin/end/strides must have the same length");
    return;
  }
  if (rank == 0) {
    emitOpError("StridedSlice on scalar input is not supported");
    return;
  }
  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("StridedSlice output shape must be static for interpretation");
    return;
  }
  const auto &inputVals = *inputValsOr;
  std::vector<float> result(outputSize, 0.0f);
  for (int64_t outFlat = 0; outFlat < outputSize; ++outFlat) {
    auto outputIndex = getMultiIndex(outputShape, outFlat);
    SmallVector<int64_t> inputIndex(rank, 0);
    for (size_t axis = 0; axis < rank; ++axis) {
      inputIndex[axis] = beginVals[axis] + outputIndex[axis] * strideVals[axis];
    }
    result[outFlat] = inputVals[getFlatIndex(inputShape, inputIndex)];
  }
  (void)setDenseResult(resultType, outputShape, result);
}
} // namespace atir