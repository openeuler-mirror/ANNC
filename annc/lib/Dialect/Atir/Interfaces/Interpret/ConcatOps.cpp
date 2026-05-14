#include "Common.h"
#include "Dialect/Atir/AtirOps.h"

using namespace atir::interpret;

namespace atir {

void ConcatOp::Interpret() {
  this->inferShape();

  SmallVector<atir::TensorType> inputTypes;
  SmallVector<std::vector<float>> inputValues;
  for (size_t i = 1; i < getNumOperands(); ++i) {
    Value value = getOperand(i);
    if (auto noneOp = dyn_cast_or_null<atir::NoneOp>(value.getDefiningOp())) {
      (void)noneOp;
      continue;
    }

    atir::TensorType tensorType;
    DenseElementsAttr attr;
    if (failed(getTensorTypeAndData(getOperation(), value, "Concat input",
                                    tensorType, attr))) {
      return;
    }

    auto valuesOr = getFloatValues(attr);
    if (failed(valuesOr)) {
      emitOpError("Concat only supports numeric input cacheData");
      return;
    }

    inputTypes.push_back(tensorType);
    inputValues.push_back(std::move(*valuesOr));
  }

  if (inputTypes.empty()) {
    emitOpError("Concat requires at least one input tensor");
    return;
  }

  auto outputType = getResult().getType();
  auto outputShape = outputType.getShape();
  int64_t rank = static_cast<int64_t>(outputShape.size());
  int64_t axis = getAxis();
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    emitOpError("Concat axis out of range");
    return;
  }

  int64_t outer = 1;
  int64_t inner = 1;
  for (int64_t i = 0; i < axis; ++i) {
    outer *= outputShape[i];
  }
  for (int64_t i = axis + 1; i < rank; ++i) {
    inner *= outputShape[i];
  }

  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("Concat output shape must be static for interpretation");
    return;
  }

  std::vector<float> result(outputSize, 0.0f);
  
  for (int64_t outerIdx = 0; outerIdx < outer; ++outerIdx) {
    int64_t dstBase = outerIdx * outputShape[axis] * inner;
    int64_t axisOffset = 0;

    for (size_t inputIdx = 0; inputIdx < inputTypes.size(); ++inputIdx) {
      auto inputShape = inputTypes[inputIdx].getShape();
      int64_t axisExtent = inputShape[axis];
      int64_t chunkSize = axisExtent * inner;
      int64_t srcBase = outerIdx * chunkSize;
      std::copy_n(inputValues[inputIdx].begin() + srcBase, chunkSize,
                  result.begin() + dstBase + axisOffset * inner);
      axisOffset += axisExtent;
    }
  }

  if (getDoRelu()) {
    float reluLimit =
        annc::apFloatToFloat(cast<FloatAttr>(getReluLimitAttr()).getValue());
    for (float &value : result) {
      value = applyReluLimit(value, true, reluLimit);
    }
  }

  (void)setDenseResult(outputType, outputShape, result);
}

void ConcatV2Op::Interpret() {
  this->inferShape();

  atir::TensorType axisType;
  DenseElementsAttr axisAttr;
  if (failed(getTensorTypeAndData(getOperation(), getAxis(), "ConcatV2 axis",
                                  axisType, axisAttr))) {
    return;
  }

  auto axisValsOr = getIntValues(axisAttr);
  if (failed(axisValsOr) || axisValsOr->empty()) {
    emitOpError("ConcatV2 axis must be a scalar integer tensor");
    return;
  }

  SmallVector<atir::TensorType> inputTypes;
  SmallVector<std::vector<float>> inputValues;
  for (Value value : getValues()) {
    atir::TensorType tensorType;
    DenseElementsAttr attr;
    if (failed(getTensorTypeAndData(getOperation(), value, "ConcatV2 input",
                                    tensorType, attr))) {
      return;
    }

    auto valuesOr = getFloatValues(attr);
    if (failed(valuesOr)) {
      emitOpError("ConcatV2 only supports numeric input cacheData");
      return;
    }

    inputTypes.push_back(tensorType);
    inputValues.push_back(std::move(*valuesOr));
  }

  if (inputTypes.empty()) {
    emitOpError("ConcatV2 requires at least one input tensor");
    return;
  }

  auto outputType = getResult().getType();
  auto outputShape = outputType.getShape();
  int64_t rank = static_cast<int64_t>(outputShape.size());
  int64_t axis = axisValsOr->front();
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    emitOpError("ConcatV2 axis out of range");
    return;
  }

  int64_t outer = 1;
  int64_t inner = 1;
  for (int64_t i = 0; i < axis; ++i) {
    outer *= outputShape[i];
  }
  for (int64_t i = axis + 1; i < rank; ++i) {
    inner *= outputShape[i];
  }

  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("ConcatV2 output shape must be static for interpretation");
    return;
  }

  std::vector<float> result(outputSize, 0.0f);
  for (int64_t outerIdx = 0; outerIdx < outer; ++outerIdx) {
    int64_t dstBase = outerIdx * outputShape[axis] * inner;
    int64_t axisOffset = 0;

    for (size_t inputIdx = 0; inputIdx < inputTypes.size(); ++inputIdx) {
      auto inputShape = inputTypes[inputIdx].getShape();
      int64_t axisExtent = inputShape[axis];
      int64_t chunkSize = axisExtent * inner;
      int64_t srcBase = outerIdx * chunkSize;
      std::copy_n(inputValues[inputIdx].begin() + srcBase, chunkSize,
                  result.begin() + dstBase + axisOffset * inner);
      axisOffset += axisExtent;
    }
  }

  (void)setDenseResult(outputType, outputShape, result);
}

void PackOp::Interpret() {
  this->inferShape();

  if (getInputs().empty()) {
    emitOpError("Pack requires at least one input");
    return;
  }

  SmallVector<atir::TensorType> inputTypes;
  SmallVector<DenseElementsAttr> inputAttrs;
  for (auto [index, input] : llvm::enumerate(getInputs())) {
    atir::TensorType inputType;
    DenseElementsAttr inputAttr;
    if (failed(getTensorTypeAndData(getOperation(), input,
                                    ("Pack input " + std::to_string(index)).c_str(),
                                    inputType, inputAttr))) {
      return;
    }
    inputTypes.push_back(inputType);
    inputAttrs.push_back(inputAttr);
  }

  SmallVector<std::vector<float>> inputValues;
  for (DenseElementsAttr attr : inputAttrs) {
    auto valuesOr = getFloatValues(attr);
    if (failed(valuesOr)) {
      emitOpError("Pack only supports numeric input cacheData");
      return;
    }
    inputValues.push_back(*valuesOr);
  }

  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("Pack output shape must be static for interpretation");
    return;
  }

  auto inputShape = inputTypes.front().getShape();
  int64_t axis = getAxis();
  int64_t inputRank = static_cast<int64_t>(inputShape.size());
  if (axis < 0) {
    axis += inputRank + 1;
  }

  std::vector<float> result(outputSize, 0.0f);
  for (int64_t outFlat = 0; outFlat < outputSize; ++outFlat) {
    auto outputIndex = getMultiIndex(outputShape, outFlat);
    int64_t inputNumber = outputIndex[axis];

    SmallVector<int64_t> inputIndex;
    inputIndex.reserve(inputRank);
    for (int64_t outAxis = 0; outAxis < static_cast<int64_t>(outputShape.size());
         ++outAxis) {
      if (outAxis == axis) {
        continue;
      }
      inputIndex.push_back(outputIndex[outAxis]);
    }

    result[outFlat] =
        inputValues[inputNumber][getFlatIndex(inputShape, inputIndex)];
  }

  (void)setDenseResult(resultType, outputShape, result);
}

}  // namespace atir
