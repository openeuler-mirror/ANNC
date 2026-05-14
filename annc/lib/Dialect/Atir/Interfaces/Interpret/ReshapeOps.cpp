#include "Common.h"
#include "Dialect/Atir/AtirOps.h"

using namespace atir::interpret;

namespace atir {
void ReshapeOp::Interpret() {
  this->inferShape();
  atir::TensorType inputType;
  DenseElementsAttr inputAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(), "Reshape input",
                                  inputType, inputAttr))) {
    return;
  }
  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("Reshape output shape must be static for interpretation");
    return;
  }
  Type elemTy = resultType.getElementType();
  if (isa<FloatType>(elemTy)) {
    auto valsOr = getFloatValues(inputAttr);
    if (failed(valsOr)) {
      emitOpError("Reshape only supports float dense input");
      return;
    }
    if (static_cast<int64_t>(valsOr->size()) != outputSize) {
      emitOpError("Reshape element count mismatch");
      return;
    }
    (void)setDenseResult(resultType, outputShape, *valsOr);
    return;
  }
  if (isa<IntegerType, IndexType>(elemTy)) {
    auto valsOr = getIntValues(inputAttr);
    if (failed(valsOr)) {
      emitOpError("Reshape only supports integer dense input");
      return;
    }
    if (static_cast<int64_t>(valsOr->size()) != outputSize) {
      emitOpError("Reshape element count mismatch");
      return;
    }
    if (failed(setDenseIntResult(resultType, outputShape, *valsOr))) {
      emitOpError("Reshape integer output element type is not supported");
    }
    return;
  }
  emitOpError("Reshape element type is not supported");
}
void TransposeOp::Interpret() {
  this->inferShape();
  atir::TensorType inputType;
  DenseElementsAttr inputAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(), "Transpose input",
                                  inputType, inputAttr))) {
    return;
  }
  auto valsOr = getFloatValues(inputAttr);
  if (failed(valsOr)) {
    emitOpError("Transpose only supports numeric input cacheData");
    return;
  }
  auto inShape = inputType.getShape();
  int64_t rank = static_cast<int64_t>(inShape.size());
  auto permAttr = getPermutationAttr();
  if (!permAttr) {
    emitOpError("Transpose missing permutation attribute");
    return;
  }
  llvm::SmallVector<int64_t> perm;
  for (Attribute a : permAttr) {
    auto ia = dyn_cast<IntegerAttr>(a);
    if (!ia) {
      emitOpError("Transpose permutation must be integer attributes");
      return;
    }
    int64_t p = ia.getInt();
    if (p < 0)
      p += rank;
    perm.push_back(p);
  }
  if (static_cast<int64_t>(perm.size()) != rank) {
    emitOpError("Transpose permutation length mismatch");
    return;
  }
  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("Transpose output shape must be static for interpretation");
    return;
  }
  const auto &vals = *valsOr;
  std::vector<float> result(outputSize, 0.0f);
  for (int64_t outFlat = 0; outFlat < outputSize; ++outFlat) {
    auto outIdx = getMultiIndex(outputShape, outFlat);
    SmallVector<int64_t, 8> inIdx(rank, 0);
    for (int64_t d = 0; d < rank; ++d)
      inIdx[perm[d]] = outIdx[d];
    result[outFlat] = vals[getFlatIndex(inShape, inIdx)];
  }
  (void)setDenseResult(resultType, outputShape, result);
}

void ExpandDimsOp::Interpret() {
  this->inferShape();
  atir::TensorType inputType;
  DenseElementsAttr inputAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(),
                                  "ExpandDims input", inputType, inputAttr))) {
    return;
  }
  auto inputValsOr = getFloatValues(inputAttr);
  if (failed(inputValsOr)) {
    emitOpError("ExpandDims only supports numeric input cacheData");
    return;
  }
  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  (void)setDenseResult(resultType, outputShape, *inputValsOr);
}
void TileOp::Interpret() {
  this->inferShape();
  atir::TensorType inputType;
  DenseElementsAttr inputAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(), "Tile input",
                                  inputType, inputAttr))) {
    return;
  }
  atir::TensorType multiplesType;
  DenseElementsAttr multiplesAttr;
  if (failed(getTensorTypeAndData(getOperation(), getMultiples(),
                                  "Tile multiples", multiplesType,
                                  multiplesAttr))) {
    return;
  }
  auto inputValsOr = getFloatValues(inputAttr);
  auto multiplesValsOr = getIntValues(multiplesAttr);
  if (failed(inputValsOr) || failed(multiplesValsOr)) {
    emitOpError("Tile only supports numeric input and integer-like multiples");
    return;
  }
  auto inputShape = inputType.getShape();
  if (multiplesValsOr->size() != inputShape.size()) {
    emitOpError("Tile multiples rank mismatch");
    return;
  }
  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("Tile output shape must be static for interpretation");
    return;
  }
  const auto &inputVals = *inputValsOr;
  std::vector<float> result(outputSize, 0.0f);
  for (int64_t outFlat = 0; outFlat < outputSize; ++outFlat) {
    auto outputIndex = getMultiIndex(outputShape, outFlat);
    SmallVector<int64_t> inputIndex;
    inputIndex.reserve(inputShape.size());
    for (size_t axis = 0; axis < inputShape.size(); ++axis) {
      inputIndex.push_back(outputIndex[axis] % inputShape[axis]);
    }
    result[outFlat] = inputVals[getFlatIndex(inputShape, inputIndex)];
  }
  (void)setDenseResult(resultType, outputShape, result);
}

void BroadcastOp::Interpret() {
  this->inferShape();
  atir::TensorType inputType;
  DenseElementsAttr inputAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(), "broadcast input",
                                  inputType, inputAttr))) {
    return;
  }
  auto inputValsOr = getFloatValues(inputAttr);
  if (failed(inputValsOr)) {
    emitOpError("broadcast requires numeric input");
    return;
  }
  auto inputShape = inputType.getShape();
  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("broadcast output shape must be static");
    return;
  }
  if (getElementCount(inputShape) < 0) {
    emitOpError("broadcast input shape must be static");
    return;
  }
  if (inputShape.size() > outputShape.size()) {
    emitOpError("broadcast input rank must be <= output rank");
    return;
  }
  int64_t rankDiff = static_cast<int64_t>(outputShape.size()) -
                     static_cast<int64_t>(inputShape.size());
  for (size_t i = 0; i < inputShape.size(); ++i) {
    int64_t inDim = inputShape[i];
    int64_t outDim = outputShape[static_cast<size_t>(rankDiff + i)];
    if (inDim != 1 && inDim != outDim) {
      emitOpError("broadcast shape mismatch");
      return;
    }
  }
  const auto &inputVals = *inputValsOr;
  std::vector<float> outputVals(static_cast<size_t>(outputSize), 0.0f);
  for (int64_t outFlat = 0; outFlat < outputSize; ++outFlat) {
    auto outIndex = getMultiIndex(outputShape, outFlat);
    SmallVector<int64_t> inIndex;
    inIndex.reserve(inputShape.size());
    for (size_t i = 0; i < inputShape.size(); ++i) {
      int64_t outAxis = rankDiff + static_cast<int64_t>(i);
      inIndex.push_back(inputShape[i] == 1 ? 0 : outIndex[static_cast<size_t>(outAxis)]);
    }
    outputVals[static_cast<size_t>(outFlat)] =
        inputVals[static_cast<size_t>(getFlatIndex(inputShape, inIndex))];
  }
  if (isa<FloatType>(resultType.getElementType())) {
    (void)setDenseResult(resultType, outputShape, outputVals);
    return;
  }
  std::vector<int64_t> intResult;
  intResult.reserve(outputVals.size());
  for (float v : outputVals) {
    intResult.push_back(static_cast<int64_t>(v));
  }
  if (failed(setDenseIntResult(resultType, outputShape, intResult))) {
    emitOpError("broadcast output element type is not supported");
  }
}
void PadOp::Interpret() {
  this->inferShape();
  atir::TensorType inputType;
  atir::TensorType paddingsType;
  atir::TensorType valueType;
  DenseElementsAttr inputAttr;
  DenseElementsAttr paddingsAttr;
  DenseElementsAttr valueAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(), "Pad input",
                                  inputType, inputAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getPaddings(), "Pad paddings",
                                  paddingsType, paddingsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getValue(), "Pad value",
                                  valueType, valueAttr))) {
    return;
  }
  auto inputValsOr = getFloatValues(inputAttr);
  auto paddingsValsOr = getIntValues(paddingsAttr);
  auto padValOr = getFloatValues(valueAttr);
  if (failed(inputValsOr) || failed(paddingsValsOr) || failed(padValOr) ||
      padValOr->size() != 1) {
    emitOpError("Pad requires numeric input/value and integer-like paddings");
    return;
  }
  auto inputShape = inputType.getShape();
  if (static_cast<size_t>(paddingsValsOr->size()) != inputShape.size() * 2) {
    emitOpError("Pad paddings size must equal 2 * input rank");
    return;
  }
  SmallVector<int64_t> outputShape;
  outputShape.reserve(inputShape.size());
  SmallVector<int64_t> padBefore;
  padBefore.reserve(inputShape.size());
  for (size_t i = 0; i < inputShape.size(); ++i) {
    int64_t before = (*paddingsValsOr)[2 * i];
    int64_t after = (*paddingsValsOr)[2 * i + 1];
    if (before < 0 || after < 0 || inputShape[i] < 0) {
      emitOpError("Pad only supports static shape with non-negative paddings");
      return;
    }
    padBefore.push_back(before);
    outputShape.push_back(inputShape[i] + before + after);
  }
  int64_t outputSize = getElementCount(outputShape);
  int64_t inputSize = getElementCount(inputShape);
  if (outputSize < 0 || inputSize < 0) {
    emitOpError("Pad requires static input/output shapes");
    return;
  }
  std::vector<float> outputVals(static_cast<size_t>(outputSize), (*padValOr)[0]);
  const auto &inputVals = *inputValsOr;
  for (int64_t inFlat = 0; inFlat < inputSize; ++inFlat) {
    auto inIndex = getMultiIndex(inputShape, inFlat);
    SmallVector<int64_t> outIndex;
    outIndex.reserve(inIndex.size());
    for (size_t a = 0; a < inIndex.size(); ++a) {
      outIndex.push_back(inIndex[a] + padBefore[a]);
    }
    int64_t outFlat = getFlatIndex(outputShape, outIndex);
    outputVals[static_cast<size_t>(outFlat)] = inputVals[static_cast<size_t>(inFlat)];
  }
  auto resultType = getResult().getType();
  if (isa<FloatType>(resultType.getElementType())) {
    (void)setDenseResult(resultType, outputShape, outputVals);
    return;
  }
  std::vector<int64_t> intResult;
  intResult.reserve(outputVals.size());
  for (float v : outputVals) {
    intResult.push_back(static_cast<int64_t>(v));
  }
  if (failed(setDenseIntResult(resultType, outputShape, intResult))) {
    emitOpError("Pad output element type is not supported");
  }
}
} // namespace atir
