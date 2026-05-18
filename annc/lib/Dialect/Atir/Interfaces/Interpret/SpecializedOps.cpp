#include "Common.h"
#include "Dialect/Atir/AtirOps.h"

using namespace atir::interpret;

namespace atir {

void CastOp::Interpret() {
  this->inferShape();

  atir::TensorType inputType;
  DenseElementsAttr inputAttr;
  if (failed(
          getTensorTypeAndData(getOperation(), getInput(), "Cast input",
                               inputType, inputAttr))) {
    return;
  }
  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  auto elementType = resultType.getElementType();
  if (tensorTypeIsBoolAsI32(resultType)) {
    if (isa<FloatType>(inputType.getElementType())) {
      auto valuesOr = getFloatValues(inputAttr);
      if (failed(valuesOr)) {
        emitOpError("Cast input type is not supported");
        return;
      }
      (void)setDenseResult(resultType, outputShape, *valuesOr);
      return;
    }
    auto valuesOr = getIntValues(inputAttr);
    if (failed(valuesOr)) {
      emitOpError("Cast input type is not supported");
      return;
    }
    if (failed(setDenseIntResult(resultType, outputShape, *valuesOr))) {
      emitOpError("Cast output element type is not supported");
    }
    return;
  }

  if (isa<FloatType>(elementType)) {
    auto valuesOr = getFloatValues(inputAttr);
    if (failed(valuesOr)) {
      emitOpError("Cast input type is not supported");
      return;
    }
    (void)setDenseResult(resultType, outputShape, *valuesOr);
    return;
  }

  auto valuesOr = getIntValues(inputAttr);
  if (failed(valuesOr)) {
    emitOpError("Cast input type is not supported");
    return;
  }
  if (failed(setDenseIntResult(resultType, outputShape, *valuesOr))) {
    emitOpError("Cast output element type is not supported");
  }
}

void StringToHashBucketFastOp::Interpret() {}

void UniqueOp::Interpret() {
  this->inferShape();
  atir::TensorType inputType;
  DenseElementsAttr inputAttr;
  if (failed(getTensorTypeAndData(getOperation(), getX(), "Unique input",
                                  inputType, inputAttr))) {
    return;
  }
  auto inputShape = inputType.getShape();
  int64_t inputSize = getElementCount(inputShape);
  if (inputSize <= 0) {
    emitOpError("Unique input must have positive size");
    return;
  }
  auto valuesResultType = dyn_cast<atir::TensorType>(getY().getType());
  auto idxResultType = dyn_cast<atir::TensorType>(getIdx().getType());
  if (!valuesResultType || !idxResultType) {
    emitOpError("Unique results must be atir::TensorType");
    return;
  }
  SmallVector<int64_t> indexShape(inputShape.begin(), inputShape.end());
  std::vector<int64_t> indices(inputSize, 0);
  if (isa<FloatType>(valuesResultType.getElementType())) {
    auto inputValsOr = getFloatValues(inputAttr);
    if (failed(inputValsOr)) {
      emitOpError("Unique float output requires numeric float-like input");
      return;
    }
    std::vector<float> uniqueValues;
    std::map<float, int64_t> valueToIndex;
    for (int64_t i = 0; i < inputSize; ++i) {
      float v = (*inputValsOr)[i];
      auto it = valueToIndex.find(v);
      if (it == valueToIndex.end()) {
        int64_t newIndex = static_cast<int64_t>(uniqueValues.size());
        valueToIndex[v] = newIndex;
        uniqueValues.push_back(v);
        indices[i] = newIndex;
      } else {
        indices[i] = it->second;
      }
    }
    SmallVector<int64_t> uniqueShape{
        static_cast<int64_t>(uniqueValues.size())};
    (void)setDenseResult(valuesResultType, uniqueShape, uniqueValues);
  } else if (isa<IntegerType, IndexType>(valuesResultType.getElementType())) {
    auto inputValsOr = getIntValues(inputAttr);
    if (failed(inputValsOr)) {
      emitOpError("Unique integer output requires integer-like input");
      return;
    }
    std::vector<int64_t> uniqueValues;
    std::map<int64_t, int64_t> valueToIndex;
    for (int64_t i = 0; i < inputSize; ++i) {
      int64_t v = (*inputValsOr)[i];
      auto it = valueToIndex.find(v);
      if (it == valueToIndex.end()) {
        int64_t newIndex = static_cast<int64_t>(uniqueValues.size());
        valueToIndex[v] = newIndex;
        uniqueValues.push_back(v);
        indices[i] = newIndex;
      } else {
        indices[i] = it->second;
      }
    }
    SmallVector<int64_t> uniqueShape{
        static_cast<int64_t>(uniqueValues.size())};
    if (failed(setDenseIntResult(valuesResultType, uniqueShape, uniqueValues))) {
      emitOpError("Unique values output integer type is not supported");
      return;
    }
  } else {
    emitOpError("Unique values output element type is not supported");
    return;
  }
  if (failed(setDenseIntResult(idxResultType, indexShape, indices))) {
    emitOpError("Unique idx output integer type is not supported");
  }
}

void TopKOp::Interpret() {
  this->inferShape();
  atir::TensorType inputType;
  atir::TensorType kType;
  DenseElementsAttr inputAttr;
  DenseElementsAttr kAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(), "TopK input",
                                  inputType, inputAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getK(), "TopK k", kType,
                                  kAttr))) {
    return;
  }
  auto kValsOr = getIntValues(kAttr);
  if (failed(kValsOr) || kValsOr->size() != 1) {
    emitOpError("TopK k must be integer-like scalar");
    return;
  }
  int64_t kValue = (*kValsOr)[0];
  if (kValue <= 0) {
    emitOpError("TopK k must be > 0");
    return;
  }
  auto inputShape = inputType.getShape();
  if (inputShape.empty()) {
    emitOpError("TopK input rank must be >= 1");
    return;
  }
  int64_t lastDim = inputShape.back();
  int64_t outerCount = 1;
  for (size_t i = 0; i + 1 < inputShape.size(); ++i) {
    if (inputShape[i] <= 0) {
      emitOpError("TopK requires static positive input shape");
      return;
    }
    outerCount *= inputShape[i];
  }
  if (lastDim <= 0) {
    emitOpError("TopK requires static positive last dimension");
    return;
  }
  int64_t outK = std::min(kValue, lastDim);
  if (outK <= 0) {
    emitOpError("TopK effective k must be > 0");
    return;
  }
  auto inputValsOr = getFloatValues(inputAttr);
  if (failed(inputValsOr)) {
    emitOpError("TopK requires numeric input");
    return;
  }
  const auto &inputVals = *inputValsOr;
  if (static_cast<int64_t>(inputVals.size()) != outerCount * lastDim) {
    emitOpError("TopK input data size mismatch");
    return;
  }
  std::vector<float> outValues;
  std::vector<int64_t> outIndices;
  outValues.reserve(static_cast<size_t>(outerCount * outK));
  outIndices.reserve(static_cast<size_t>(outerCount * outK));
  bool sorted = getSorted();
  for (int64_t outer = 0; outer < outerCount; ++outer) {
    std::vector<std::pair<float, int64_t>> row;
    row.reserve(static_cast<size_t>(outK));
    int64_t base = outer * lastDim;
    auto better = [](const std::pair<float, int64_t> &a,
                     const std::pair<float, int64_t> &b) {
      if (a.first != b.first) {
        return a.first > b.first;
      }
      return a.second < b.second;
    };
    auto minHeapCmp = [](const std::pair<float, int64_t> &a,
                         const std::pair<float, int64_t> &b) {
      if (a.first != b.first) {
        return a.first > b.first;
      }
      return a.second < b.second;
    };
    for (int64_t i = 0; i < lastDim; ++i) {
      std::pair<float, int64_t> cand{inputVals[static_cast<size_t>(base + i)], i};
      if (static_cast<int64_t>(row.size()) < outK) {
        row.push_back(cand);
        std::push_heap(row.begin(), row.end(), minHeapCmp);
      } else if (better(cand, row.front())) {
        std::pop_heap(row.begin(), row.end(), minHeapCmp);
        row.back() = cand;
        std::push_heap(row.begin(), row.end(), minHeapCmp);
      }
    }
    if (sorted) {
      std::sort(row.begin(), row.end(), better);
    }
    for (int64_t i = 0; i < outK; ++i) {
      outValues.push_back(row[static_cast<size_t>(i)].first);
      outIndices.push_back(row[static_cast<size_t>(i)].second);
    }
  }
  SmallVector<int64_t> outShape(inputShape.begin(), inputShape.end());
  outShape.back() = outK;
  auto valuesType = dyn_cast<atir::TensorType>(getValues().getType());
  auto indicesType = dyn_cast<atir::TensorType>(getIndices().getType());
  if (!valuesType || !indicesType) {
    emitOpError("TopK results must be atir::TensorType");
    return;
  }
  if (failed(setDenseResult(valuesType, outShape, outValues))) {
    emitOpError("TopK values output type is not supported");
    return;
  }
  if (isa<FloatType>(indicesType.getElementType())) {
    std::vector<float> floatIndices;
    floatIndices.reserve(outIndices.size());
    for (int64_t v : outIndices) {
      floatIndices.push_back(static_cast<float>(v));
    }
    if (failed(setDenseResult(indicesType, outShape, floatIndices))) {
      emitOpError("TopK indices output float type is not supported");
    }
    return;
  }
  if (failed(setDenseIntResult(indicesType, outShape, outIndices))) {
    emitOpError("TopK indices output integer type is not supported");
  }
}

void UnsortedSegmentMinOp::Interpret() {
  this->inferShape();
  atir::TensorType inputType;
  atir::TensorType segmentIdsType;
  atir::TensorType numSegmentsType;
  DenseElementsAttr inputAttr;
  DenseElementsAttr segmentIdsAttr;
  DenseElementsAttr numSegmentsAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(),
                                  "UnsortedSegmentMin input", inputType,
                                  inputAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getSegmentIds(),
                                  "UnsortedSegmentMin segmentIds",
                                  segmentIdsType, segmentIdsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getNumSegments(),
                                  "UnsortedSegmentMin numSegments",
                                  numSegmentsType, numSegmentsAttr))) {
    return;
  }
  auto inputValsOr = getFloatValues(inputAttr);
  auto segmentValsOr = getIntValues(segmentIdsAttr);
  auto numSegmentsValsOr = getIntValues(numSegmentsAttr);
  if (failed(inputValsOr) || failed(segmentValsOr) || failed(numSegmentsValsOr) ||
      numSegmentsValsOr->size() != 1) {
    emitOpError("UnsortedSegmentMin requires numeric input, integer-like segmentIds and scalar numSegments");
    return;
  }
  int64_t numSegments = (*numSegmentsValsOr)[0];
  if (numSegments < 0) {
    emitOpError("UnsortedSegmentMin numSegments must be >= 0");
    return;
  }
  auto inputShape = inputType.getShape();
  if (inputShape.empty()) {
    emitOpError("UnsortedSegmentMin input rank must be >= 1");
    return;
  }
  int64_t rows = inputShape[0];
  int64_t rowWidth = getElementCount(ArrayRef<int64_t>(inputShape).drop_front());
  if (rows < 0 || rowWidth < 1) {
    emitOpError("UnsortedSegmentMin only supports static non-empty trailing shape");
    return;
  }
  if (static_cast<int64_t>(segmentValsOr->size()) != rows) {
    emitOpError("UnsortedSegmentMin segmentIds size must equal input first dimension");
    return;
  }
  SmallVector<int64_t> outputShape;
  outputShape.push_back(numSegments);
  outputShape.append(inputShape.begin() + 1, inputShape.end());
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("UnsortedSegmentMin output shape must be static");
    return;
  }
  constexpr float kInitMax = std::numeric_limits<float>::max();
  std::vector<float> result(static_cast<size_t>(outputSize), kInitMax);
  for (int64_t row = 0; row < rows; ++row) {
    int64_t seg = (*segmentValsOr)[static_cast<size_t>(row)];
    if (seg < 0 || seg >= numSegments) {
      continue;
    }
    int64_t srcBase = row * rowWidth;
    int64_t dstBase = seg * rowWidth;
    for (int64_t j = 0; j < rowWidth; ++j) {
      float v = (*inputValsOr)[static_cast<size_t>(srcBase + j)];
      float &d = result[static_cast<size_t>(dstBase + j)];
      d = std::min(d, v);
    }
  }
  auto resultType = getResult().getType();
  if (isa<FloatType>(resultType.getElementType())) {
    (void)setDenseResult(resultType, outputShape, result);
    return;
  }
  std::vector<int64_t> intResult;
  intResult.reserve(result.size());
  for (float v : result) {
    intResult.push_back(static_cast<int64_t>(v));
  }
  if (failed(setDenseIntResult(resultType, outputShape, intResult))) {
    emitOpError("UnsortedSegmentMin output element type is not supported");
  }
}

void TensorScatterUpdateOp::Interpret() {
  this->inferShape();
  atir::TensorType inputType;
  atir::TensorType indicesType;
  atir::TensorType updatesType;
  DenseElementsAttr inputAttr;
  DenseElementsAttr indicesAttr;
  DenseElementsAttr updatesAttr;
  if (failed(getTensorTypeAndData(getOperation(), getInput(),
                                  "TensorScatterUpdate input", inputType,
                                  inputAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getIndices(),
                                  "TensorScatterUpdate indices", indicesType,
                                  indicesAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getUpdates(),
                                  "TensorScatterUpdate updates", updatesType,
                                  updatesAttr))) {
    return;
  }
  auto inputValsOr = getFloatValues(inputAttr);
  auto indicesValsOr = getIntValues(indicesAttr);
  auto updatesValsOr = getFloatValues(updatesAttr);
  if (failed(inputValsOr) || failed(indicesValsOr) || failed(updatesValsOr)) {
    emitOpError("TensorScatterUpdate requires numeric input/updates and integer-like indices");
    return;
  }
  auto inputShape = inputType.getShape();
  auto indicesShape = indicesType.getShape();
  auto updatesShape = updatesType.getShape();
  if (inputShape.empty() || indicesShape.size() < 2) {
    emitOpError("TensorScatterUpdate expects input rank >=1 and indices rank >=2");
    return;
  }
  int64_t indexDepth = indicesShape.back();
  int64_t inputRank = static_cast<int64_t>(inputShape.size());
  if (indexDepth <= 0 || indexDepth > inputRank) {
    emitOpError("TensorScatterUpdate indices last dim must be in [1, input_rank]");
    return;
  }
  int64_t numUpdates = getElementCount(ArrayRef<int64_t>(indicesShape).drop_back());
  if (numUpdates < 0) {
    emitOpError("TensorScatterUpdate indices prefix shape must be static");
    return;
  }
  SmallVector<int64_t> expectedUpdatesShape(indicesShape.begin(),
                                            indicesShape.end() - 1);
  expectedUpdatesShape.append(inputShape.begin() + indexDepth, inputShape.end());
  if (SmallVector<int64_t>(updatesShape.begin(), updatesShape.end()) !=
      expectedUpdatesShape) {
    emitOpError("TensorScatterUpdate updates shape mismatch");
    return;
  }
  int64_t sliceSize = getElementCount(ArrayRef<int64_t>(inputShape).drop_front(indexDepth));
  if (sliceSize < 1) {
    emitOpError("TensorScatterUpdate only supports static slice size");
    return;
  }
  if (static_cast<int64_t>(indicesValsOr->size()) != numUpdates * indexDepth ||
      static_cast<int64_t>(updatesValsOr->size()) != numUpdates * sliceSize) {
    emitOpError("TensorScatterUpdate data size mismatch");
    return;
  }
  std::vector<float> result = *inputValsOr;
  for (int64_t u = 0; u < numUpdates; ++u) {
    SmallVector<int64_t> targetPrefix;
    targetPrefix.reserve(static_cast<size_t>(indexDepth));
    for (int64_t d = 0; d < indexDepth; ++d) {
      int64_t idx = (*indicesValsOr)[static_cast<size_t>(u * indexDepth + d)];
      if (idx < 0 || idx >= inputShape[static_cast<size_t>(d)]) {
        emitOpError("TensorScatterUpdate index out of range");
        return;
      }
      targetPrefix.push_back(idx);
    }
    ArrayRef<int64_t> prefixShape = ArrayRef<int64_t>(inputShape).take_front(
        static_cast<size_t>(indexDepth));
    int64_t dstBase = getFlatIndex(prefixShape, targetPrefix) * sliceSize;
    int64_t srcBase = u * sliceSize;
    for (int64_t j = 0; j < sliceSize; ++j) {
      result[static_cast<size_t>(dstBase + j)] =
          (*updatesValsOr)[static_cast<size_t>(srcBase + j)];
    }
  }
  auto resultType = getResult().getType();
  if (isa<FloatType>(resultType.getElementType())) {
    (void)setDenseResult(resultType, inputShape, result);
    return;
  }
  std::vector<int64_t> intResult;
  intResult.reserve(result.size());
  for (float v : result) {
    intResult.push_back(static_cast<int64_t>(v));
  }
  if (failed(setDenseIntResult(resultType, inputShape, intResult))) {
    emitOpError("TensorScatterUpdate output element type is not supported");
  }
}

void ResourceGatherOp::Interpret() {
  this->inferShape();
  if (getBatchDims() != 0) {
    emitOpError("ResourceGather currently only supports batchDims == 0");
    return;
  }
  atir::TensorType resourceType;
  DenseElementsAttr resourceAttr;
  if (failed(getTensorTypeAndData(getOperation(), getResource(),
                                  "ResourceGather resource", resourceType,
                                  resourceAttr))) {
    return;
  }
  auto resourceShape = resourceType.getShape();
  if (resourceShape.empty()) {
    emitOpError("ResourceGather resource must have rank >= 1");
    return;
  }
  if (getIndices().empty()) {
    emitOpError("ResourceGather expects at least one indices operand");
    return;
  }
  if (getIndices().size() != 1) {
    emitOpError("ResourceGather currently only supports a single indices tensor operand");
    return;
  }
  Value indicesVal = getIndices().front();
  atir::TensorType indicesType;
  DenseElementsAttr indicesAttr;
  if (failed(getTensorTypeAndData(getOperation(), indicesVal,
                                  "ResourceGather indices", indicesType,
                                  indicesAttr))) {
    return;
  }
  auto indicesValsOr = getIntValues(indicesAttr);
  if (failed(indicesValsOr)) {
    emitOpError("ResourceGather indices must be integer-like");
    return;
  }
  auto indicesShape = indicesType.getShape();
  int64_t indicesSize = getElementCount(indicesShape);
  if (indicesSize < 0) {
    emitOpError("ResourceGather indices shape must be static for interpretation");
    return;
  }
  if (static_cast<int64_t>(indicesValsOr->size()) != indicesSize) {
    emitOpError("ResourceGather indices data size mismatch");
    return;
  }
  SmallVector<int64_t> tailShape(resourceShape.begin() + 1, resourceShape.end());
  int64_t tailSize = getElementCount(tailShape);
  if (tailSize < 0) {
    emitOpError("ResourceGather resource tail shape must be static for interpretation");
    return;
  }
  // output.shape = indices.shape + resource.shape[1:]
  SmallVector<int64_t> outShape;
  outShape.append(indicesShape.begin(), indicesShape.end());
  outShape.append(tailShape.begin(), tailShape.end());
  int64_t outputSize = getElementCount(outShape);
  if (outputSize < 0) {
    emitOpError("ResourceGather output shape must be static for interpretation");
    return;
  }
  int64_t dim0 = resourceShape.front();
  if (dim0 < 0) {
    emitOpError("ResourceGather resource first dimension must be static");
    return;
  }
  auto outType = getResult().getType();
  if (isa<FloatType>(outType.getElementType())) {
    auto paramsValsOr = getFloatValues(resourceAttr);
    if (failed(paramsValsOr)) {
      emitOpError("ResourceGather resource requires float-like cacheData");
      return;
    }
    const auto &paramsVals = *paramsValsOr;
    if (static_cast<int64_t>(paramsVals.size()) != getElementCount(resourceShape)) {
      emitOpError("ResourceGather resource data size mismatch");
      return;
    }
    std::vector<float> outVals(static_cast<size_t>(outputSize), 0.0f);
    for (int64_t i = 0; i < indicesSize; ++i) {
      int64_t idx = (*indicesValsOr)[static_cast<size_t>(i)];
      if (idx < 0 || idx >= dim0) {
        emitOpError("ResourceGather index out of range");
        return;
      }
      int64_t srcBase = idx * tailSize;
      int64_t dstBase = i * tailSize;
      for (int64_t j = 0; j < tailSize; ++j) {
        outVals[static_cast<size_t>(dstBase + j)] =
            paramsVals[static_cast<size_t>(srcBase + j)];
      }
    }
    (void)setDenseResult(outType, outShape, outVals);
    return;
  }
  auto paramsValsOr = getIntValues(resourceAttr);
  if (failed(paramsValsOr)) {
    emitOpError("ResourceGather resource requires integer-like cacheData");
    return;
  }
  const auto &paramsVals = *paramsValsOr;
  std::vector<int64_t> outVals(static_cast<size_t>(outputSize), 0);
  for (int64_t i = 0; i < indicesSize; ++i) {
    int64_t idx = (*indicesValsOr)[static_cast<size_t>(i)];
    if (idx < 0 || idx >= dim0) {
      emitOpError("ResourceGather index out of range");
      return;
    }
    int64_t srcBase = idx * tailSize;
    int64_t dstBase = i * tailSize;
    for (int64_t j = 0; j < tailSize; ++j) {
      outVals[static_cast<size_t>(dstBase + j)] =
          paramsVals[static_cast<size_t>(srcBase + j)];
    }
  }
  if (failed(setDenseIntResult(outType, outShape, outVals))) {
    emitOpError("ResourceGather output element type is not supported");
  }
}
}  // namespace atir
