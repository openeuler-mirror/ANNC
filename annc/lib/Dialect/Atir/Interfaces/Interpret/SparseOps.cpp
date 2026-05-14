#include "Common.h"
#include "Dialect/Atir/AtirOps.h"

using namespace atir::interpret;

namespace atir {

enum class SparseSegmentMode { Sum, Min, Mean };

static LogicalResult interpretSparseSegment(Operation *op,
    Value dataVal, Value indicesVal, Value segmentIdsVal,
    Value numSegmentsVal, SparseSegmentMode mode) {
  atir::TensorType dataType, indicesType, segmentIdsType, numSegType;
  DenseElementsAttr dataAttr, indicesAttr, segmentIdsAttr, numSegAttr;
  if (failed(getTensorTypeAndData(op, dataVal, "data", dataType, dataAttr)) ||
      failed(getTensorTypeAndData(op, indicesVal, "indices", indicesType, indicesAttr)) ||
      failed(getTensorTypeAndData(op, segmentIdsVal, "segmentIds", segmentIdsType, segmentIdsAttr)) ||
      failed(getTensorTypeAndData(op, numSegmentsVal, "numSegments", numSegType, numSegAttr)))
    return failure();
  if (numSegAttr.getNumElements() != 1) {
    op->emitOpError() << "numSegments must be a scalar (single element)";
    return failure();
  }
  auto dataValsOr = getFloatValues(dataAttr);
  auto indicesValsOr = getIntValues(indicesAttr);
  auto segmentValsOr = getIntValues(segmentIdsAttr);
  if (failed(dataValsOr) || failed(indicesValsOr) || failed(segmentValsOr)) {
    op->emitOpError() << "requires numeric data and integer-like indices/segmentIds";
    return failure();
  }
  const auto &dataVals = *dataValsOr;
  const auto &indicesVals = *indicesValsOr;
  const auto &segmentVals = *segmentValsOr;
  if (indicesVals.size() != segmentVals.size()) {
    op->emitOpError() << "indices and segmentIds must have same length";
    return failure();
  }
  auto dataShape = dataType.getShape();
  if (dataShape.empty()) {
    op->emitOpError() << "data rank must be >= 1";
    return failure();
  }
  int64_t rows = dataShape[0];
  int64_t rowWidth = getElementCount(ArrayRef<int64_t>(dataShape).drop_front());
  if (rows < 0 || rowWidth < 1) {
    op->emitOpError() << "only supports static non-empty trailing shape";
    return failure();
  }
  auto resultType = cast<atir::TensorType>(op->getResult(0).getType());
  auto outputShape = resultType.getShape();
  if (outputShape.empty() || outputShape[0] < 0) {
    op->emitOpError() << "output shape must be static";
    return failure();
  }
  int64_t numSegments = outputShape[0];
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    op->emitOpError() << "output shape must be static";
    return failure();
  }
  std::vector<float> result;
  if (mode == SparseSegmentMode::Min)
    result.assign(static_cast<size_t>(outputSize), std::numeric_limits<float>::max());
  else
    result.assign(static_cast<size_t>(outputSize), 0.0f);
  std::vector<int64_t> counts;
  if (mode == SparseSegmentMode::Mean)
    counts.assign(static_cast<size_t>(numSegments), 0);
  for (size_t i = 0; i < indicesVals.size(); ++i) {
    int64_t dataRow = indicesVals[i];
    int64_t segmentId = segmentVals[i];
    if (dataRow < 0 || dataRow >= rows || segmentId < 0 || segmentId >= numSegments) {
      op->emitOpError() << "has out-of-range index/segment_id";
      return failure();
    }
    int64_t srcBase = dataRow * rowWidth;
    int64_t dstBase = segmentId * rowWidth;
    for (int64_t j = 0; j < rowWidth; ++j) {
      if (mode == SparseSegmentMode::Min)
        result[static_cast<size_t>(dstBase + j)] = std::min(result[static_cast<size_t>(dstBase + j)], dataVals[static_cast<size_t>(srcBase + j)]);
      else
        result[static_cast<size_t>(dstBase + j)] += dataVals[static_cast<size_t>(srcBase + j)];
    }
    if (mode == SparseSegmentMode::Mean)
      counts[static_cast<size_t>(segmentId)] += 1;
  }
  if (mode == SparseSegmentMode::Mean) {
    for (int64_t s = 0; s < numSegments; ++s) {
      int64_t cnt = counts[static_cast<size_t>(s)];
      if (cnt == 0) continue;
      int64_t base = s * rowWidth;
      for (int64_t j = 0; j < rowWidth; ++j)
        result[static_cast<size_t>(base + j)] /= static_cast<float>(cnt);
    }
  }
  (void)setDenseResult(resultType, outputShape, result);
  return success();
}
void SparseToDenseOp::Interpret() {
  this->inferShape();
  atir::TensorType indicesType;
  atir::TensorType outputShapeType;
  atir::TensorType sparseValueType;
  atir::TensorType defaultValueType;
  DenseElementsAttr indicesAttr;
  DenseElementsAttr outputShapeAttr;
  DenseElementsAttr sparseValueAttr;
  DenseElementsAttr defaultValueAttr;
  if (failed(getTensorTypeAndData(getOperation(), getSparseIndice(),
                                  "SparseToDense sparseIndice", indicesType,
                                  indicesAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getOutputShape(),
                                  "SparseToDense outputShape", outputShapeType,
                                  outputShapeAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getSparseValue(),
                                  "SparseToDense sparseValue", sparseValueType,
                                  sparseValueAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getDefaultValue(),
                                  "SparseToDense defaultValue", defaultValueType,
                                  defaultValueAttr))) {
    return;
  }
  auto indicesValsOr = getIntValues(indicesAttr);
  auto denseShapeOr = getIntValues(outputShapeAttr);
  auto sparseValsOr = getFloatValues(sparseValueAttr);
  auto defaultValOr = getFloatValues(defaultValueAttr);
  if (failed(indicesValsOr) || failed(denseShapeOr) || failed(sparseValsOr) ||
      failed(defaultValOr)) {
    emitOpError("SparseToDense expects numeric values and integer-like indices/shape");
    return;
  }
  auto indicesShape = indicesType.getShape();
  if (indicesShape.size() != 2) {
    emitOpError("SparseToDense currently expects sparseIndice rank-2");
    return;
  }
  int64_t n = indicesShape[0];
  int64_t rank = indicesShape[1];
  if (rank <= 0 || static_cast<int64_t>(denseShapeOr->size()) != rank) {
    emitOpError("SparseToDense outputShape length must match indices width");
    return;
  }
  if (static_cast<int64_t>(indicesValsOr->size()) != n * rank) {
    emitOpError("SparseToDense sparseIndice element count mismatch");
    return;
  }
  int64_t denseSize = 1;
  for (int64_t d : *denseShapeOr) {
    if (d <= 0) {
      emitOpError("SparseToDense outputShape dims must be positive");
      return;
    }
    denseSize *= d;
  }
  const auto &sv = *sparseValsOr;
  bool valueScalar = (sv.size() == 1);
  if (!valueScalar && static_cast<int64_t>(sv.size()) != n) {
    emitOpError("SparseToDense sparseValue must be scalar or length N");
    return;
  }
  if (defaultValOr->size() != 1) {
    emitOpError("SparseToDense defaultValue must be scalar");
    return;
  }
  std::vector<float> result(static_cast<size_t>(denseSize), (*defaultValOr)[0]);
  for (int64_t i = 0; i < n; ++i) {
    int64_t flat = 0;
    for (int64_t a = 0; a < rank; ++a) {
      int64_t c = (*indicesValsOr)[i * rank + a];
      int64_t d = (*denseShapeOr)[a];
      if (c < 0 || c >= d) {
        emitOpError("SparseToDense index out of range");
        return;
      }
      flat = flat * d + c;
    }
    result[flat] = valueScalar ? sv[0] : sv[static_cast<size_t>(i)];
  }
  auto outType = getResult().getType();
  if (isa<FloatType>(outType.getElementType())) {
    (void)setDenseResult(outType, *denseShapeOr, result);
    return;
  }
  std::vector<int64_t> intResult;
  intResult.reserve(result.size());
  for (float v : result) {
    intResult.push_back(static_cast<int64_t>(v));
  }
  if (failed(setDenseIntResult(outType, *denseShapeOr, intResult))) {
    emitOpError("SparseToDense output element type is not supported");
  }
}

void SparseReshapeOp::Interpret() {
  this->inferShape();
  atir::TensorType indicesType;
  atir::TensorType inputShapeType;
  atir::TensorType newShapeType;
  DenseElementsAttr indicesAttr;
  DenseElementsAttr inputShapeAttr;
  DenseElementsAttr newShapeAttr;
  if (failed(getTensorTypeAndData(getOperation(), getIndices(),
                                  "SparseReshape indices", indicesType,
                                  indicesAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getInputShape(),
                                  "SparseReshape inputShape", inputShapeType,
                                  inputShapeAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getNewShape(),
                                  "SparseReshape newShape", newShapeType,
                                  newShapeAttr))) {
    return;
  }
  auto indicesValsOr = getIntValues(indicesAttr);
  auto inputShapeValsOr = getIntValues(inputShapeAttr);
  auto newShapeValsOr = getIntValues(newShapeAttr);
  if (failed(indicesValsOr) || failed(inputShapeValsOr) || failed(newShapeValsOr)) {
    emitOpError("SparseReshape requires integer-like indices/inputShape/newShape");
    return;
  }
  auto indicesShape = indicesType.getShape();
  if (indicesShape.size() != 2) {
    emitOpError("SparseReshape indices must be rank-2");
    return;
  }
  int64_t nnz = indicesShape[0];
  int64_t inRank = indicesShape[1];
  if (inRank <= 0 || static_cast<int64_t>(inputShapeValsOr->size()) != inRank) {
    emitOpError("SparseReshape inputShape size must equal indices second dim");
    return;
  }
  std::vector<int64_t> outDenseShape(newShapeValsOr->begin(), newShapeValsOr->end());
  int64_t unknownAxis = -1;
  int64_t knownProduct = 1;
  for (int64_t i = 0; i < static_cast<int64_t>(outDenseShape.size()); ++i) {
    int64_t d = outDenseShape[i];
    if (d == -1) {
      if (unknownAxis != -1) {
        emitOpError("SparseReshape newShape can contain at most one -1");
        return;
      }
      unknownAxis = i;
      continue;
    }
    if (d <= 0) {
      emitOpError("SparseReshape newShape dims must be positive or -1");
      return;
    }
    knownProduct *= d;
  }
  int64_t inProduct = 1;
  for (int64_t d : *inputShapeValsOr) {
    if (d <= 0) {
      emitOpError("SparseReshape inputShape dims must be positive");
      return;
    }
    inProduct *= d;
  }
  if (unknownAxis != -1) {
    if (knownProduct == 0 || inProduct % knownProduct != 0) {
      emitOpError("SparseReshape cannot infer -1 dimension");
      return;
    }
    outDenseShape[unknownAxis] = inProduct / knownProduct;
  } else if (knownProduct != inProduct) {
    emitOpError("SparseReshape element count mismatch between input and new shape");
    return;
  }
  int64_t outRank = static_cast<int64_t>(outDenseShape.size());
  std::vector<int64_t> outIndices;
  outIndices.resize(static_cast<size_t>(nnz * outRank));
  for (int64_t n = 0; n < nnz; ++n) {
    int64_t linear = 0;
    for (int64_t axis = 0; axis < inRank; ++axis) {
      int64_t coord = (*indicesValsOr)[n * inRank + axis];
      int64_t dim = (*inputShapeValsOr)[axis];
      if (coord < 0 || coord >= dim) {
        emitOpError("SparseReshape indices coordinate out of range");
        return;
      }
      linear = linear * dim + coord;
    }
    for (int64_t axis = outRank - 1; axis >= 0; --axis) {
      int64_t dim = outDenseShape[axis];
      outIndices[n * outRank + axis] = linear % dim;
      linear /= dim;
    }
  }
  auto outIndicesType = getOutputIndices().getType();
  auto outShapeType = getOutputShape().getType();
  SmallVector<int64_t> outIndicesShape = {nnz, outRank};
  SmallVector<int64_t> outShapeShape = {outRank};
  auto setIndexLikeResult = [&](atir::TensorType ty, ArrayRef<int64_t> shape,
                                ArrayRef<int64_t> values) -> LogicalResult {
    if (isa<FloatType>(ty.getElementType())) {
      std::vector<float> casted;
      casted.reserve(values.size());
      for (int64_t v : values) {
        casted.push_back(static_cast<float>(v));
      }
      return setDenseResult(ty, shape, casted);
    }
    return setDenseIntResult(ty, shape, values);
  };
  if (failed(setIndexLikeResult(outIndicesType, outIndicesShape, outIndices)) ||
      failed(setIndexLikeResult(outShapeType, outShapeShape, outDenseShape))) {
    emitOpError("SparseReshape output element type is not supported");
  }
}

void SparseFillEmptyRowsOp::Interpret() {
  this->inferShape();
  atir::TensorType indicesType;
  atir::TensorType valuesType;
  atir::TensorType denseShapeType;
  atir::TensorType defaultValueType;
  DenseElementsAttr indicesAttr;
  DenseElementsAttr valuesAttr;
  DenseElementsAttr denseShapeAttr;
  DenseElementsAttr defaultValueAttr;
  if (failed(getTensorTypeAndData(getOperation(), getIndices(),
                                  "SparseFillEmptyRows indices", indicesType,
                                  indicesAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getValues(),
                                  "SparseFillEmptyRows values", valuesType,
                                  valuesAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getDenseShape(),
                                  "SparseFillEmptyRows denseShape",
                                  denseShapeType, denseShapeAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getDefaultValue(),
                                  "SparseFillEmptyRows defaultValue",
                                  defaultValueType, defaultValueAttr))) {
    return;
  }
  auto indicesValsOr = getIntValues(indicesAttr);
  auto denseShapeValsOr = getIntValues(denseShapeAttr);
  if (failed(indicesValsOr) || failed(denseShapeValsOr)) {
    emitOpError("SparseFillEmptyRows requires integer-like indices and dense_shape");
    return;
  }
  auto indicesShape = indicesType.getShape();
  if (indicesShape.size() != 2) {
    emitOpError("SparseFillEmptyRows expects indices rank-2");
    return;
  }
  int64_t nnz = indicesShape[0];
  int64_t idxRank = indicesShape[1];
  int64_t denseRank = denseShapeValsOr->size();
  if (denseRank != idxRank) {
    emitOpError("SparseFillEmptyRows dense_shape rank must match indices width");
    return;
  }
  if (denseRank <= 0) {
    emitOpError("SparseFillEmptyRows dense_shape rank must be positive");
    return;
  }
  int64_t numRows = (*denseShapeValsOr)[0];
  if (numRows <= 0) {
    emitOpError("SparseFillEmptyRows dense_shape[0] must be positive");
    return;
  }
  bool valuesAreFloat = isa<FloatType>(valuesType.getElementType());
  if (!valuesAreFloat && !isa<IntegerType, IndexType>(valuesType.getElementType())) {
    emitOpError("SparseFillEmptyRows values must be float or integer-like");
    return;
  }
  auto valuesValsOr = getFloatValues(valuesAttr);
  auto defaultValsOr = getFloatValues(defaultValueAttr);
  if (failed(valuesValsOr) || failed(defaultValsOr) || defaultValsOr->empty()) {
    emitOpError("SparseFillEmptyRows requires cacheData for values/defaultValue");
    return;
  }
  std::vector<float> inValues = *valuesValsOr;
  float defaultVal = (*defaultValsOr)[0];
  if (static_cast<int64_t>(inValues.size()) != nnz) {
    emitOpError("SparseFillEmptyRows values element count mismatch");
    return;
  }
  std::vector<std::vector<int64_t>> rowEntries(numRows);
  for (int64_t j = 0; j < nnz; ++j) {
    int64_t row = (*indicesValsOr)[j * idxRank + 0];
    if (row < 0 || row >= numRows) {
      emitOpError("SparseFillEmptyRows indices row out of range");
      return;
    }
    rowEntries[static_cast<size_t>(row)].push_back(j);
  }
  std::vector<int64_t> emptyRowIndicator(static_cast<size_t>(numRows), 0);
  int64_t numEmptyRows = 0;
  for (int64_t r = 0; r < numRows; ++r) {
    if (rowEntries[static_cast<size_t>(r)].empty()) {
      emptyRowIndicator[static_cast<size_t>(r)] = 1;
      ++numEmptyRows;
    }
  }
  int64_t outNnz = nnz + numEmptyRows;
  std::vector<int64_t> outIndices;
  outIndices.resize(static_cast<size_t>(outNnz) * idxRank);
  std::vector<float> outValues(static_cast<size_t>(outNnz));
  std::vector<int64_t> reverseIndexMap;
  reverseIndexMap.resize(nnz, -1);
  auto getIndexCoord = [&](int64_t entryIdx, int64_t axis) -> int64_t {
    return (*indicesValsOr)[entryIdx * idxRank + axis];
  };
  int64_t outPos = 0;
  for (int64_t r = 0; r < numRows; ++r) {
    auto &entries = rowEntries[static_cast<size_t>(r)];
    if (entries.empty()) {
      outIndices[outPos * idxRank + 0] = r;
      for (int64_t axis = 1; axis < idxRank; ++axis) {
        outIndices[outPos * idxRank + axis] = 0;
      }
      outValues[outPos] = defaultVal;
      ++outPos;
      continue;
    }
    std::stable_sort(entries.begin(), entries.end(),
                     [&](int64_t a, int64_t b) {
                       for (int64_t axis = 1; axis < idxRank; ++axis) {
                         int64_t ca = getIndexCoord(a, axis);
                         int64_t cb = getIndexCoord(b, axis);
                         if (ca != cb)
                           return ca < cb;
                       }
                       return a < b;
                     });
    for (int64_t entryIdx : entries) {
      outIndices[outPos * idxRank + 0] = r;
      for (int64_t axis = 1; axis < idxRank; ++axis) {
        outIndices[outPos * idxRank + axis] = getIndexCoord(entryIdx, axis);
      }
      outValues[outPos] = inValues[entryIdx];
      reverseIndexMap[entryIdx] = outPos;
      ++outPos;
    }
  }
  if (outPos != outNnz) {
    emitOpError("SparseFillEmptyRows internal error: output nnz mismatch");
    return;
  }
  auto outputIndicesType = getOutputIndices().getType();
  auto outputValuesType = getOutputValues().getType();
  auto emptyRowIndicatorType = getEmptyRowIndicator().getType();
  auto reverseIndexMapType = getReverseIndexMap().getType();
  SmallVector<int64_t> outputIndicesShape{outNnz, idxRank};
  SmallVector<int64_t> outputValuesShape{outNnz};
  SmallVector<int64_t> emptyRowShape{numRows};
  SmallVector<int64_t> reverseIndexShape{nnz};
  if (failed(setDenseIntResult(outputIndicesType, outputIndicesShape, outIndices))) {
    emitOpError("SparseFillEmptyRows outputIndices element type is not supported");
    return;
  }
  if (failed(setDenseResult(outputValuesType, outputValuesShape, outValues))) {
    emitOpError("SparseFillEmptyRows outputValues element type is not supported");
    return;
  }
  if (failed(setBooleanLikeResult(emptyRowIndicatorType, emptyRowShape, emptyRowIndicator))) {
    emitOpError("SparseFillEmptyRows emptyRowIndicator element type is not supported");
    return;
  }
  if (failed(setDenseIntResult(reverseIndexMapType, reverseIndexShape, reverseIndexMap))) {
    emitOpError("SparseFillEmptyRows reverseIndexMap element type is not supported");
    return;
  }
}

void SparseSegmentSumOp::Interpret() {
  this->inferShape();
  (void)interpretSparseSegment(getOperation(),
      getInput(), getIndices(), getSegmentIds(), getNumSegments(), SparseSegmentMode::Sum);
}

void SparseSegmentMinOp::Interpret() {
  this->inferShape();
  (void)interpretSparseSegment(getOperation(),
      getInput(), getIndices(), getSegmentIds(), getNumSegments(), SparseSegmentMode::Min);
}

void SparseSegmentMeanOp::Interpret() {
  this->inferShape();
  (void)interpretSparseSegment(getOperation(),
      getData(), getIndices(), getSegmentIds(), getNumSegments(), SparseSegmentMode::Mean);
}
} // namespace atir