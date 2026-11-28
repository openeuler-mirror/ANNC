#include "Common.h"
#include "Dialect/Atir/AtirOps.h"

using namespace atir::interpret;

namespace atir {

void MergeOp::Interpret() {
  this->inferShape();
  auto inputs = getInputsAndControl();
  if (inputs.empty()) {
    emitOpError("Merge expects at least one input");
    return;
  }

  int64_t selectedIndex = -1;
  atir::TensorType selectedType;
  DenseElementsAttr selectedAttr;
  for (auto it : llvm::enumerate(inputs)) {
    auto inputType = dyn_cast<atir::TensorType>(it.value().getType());
    if (!inputType || !inputType.getCacheData()) {
      continue;
    }
    selectedIndex = static_cast<int64_t>(it.index());
    selectedType = inputType;
    selectedAttr = inputType.getCacheData();
    break;
  }
  if (selectedIndex < 0) {
    emitOpError("Merge requires at least one input with cacheData");
    return;
  }

  auto outputType = dyn_cast<atir::TensorType>(getOutput().getType());
  auto valueIndexType = dyn_cast<atir::TensorType>(getValueIndex().getType());
  if (!outputType || !valueIndexType) {
    emitOpError("Merge results must be atir::TensorType");
    return;
  }

  auto outputShape = outputType.getShape();
  if (isa<FloatType>(outputType.getElementType())) {
    auto valsOr = getFloatValues(selectedAttr);
    if (failed(valsOr)) {
      emitOpError("Merge float output requires numeric input");
      return;
    }
    if (failed(setDenseResult(outputType, outputShape, *valsOr))) {
      emitOpError("Merge output float type is not supported");
      return;
    }
  } else {
    auto valsOr = getIntValues(selectedAttr);
    if (failed(valsOr)) {
      emitOpError("Merge integer output requires integer-like input");
      return;
    }
    if (failed(setDenseIntResult(outputType, outputShape, *valsOr))) {
      emitOpError("Merge output integer type is not supported");
      return;
    }
  }

  if (isa<FloatType>(valueIndexType.getElementType())) {
    std::vector<float> idxVal = {static_cast<float>(selectedIndex)};
    if (failed(setDenseResult(valueIndexType, {}, idxVal))) {
      emitOpError("Merge value_index float type is not supported");
    }
    return;
  }
  if (failed(setDenseIntResult(valueIndexType, {}, {selectedIndex}))) {
    emitOpError("Merge value_index type is not supported");
  }
}

void DynamicPartitionOp::Interpret() {
  this->inferShape();

  atir::TensorType dataType;
  atir::TensorType partitionsType;
  DenseElementsAttr dataAttr;
  DenseElementsAttr partitionsAttr;
  if (failed(getTensorTypeAndData(getOperation(), getData(), "DynamicPartition data",
                                  dataType, dataAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getPartitions(),
                                  "DynamicPartition partitions", partitionsType,
                                  partitionsAttr))) {
    return;
  }

  auto dataShape = dataType.getShape();
  auto partitionsShape = partitionsType.getShape();
  int64_t dataSize = getElementCount(dataShape);
  int64_t partitionsSize = getElementCount(partitionsShape);
  if (dataSize < 0 || partitionsSize < 0) {
    emitOpError("DynamicPartition requires static input shapes");
    return;
  }

  int64_t numPartitions = static_cast<int64_t>(getNumPartitions());
  if (numPartitions <= 0 || static_cast<size_t>(numPartitions) != getOutputs().size()) {
    emitOpError("DynamicPartition numPartitions must match output count");
    return;
  }

  if (dataShape.size() < partitionsShape.size()) {
    emitOpError("DynamicPartition data rank must be >= partitions rank");
    return;
  }
  int64_t innerSize = 1;
  for (size_t i = partitionsShape.size(); i < dataShape.size(); ++i) {
    innerSize *= dataShape[i];
  }
  if (partitionsSize * innerSize != dataSize) {
    emitOpError("DynamicPartition data size mismatch");
    return;
  }

  auto partValsOr = getIntValues(partitionsAttr);
  if (failed(partValsOr)) {
    emitOpError("DynamicPartition partitions must be integer-like");
    return;
  }

  Type dataElemTy = dataType.getElementType();
  if (isa<FloatType>(dataElemTy)) {
    auto dataValsOr = getFloatValues(dataAttr);
    if (failed(dataValsOr)) {
      emitOpError("DynamicPartition data values not available");
      return;
    }

    std::vector<std::vector<float>> buckets(static_cast<size_t>(numPartitions));
    for (int64_t i = 0; i < partitionsSize; ++i) {
      int64_t p = (*partValsOr)[i];
      if (p < 0 || p >= numPartitions) {
        emitOpError("DynamicPartition partition id out of range");
        return;
      }
      int64_t base = i * innerSize;
      for (int64_t j = 0; j < innerSize; ++j) {
        buckets[static_cast<size_t>(p)].push_back((*dataValsOr)[base + j]);
      }
    }

    for (auto [i, outVal] : llvm::enumerate(getOutputs())) {
      auto outType = dyn_cast<atir::TensorType>(outVal.getType());
      if (!outType) {
        emitOpError("DynamicPartition output must be atir::TensorType");
        return;
      }
      SmallVector<int64_t> outShape;
      outShape.push_back(static_cast<int64_t>(buckets[i].size() / innerSize));
      outShape.append(dataShape.begin() + partitionsShape.size(), dataShape.end());
      (void)setDenseResult(outType, outShape, buckets[i]);
    }
    return;
  }

  if (isa<IntegerType, IndexType>(dataElemTy)) {
    auto dataValsOr = getIntValues(dataAttr);
    if (failed(dataValsOr)) {
      emitOpError("DynamicPartition integer data values not available");
      return;
    }

    std::vector<std::vector<int64_t>> buckets(static_cast<size_t>(numPartitions));
    for (int64_t i = 0; i < partitionsSize; ++i) {
      int64_t p = (*partValsOr)[i];
      if (p < 0 || p >= numPartitions) {
        emitOpError("DynamicPartition partition id out of range");
        return;
      }
      int64_t base = i * innerSize;
      for (int64_t j = 0; j < innerSize; ++j) {
        buckets[static_cast<size_t>(p)].push_back((*dataValsOr)[base + j]);
      }
    }

    for (auto [i, outVal] : llvm::enumerate(getOutputs())) {
      auto outType = dyn_cast<atir::TensorType>(outVal.getType());
      if (!outType) {
        emitOpError("DynamicPartition output must be atir::TensorType");
        return;
      }
      SmallVector<int64_t> outShape;
      outShape.push_back(static_cast<int64_t>(buckets[i].size() / innerSize));
      outShape.append(dataShape.begin() + partitionsShape.size(), dataShape.end());
      if (failed(setDenseIntResult(outType, outShape, buckets[i]))) {
        emitOpError("DynamicPartition output integer element type not supported");
        return;
      }
    }
    return;
  }

  emitOpError("DynamicPartition data element type is not supported");
}

void ParallelDynamicStitchOp::Interpret() {
  this->inferShape();

  if (getIndices().empty() || getData().empty() ||
      getIndices().size() != getData().size()) {
    emitOpError("ParallelDynamicStitch expects matched non-empty indices/data");
    return;
  }

  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  if (outputShape.empty() || outputShape[0] < 0) {
    emitOpError("ParallelDynamicStitch output shape must be static");
    return;
  }

  int64_t rowWidth = 1;
  for (size_t i = 1; i < outputShape.size(); ++i) {
    if (outputShape[i] < 0) {
      emitOpError("ParallelDynamicStitch output shape must be static");
      return;
    }
    rowWidth *= outputShape[i];
  }
  int64_t outRows = outputShape[0];

  Type elemTy = resultType.getElementType();
  if (isa<FloatType>(elemTy)) {
    std::vector<float> out(static_cast<size_t>(getElementCount(outputShape)), 0.0f);

    for (auto [idxVal, dataVal] : llvm::zip(getIndices(), getData())) {
      atir::TensorType idxType, dataType;
      DenseElementsAttr idxAttr, dataAttr;
      if (failed(getTensorTypeAndData(getOperation(), idxVal,
                                      "ParallelDynamicStitch indices",
                                      idxType, idxAttr)) ||
          failed(getTensorTypeAndData(getOperation(), dataVal,
                                      "ParallelDynamicStitch data",
                                      dataType, dataAttr))) {
        return;
      }
      auto idxsOr = getIntValues(idxAttr);
      auto dataOr = getFloatValues(dataAttr);
      if (failed(idxsOr) || failed(dataOr)) {
        emitOpError("ParallelDynamicStitch expects integer indices and numeric data");
        return;
      }
      const auto &idxs = *idxsOr;
      const auto &vals = *dataOr;
      if (static_cast<int64_t>(vals.size()) !=
          static_cast<int64_t>(idxs.size()) * rowWidth) {
        emitOpError("ParallelDynamicStitch data/indices size mismatch");
        return;
      }
      for (int64_t i = 0; i < static_cast<int64_t>(idxs.size()); ++i) {
        int64_t dst = idxs[i];
        if (dst < 0 || dst >= outRows) {
          emitOpError("ParallelDynamicStitch index out of range");
          return;
        }
        int64_t srcBase = i * rowWidth;
        int64_t dstBase = dst * rowWidth;
        for (int64_t j = 0; j < rowWidth; ++j) {
          out[dstBase + j] = vals[srcBase + j];
        }
      }
    }

    (void)setDenseResult(resultType, outputShape, out);
    return;
  }

  if (isa<IntegerType, IndexType>(elemTy)) {
    std::vector<int64_t> out(static_cast<size_t>(getElementCount(outputShape)), 0);

    for (auto [idxVal, dataVal] : llvm::zip(getIndices(), getData())) {
      atir::TensorType idxType, dataType;
      DenseElementsAttr idxAttr, dataAttr;
      if (failed(getTensorTypeAndData(getOperation(), idxVal,
                                      "ParallelDynamicStitch indices",
                                      idxType, idxAttr)) ||
          failed(getTensorTypeAndData(getOperation(), dataVal,
                                      "ParallelDynamicStitch data",
                                      dataType, dataAttr))) {
        return;
      }
      auto idxsOr = getIntValues(idxAttr);
      auto dataOr = getIntValues(dataAttr);
      if (failed(idxsOr) || failed(dataOr)) {
        emitOpError("ParallelDynamicStitch expects integer-like data here");
        return;
      }
      const auto &idxs = *idxsOr;
      const auto &vals = *dataOr;
      if (static_cast<int64_t>(vals.size()) !=
          static_cast<int64_t>(idxs.size()) * rowWidth) {
        emitOpError("ParallelDynamicStitch data/indices size mismatch");
        return;
      }
      for (int64_t i = 0; i < static_cast<int64_t>(idxs.size()); ++i) {
        int64_t dst = idxs[i];
        if (dst < 0 || dst >= outRows) {
          emitOpError("ParallelDynamicStitch index out of range");
          return;
        }
        int64_t srcBase = i * rowWidth;
        int64_t dstBase = dst * rowWidth;
        for (int64_t j = 0; j < rowWidth; ++j) {
          out[dstBase + j] = vals[srcBase + j];
        }
      }
    }

    if (failed(setDenseIntResult(resultType, outputShape, out))) {
      emitOpError("ParallelDynamicStitch integer output type unsupported");
    }
    return;
  }

  emitOpError("ParallelDynamicStitch output element type unsupported");
}

void WhereOp::Interpret() {
  this->inferShape();

  if (getInputs().size() == 1) {
    // 当只有一个输入时，返回满足条件的索引
    atir::TensorType condType;
    DenseElementsAttr condAttr;
    if (failed(getTensorTypeAndData(getOperation(), getInputs().front(),
                                    "Where condition", condType, condAttr))) {
      return;
    }

    auto condValsOr = getFloatValues(condAttr);
    if (failed(condValsOr)) {
      emitOpError("Where condition must be numeric");
      return;
    }
    
    auto condShape = condType.getShape();
    int64_t rank = static_cast<int64_t>(condShape.size());
    
    std::vector<int64_t> indices;
    for (size_t i = 0; i < condValsOr->size(); ++i) {
      if ((*condValsOr)[i] != 0.0f) {
        auto multiIndex = getMultiIndex(condShape, static_cast<int64_t>(i));
        for (int64_t dim = 0; dim < rank; ++dim) {
          indices.push_back(multiIndex[dim]);
        }
      }
    }

    // 计算实际的输出形状: [N, R]
    int64_t numMatches = static_cast<int64_t>(indices.size()) / rank;
    llvm::SmallVector<int64_t> actualShape{numMatches, rank};
    
    auto resultType = getResult().getType();
    if (failed(setDenseIntResult(resultType, actualShape, indices))) {
      emitOpError("Where output element type is not supported");
    }
    return;
  }

  if (getInputs().size() == 3) {
    // 三个输入: condition, x, y
    atir::TensorType condType;
    atir::TensorType xType;
    atir::TensorType yType;
    DenseElementsAttr condAttr;
    DenseElementsAttr xAttr;
    DenseElementsAttr yAttr;
    
    if (failed(getTensorTypeAndData(getOperation(), getInputs()[0],
                                    "Where condition", condType, condAttr)) ||
        failed(getTensorTypeAndData(getOperation(), getInputs()[1],
                                    "Where x", xType, xAttr)) ||
        failed(getTensorTypeAndData(getOperation(), getInputs()[2],
                                    "Where y", yType, yAttr))) {
      return;
    }

    auto condValsOr = getFloatValues(condAttr);
    auto xValsOr = getFloatValues(xAttr);
    auto yValsOr = getFloatValues(yAttr);
    if (failed(condValsOr) || failed(xValsOr) || failed(yValsOr)) {
      emitOpError("Where inputs must be numeric");
      return;
    }

    auto resultType = getResult().getType();
    auto outputShape = resultType.getShape();
    int64_t outputSize = getElementCount(outputShape);
    if (outputSize < 0) {
      emitOpError("Where output shape must be static for interpretation");
      return;
    }

    std::vector<float> result(outputSize, 0.0f);
    for (int64_t outIdx = 0; outIdx < outputSize; ++outIdx) {
      auto outputIndex = getMultiIndex(outputShape, outIdx);
      auto condIndex =
          getBroadcastIndex(outputShape, condType.getShape(), outputIndex);
      auto xIndex = getBroadcastIndex(outputShape, xType.getShape(), outputIndex);
      auto yIndex = getBroadcastIndex(outputShape, yType.getShape(), outputIndex);
      
      float condValue = (*condValsOr)[getFlatIndex(condType.getShape(), condIndex)];
      float xValue = (*xValsOr)[getFlatIndex(xType.getShape(), xIndex)];
      float yValue = (*yValsOr)[getFlatIndex(yType.getShape(), yIndex)];
      
      result[outIdx] = condValue != 0.0f ? xValue : yValue;
    }

    (void)setDenseResult(resultType, outputShape, result);
    return;
  }

  emitOpError("Where expects either 1 input (indices) or 3 inputs (condition, x, y)");
}

}  // namespace atir
