#include "Common.h"

namespace atir {
namespace interpret {

/// ATIR 用 i32 + encoding "bool" 表示 tf.bool（与 NewBuilder 一致）。
bool tensorTypeIsBoolAsI32(atir::TensorType resultType) {
  if (!resultType.getElementType().isInteger(32))
    return false;
  auto enc = dyn_cast_or_null<StringAttr>(resultType.getEncoding());
  return enc && enc.getValue() == "bool";
}

int64_t getElementCount(ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    return 1;
  }
  int64_t count = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      return -1;
    }
    count *= dim;
  }
  return count;
}

SmallVector<int64_t> getMultiIndex(ArrayRef<int64_t> shape, int64_t flatIndex) {
  if (shape.empty()) {
    return {};
  }
  SmallVector<int64_t> index(shape.size(), 0);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    index[i] = flatIndex % shape[i];
    flatIndex /= shape[i];
  }
  return index;
}

int64_t getFlatIndex(ArrayRef<int64_t> shape, ArrayRef<int64_t> index) {
  if (shape.empty()) {
    return 0;
  }
  int64_t flat = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    flat = flat * shape[i] + index[i];
  }
  return flat;
}

SmallVector<int64_t> getBroadcastIndex(ArrayRef<int64_t> outputShape,
                                       ArrayRef<int64_t> inputShape,
                                       ArrayRef<int64_t> outputIndex) {
  if (inputShape.empty()) {
    return {};
  }

  SmallVector<int64_t> inputIndex(inputShape.size(), 0);
  int64_t rankDiff = static_cast<int64_t>(outputShape.size()) -
                     static_cast<int64_t>(inputShape.size());
  for (int64_t i = 0; i < static_cast<int64_t>(inputShape.size()); ++i) {
    int64_t outAxis = i + rankDiff;
    int64_t dim = inputShape[i];
    inputIndex[i] = (dim == 1) ? 0 : outputIndex[outAxis];
  }
  return inputIndex;
}

LogicalResult getTensorTypeAndData(Operation *op, Value value, StringRef name,
                                   atir::TensorType &tensorType,
                                   DenseElementsAttr &attr) {
  auto type = dyn_cast<atir::TensorType>(value.getType());
  if (!type) {
    op->emitOpError() << name << " must be atir::TensorType";
    return failure();
  }

  attr = type.getCacheData();
  if (!attr) {
    op->emitOpError() << "missing cacheData for " << name;
    return failure();
  }

  tensorType = type;
  return success();
}

FailureOr<std::vector<float>> getFloatValues(DenseElementsAttr attr) {
  std::vector<float> values;
  auto elementType = attr.getElementType();

  if (isa<FloatType>(elementType)) {
    values.reserve(attr.getNumElements());
    for (const APFloat &value : attr.getValues<APFloat>()) {
      values.push_back(annc::apFloatToFloat(value));
    }
    return values;
  }

  if (isa<IntegerType, IndexType>(elementType)) {
    values.reserve(attr.getNumElements());
    for (const APInt &value : attr.getValues<APInt>()) {
      values.push_back(static_cast<float>(value.getSExtValue()));
    }
    return values;
  }

  return failure();
}

FailureOr<SmallVector<int64_t>> getIntValues(DenseElementsAttr attr) {
  SmallVector<int64_t> values;
  auto elementType = attr.getElementType();

  if (isa<IntegerType, IndexType>(elementType)) {
    values.reserve(attr.getNumElements());
    for (const APInt &value : attr.getValues<APInt>()) {
      values.push_back(value.getSExtValue());
    }
    return values;
  }

  if (isa<FloatType>(elementType)) {
    values.reserve(attr.getNumElements());
    for (const APFloat &value : attr.getValues<APFloat>()) {
      // tf.cast float→signed int: truncate toward zero (same as C++ static_cast).
      values.push_back(
          static_cast<int64_t>(annc::apFloatToDouble(value)));
    }
    return values;
  }

  return failure();
}

SmallVector<int64_t> getIntArrayAttrValues(ArrayAttr attr) {
  SmallVector<int64_t> values;
  if (!attr) {
    return values;
  }

  values.reserve(attr.size());
  for (Attribute element : attr) {
    if (auto intAttr = dyn_cast<IntegerAttr>(element)) {
      values.push_back(intAttr.getInt());
    }
  }
  return values;
}

Attribute getZeroElementAttr(Type elementType, MLIRContext *context) {
  Builder builder(context);
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    return builder.getFloatAttr(floatType, 0.0);
  }
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    return builder.getIntegerAttr(intType, 0);
  }
  if (isa<IndexType>(elementType)) {
    return builder.getIndexAttr(0);
  }
  return {};
}

// 设置Dense结果
LogicalResult setDenseResult(atir::TensorType resultType,
                             ArrayRef<int64_t> outputShape,
                             ArrayRef<float> values) {
  auto standardTensorType =
      RankedTensorType::get(outputShape, resultType.getElementType());
  auto elementType = resultType.getElementType();

  if (elementType.isF32()) {
    resultType.setCacheData(DenseElementsAttr::get(standardTensorType, values));
    return success();
  }

  if (elementType.isF64()) {
    std::vector<double> casted;
    casted.reserve(values.size());
    for (float v : values) {
      casted.push_back(static_cast<double>(v));
    }
    resultType.setCacheData(
        DenseElementsAttr::get(standardTensorType, ArrayRef<double>(casted)));
    return success();
  }

  if (elementType.isF16()) {
    std::vector<APFloat> casted;
    casted.reserve(values.size());
    for (float v : values) {
      APFloat ap(v);
      bool losesInfo = false;
      (void)ap.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven,
                       &losesInfo);
      casted.push_back(ap);
    }
    resultType.setCacheData(
        DenseElementsAttr::get(standardTensorType, ArrayRef<APFloat>(casted)));
    return success();
  }

  // 实现TensorFlow的布尔转换语义：非零值为true(1)，零值为false(0)
  if (tensorTypeIsBoolAsI32(resultType)) {
    std::vector<int32_t> casted;
    casted.reserve(values.size());
    for (float v : values)
      casted.push_back(v != 0.0f ? 1 : 0);
    resultType.setCacheData(DenseElementsAttr::get(standardTensorType,
                                                     ArrayRef<int32_t>(casted)));
    return success();
  }

  std::vector<int64_t> intValues;
  intValues.reserve(values.size());
  for (float v : values) {
    intValues.push_back(static_cast<int64_t>(v));
  }
  return setDenseIntResult(resultType, outputShape, intValues);
}

// 设置密集整数结果
LogicalResult setDenseIntResult(atir::TensorType resultType,
                                ArrayRef<int64_t> outputShape,
                                ArrayRef<int64_t> values) {
  auto standardTensorType =
      RankedTensorType::get(outputShape, resultType.getElementType());
  auto elementType = resultType.getElementType();
  if (elementType.isInteger(1)) {
    std::vector<int64_t> casted;
    casted.reserve(values.size());
    for (int64_t v : values) {
      casted.push_back(v ? 1 : 0);
    }
    resultType.setCacheData(
        DenseElementsAttr::get(standardTensorType, ArrayRef<int64_t>(casted)));
    return success();
  }

  if (elementType.isInteger(32)) {
    if (tensorTypeIsBoolAsI32(resultType)) {
      std::vector<int32_t> casted;
      casted.reserve(values.size());
      for (int64_t v : values)
        casted.push_back(v != 0 ? 1 : 0);
      resultType.setCacheData(DenseElementsAttr::get(
          standardTensorType, ArrayRef<int32_t>(casted)));
      return success();
    }
    std::vector<int32_t> casted(values.begin(), values.end());
    resultType.setCacheData(DenseElementsAttr::get(
        standardTensorType, ArrayRef<int32_t>(casted)));
    return success();
  }

  if (elementType.isInteger(16)) {
    std::vector<int16_t> casted;
    casted.reserve(values.size());
    for (int64_t v : values) {
      casted.push_back(static_cast<int16_t>(v));
    }
    resultType.setCacheData(DenseElementsAttr::get(
        standardTensorType, ArrayRef<int16_t>(casted)));
    return success();
  }

  if (elementType.isInteger(8)) {
    std::vector<int8_t> casted;
    casted.reserve(values.size());
    for (int64_t v : values) {
      casted.push_back(static_cast<int8_t>(v));
    }
    resultType.setCacheData(
        DenseElementsAttr::get(standardTensorType, ArrayRef<int8_t>(casted)));
    return success();
  }

  if (elementType.isInteger(64) || elementType.isIndex()) {
    resultType.setCacheData(
        DenseElementsAttr::get(standardTensorType, values));
    return success();
  }

  return failure();
}

// 设置布尔结果
LogicalResult setBooleanLikeResult(atir::TensorType resultType,
                                   ArrayRef<int64_t> outputShape,
                                   ArrayRef<int64_t> values) {
  auto elementType = resultType.getElementType();
  if (isa<FloatType>(elementType)) {
    std::vector<float> casted;
    casted.reserve(values.size());
    for (int64_t value : values) {
      casted.push_back(static_cast<float>(value));
    }
    return setDenseResult(resultType, outputShape, casted);
  }

  return setDenseIntResult(resultType, outputShape, values);
}

std::vector<int64_t> inferReductionAxesFromShapes(ArrayRef<int64_t> inputShape,
                                                  ArrayRef<int64_t> outputShape) {
  std::vector<int64_t> axes;

  if (inputShape.size() == outputShape.size()) {
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (outputShape[i] == 1 && inputShape[i] != 1) {
        axes.push_back(i);
      }
    }
    return axes;
  }

  size_t outPos = 0;
  for (size_t inPos = 0; inPos < inputShape.size(); ++inPos) {
    if (outPos < outputShape.size() && inputShape[inPos] == outputShape[outPos]) {
      ++outPos;
      continue;
    }
    axes.push_back(inPos);
  }

  if (outPos != outputShape.size()) {
    axes.clear();
  }
  return axes;
}

float applyReluLimit(float value, bool doRelu, float reluLimit) {
  if (!doRelu) {
    return value;
  }
  value = std::max(0.0f, value);
  if (reluLimit >= 0.0f) {
    value = std::min(value, reluLimit);
  }
  return value;
}

} // namespace interpret
} // namespace atir
