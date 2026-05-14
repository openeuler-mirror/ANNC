#include <cmath>
#include <optional>
#include <string>

#include <llvm/Support/Debug.h>
#include "Helper.h"
#include "Dialect/Atir/AtirOps.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "Dialect/Atir/Interfaces/ShapeInfer.cpp.inc"

namespace atir {
#define UNREACHABLE_OP(info) emitError(info);

namespace {

bool isNoneOperand(Value value) {
  return llvm::isa_and_nonnull<atir::NoneOp>(value.getDefiningOp());
}

atir::TensorType cloneWithShape(atir::TensorType type,
                                llvm::ArrayRef<int64_t> shape) {
  return atir::TensorType::get(
      shape, type.getElementType(), type.getName(), type.getEncoding(),
      type.getStride(), type.getLayout(), type.getMemType(), type.getAddress(),
      type.getDeviceParallel(), type.getOnchipParallel(), type.getCacheData());
}

LogicalResult setSingleResultShape(Operation *op, llvm::ArrayRef<int64_t> shape) {
  if (op->getNumResults() != 1) {
    return op->emitOpError("expects exactly one result");
  }

  auto resultType = dyn_cast<atir::TensorType>(op->getResult(0).getType());
  if (!resultType) {
    return op->emitOpError("result must be atir::TensorType");
  }

  op->getResult(0).setType(cloneWithShape(resultType, shape));
  return success();
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

std::optional<int64_t> mergeBroadcastDims(Operation *op, int64_t lhsDim,
                                          int64_t rhsDim) {
  if (lhsDim == rhsDim) {
    return lhsDim;
  }
  if (lhsDim == 1) {
    return rhsDim;
  }
  if (rhsDim == 1) {
    return lhsDim;
  }
  if (lhsDim == ShapedType::kDynamic || rhsDim == ShapedType::kDynamic) {
    return ShapedType::kDynamic;
  }

  op->emitOpError() << "broadcast shape mismatch: " << lhsDim << " vs "
                    << rhsDim;
  return std::nullopt;
}

LogicalResult inferConcatLikeOpShape(Operation *op, ValueRange inputs,
                                     int64_t axis) {
  llvm::SmallVector<int64_t> outShape;
  bool initialized = false;

  for (Value input : inputs) {
    if (isNoneOperand(input)) {
      continue;
    }

    auto tensorType = dyn_cast<atir::TensorType>(input.getType());
    if (!tensorType) {
      return op->emitOpError("expects atir::TensorType operands");
    }

    auto shape = tensorType.getShape();
    if (!initialized) {
      outShape.assign(shape.begin(), shape.end());
      initialized = true;
      continue;
    }

    if (shape.size() != outShape.size()) {
      return op->emitOpError("concat inputs must have the same rank");
    }

    int64_t normalizedAxis = axis;
    if (normalizedAxis < 0) {
      normalizedAxis += static_cast<int64_t>(shape.size());
    }
    if (normalizedAxis < 0 ||
        normalizedAxis >= static_cast<int64_t>(shape.size())) {
      return op->emitOpError("concat axis out of range");
    }

    for (size_t dim = 0; dim < shape.size(); ++dim) {
      if (static_cast<int64_t>(dim) == normalizedAxis) {
        if (outShape[dim] == ShapedType::kDynamic ||
            shape[dim] == ShapedType::kDynamic) {
          outShape[dim] = ShapedType::kDynamic;
        } else {
          outShape[dim] += shape[dim];
        }
        continue;
      }

      if (outShape[dim] != shape[dim]) {
        return op->emitOpError()
               << "concat inputs must match on non-axis dimensions";
      }
    }
  }

  if (!initialized) {
    return op->emitOpError("expects at least one tensor operand");
  }
  return setSingleResultShape(op, outShape);
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

}  // namespace

// 形状广播函数
llvm::SmallVector<int64_t> computeBroadcastShape(mlir::Operation *op) {
  llvm::SmallVector<int64_t> outShape;
  bool initialized = false;

  for (Value operand : op->getOperands()) {
    if (isNoneOperand(operand)) {
      continue;
    }

    auto operandTensorType = dyn_cast<atir::TensorType>(operand.getType());
    if (!operandTensorType) {
      op->emitOpError("expected atir::TensorType operand for shape inference");
      return {};
    }

    auto operandShape = operandTensorType.getShape();
    if (!initialized) {
      outShape.assign(operandShape.begin(), operandShape.end());
      initialized = true;
      continue;
    }

    size_t outRank = outShape.size();
    size_t operandRank = operandShape.size();
    size_t resultRank = std::max(outRank, operandRank);
    llvm::SmallVector<int64_t> merged(resultRank, 1);

    for (size_t idx = 0; idx < resultRank; ++idx) {  // 从右往左记录维度信息
      int64_t lhsDim = 1;
      int64_t rhsDim = 1;
      if (idx < outRank) {
        lhsDim = outShape[outRank - 1 - idx];
      }
      if (idx < operandRank) {
        rhsDim = operandShape[operandRank - 1 - idx];
      }

      auto mergedDim = mergeBroadcastDims(op, lhsDim, rhsDim);
      if (!mergedDim) {
        return {};
      }
      merged[resultRank - 1 - idx] = *mergedDim;
    }

    outShape = std::move(merged);
  }

  return outShape;
}

void inferEltwiseOpShape(mlir::Operation *op) {
  bool hasMaterializedOperand = false;
  for (Value operand : op->getOperands()) {
    if (!isNoneOperand(operand)) {
      hasMaterializedOperand = true;
      break;
    }
  }
  if (!hasMaterializedOperand) {
    op->emitOpError("expects at least one tensor operand");
    return;
  }

  auto outShape = computeBroadcastShape(op); // 获取广播后的输出形状
  if (outShape.empty()) {
    return;
  }
  (void)setSingleResultShape(op, outShape);
}

int64_t inferStaticRangeOutputDim(double start, double limit, double delta,
                                  int64_t maxElements) {
  if (delta == 0.0) {
    return -1;
  }
  int64_t n = 0;
  if (delta > 0) {
    for (double cur = start; cur < limit; cur += delta) {
      ++n;
      if (n > maxElements) {
        return -2;
      }
    }
  } else {
    for (double cur = start; cur > limit; cur += delta) {
      ++n;
      if (n > maxElements) {
        return -2;
      }
    }
  }
  return n;
}

static bool readStridedSlice1DInts(atir::TensorType ty,
                                   SmallVector<int64_t> &out) {
  out.clear();
  if (!ty)
    return false;
  auto shape = ty.getShape();
  if (shape.size() != 1)
    return false;
  DenseElementsAttr cache = ty.getCacheData();
  if (!cache)
    return false;
  int64_t dim = shape[0];
  if (dim >= 0 && static_cast<int64_t>(cache.getNumElements()) != dim)
    return false;
  auto elemTy = cache.getElementType();
  if (isa<IntegerType, IndexType>(elemTy)) {
    for (const APInt &v : cache.getValues<APInt>())
      out.push_back(v.getSExtValue());
    return true;
  }
  if (isa<FloatType>(elemTy)) {
    for (const APFloat &v : cache.getValues<APFloat>())
      out.push_back(static_cast<int64_t>(annc::apFloatToDouble(v)));
    return true;
  }
  return false;
}

// --- TensorFlow-compatible StridedSlice dense spec (no separate header) ---

constexpr int64_t kStridedSliceShrinkGather = -1;
constexpr int64_t kStridedSliceNewAxisGather = -2;

struct StridedSliceDensePlan {
  bool ok = false;
  int64_t input_rank = 0;
  SmallVector<int64_t, 8> dense_begin;
  SmallVector<int64_t, 8> dense_end;
  SmallVector<int64_t, 8> dense_stride;
  int32_t dense_begin_mask = 0;
  int32_t dense_end_mask = 0;
  int32_t dense_shrink_mask = 0;
  SmallVector<int64_t, 8> final_gather;
};

static int64_t stridedSliceDimLen(int64_t dim, int64_t start, int64_t stop,
                                  int64_t step, bool shrink, bool useBeginMask,
                                  bool useEndMask) {
  if (step == 0)
    return ShapedType::kDynamic;
  if (shrink)
    return 1;
  if (dim == ShapedType::kDynamic)
    return ShapedType::kDynamic;

  if (useBeginMask)
    start = step > 0 ? 0 : dim - 1;
  else if (start < 0)
    start += dim;

  if (useEndMask)
    stop = step > 0 ? dim : -1;
  else if (stop < 0)
    stop += dim;

  if (step > 0) {
    start = std::min<int64_t>(std::max<int64_t>(start, 0), dim);
    stop = std::min<int64_t>(std::max<int64_t>(stop, 0), dim);
    if (stop <= start)
      return 0;
    return (stop - start + step - 1) / step;
  }
  start = std::min<int64_t>(std::max<int64_t>(start, -1), dim - 1);
  stop = std::min<int64_t>(std::max<int64_t>(stop, -1), dim - 1);
  int64_t negStep = -step;
  if (start <= stop)
    return 0;
  return (start - stop + negStep - 1) / negStep;
}

static bool stridedSliceMaskBit32(int32_t m, int32_t bit) {
  return bit < 32 && ((static_cast<uint32_t>(m) >> bit) & 1u) != 0;
}

static bool buildStridedSliceDensePlan(llvm::ArrayRef<int64_t> input_shape,
                                       llvm::ArrayRef<int64_t> sparse_begin,
                                       llvm::ArrayRef<int64_t> sparse_end,
                                       llvm::ArrayRef<int64_t> sparse_stride,
                                       int32_t begin_mask_spec,
                                       int32_t end_mask_spec, int32_t ellipsis_mask,
                                       int32_t new_axis_mask, int32_t shrink_axis_mask,
                                       StridedSliceDensePlan &out, std::string *err) {
  out = StridedSliceDensePlan{};
  const int64_t input_shape_size = static_cast<int64_t>(input_shape.size());
  const size_t sparse_stride_size = sparse_stride.size();
  if (sparse_begin.size() != sparse_stride_size || sparse_end.size() != sparse_stride_size) {
    if (err)
      *err = "begin, end, and strides must have the same length";
    return false;
  }
  if (sparse_stride_size >= 32) {
    if (err)
      *err = "begin/end/strides tensor must have fewer than 32 elements";
    return false;
  }
  // 省略号掩码最多只有一位非零
  if (ellipsis_mask && ((ellipsis_mask & (ellipsis_mask - 1)) != 0)) {
    if (err)
      *err = "ellipsis_mask must have at most one bit set";
    return false;
  }

  int32_t sparse_dims = static_cast<int32_t>(sparse_stride_size);
  int32_t num_add_after_ellipsis = 0;
  bool ellipsis_seen = false;
  for (int32_t i = 0; i < sparse_dims; ++i) {
    if (ellipsis_seen && ((static_cast<uint32_t>(new_axis_mask) >> i) & 1u))
      ++num_add_after_ellipsis;
    if (((static_cast<uint32_t>(ellipsis_mask) >> i) & 1u))
      ellipsis_seen = true;
  }

  int32_t ellipsis_mask_eff = ellipsis_mask;
  if (!ellipsis_seen) {
    ellipsis_mask_eff |= (1 << sparse_dims);
    ++sparse_dims;
  }

  out.input_rank = input_shape_size;
  out.dense_begin.assign(input_shape_size, 0);
  out.dense_end.assign(input_shape_size, 0);
  out.dense_stride.assign(input_shape_size, 1);
  out.dense_begin_mask = 0;
  out.dense_end_mask = 0;
  out.dense_shrink_mask = 0;
  out.final_gather.clear();

  int32_t full_index = 0;
  const int32_t input_shape_dims = static_cast<int32_t>(input_shape_size);

  for (int32_t i = 0; i < sparse_dims; ++i) {
    if (((static_cast<uint32_t>(ellipsis_mask_eff) >> i) & 1u)) {
      // 省略号掩码位置
      int32_t next_index =
          std::min(input_shape_dims - (sparse_dims - i) + 1 + num_add_after_ellipsis, input_shape_dims);
      for (; full_index < next_index; ++full_index) {
        if (full_index < 0 || full_index >= input_shape_dims) {
          if (err)
            *err = "ellipsis expansion out of range for input rank";
          return false;
        }
        out.dense_begin[full_index] = 0;
        out.dense_end[full_index] = 0;
        out.dense_stride[full_index] = 1;
        out.dense_begin_mask |= (1 << full_index);
        out.dense_end_mask |= (1 << full_index);
        out.final_gather.push_back(full_index);
      }
    } else if (((static_cast<uint32_t>(new_axis_mask) >> i) & 1u)) {
      out.final_gather.push_back(kStridedSliceNewAxisGather);
    } else {
      if (full_index < 0 || full_index >= input_shape_dims) {
        if (err)
          *err = "slice index out of range for input rank";
        return false;
      }
      out.dense_begin[full_index] = sparse_begin[i];
      out.dense_end[full_index] = sparse_end[i];
      out.dense_stride[full_index] = sparse_stride[i];
      if (stridedSliceMaskBit32(begin_mask_spec, i))
        out.dense_begin_mask |= (1 << full_index);
      if (stridedSliceMaskBit32(end_mask_spec, i))
        out.dense_end_mask |= (1 << full_index);
      if (stridedSliceMaskBit32(shrink_axis_mask, i)) {
        out.final_gather.push_back(kStridedSliceShrinkGather);
        out.dense_shrink_mask |= (1 << full_index);
      } else {
        out.final_gather.push_back(full_index);
      }
      ++full_index;
    }
  }

  if (full_index != input_shape_dims) {
    if (err)
      *err = "strided slice spec does not match input tensor rank";
    return false;
  }

  out.ok = true;
  return true;
}

static void inferStridedSliceProcessingShape(
    llvm::ArrayRef<int64_t> input_shape, const StridedSliceDensePlan &plan,
    llvm::SmallVectorImpl<int64_t> &processing_shape) {
  processing_shape.clear();
  if (!plan.ok || static_cast<size_t>(plan.input_rank) != input_shape.size())
    return;
  processing_shape.resize(static_cast<size_t>(plan.input_rank));
  for (int64_t i = 0; i < plan.input_rank; ++i) {
    int32_t ii = static_cast<int32_t>(i);
    int64_t dim = input_shape[i];
    int64_t step = plan.dense_stride[i];
    bool shrink = stridedSliceMaskBit32(plan.dense_shrink_mask, ii);

    if (step == 0) {
      processing_shape[static_cast<size_t>(i)] = ShapedType::kDynamic;
      continue;
    }
    if (shrink && step != 1) {
      processing_shape[static_cast<size_t>(i)] = ShapedType::kDynamic;
      continue;
    }

    bool bm = stridedSliceMaskBit32(plan.dense_begin_mask, ii);
    bool em = stridedSliceMaskBit32(plan.dense_end_mask, ii);
    processing_shape[static_cast<size_t>(i)] = stridedSliceDimLen(
        dim, plan.dense_begin[i], plan.dense_end[i], step, shrink, bm, em);
  }
}

static void inferStridedSliceShapes(llvm::ArrayRef<int64_t> input_shape,
                                    const StridedSliceDensePlan &plan,
                                    llvm::SmallVectorImpl<int64_t> &final_shape) {
  final_shape.clear();
  if (!plan.ok || static_cast<size_t>(plan.input_rank) != input_shape.size())
    return;
  SmallVector<int64_t, 8> processing;
  inferStridedSliceProcessingShape(input_shape, plan, processing);
  if (processing.size() != static_cast<size_t>(plan.input_rank))
    return;

  for (int64_t g : plan.final_gather) {
    if (g == kStridedSliceNewAxisGather)
      final_shape.push_back(1);
    else if (g == kStridedSliceShrinkGather)
      continue;
    else if (g >= 0 && g < plan.input_rank)
      final_shape.push_back(processing[static_cast<size_t>(g)]);
  }
}

// BinaryOps
void AddOp::inferShape() { inferEltwiseOpShape(getOperation()); }
void SubOp::inferShape() { inferEltwiseOpShape(getOperation()); }
void MulOp::inferShape() { inferEltwiseOpShape(getOperation()); }
void RealDivOp::inferShape() { inferEltwiseOpShape(getOperation()); }

// ComparisonOps
void NotEqualOp::inferShape() { inferEltwiseOpShape(getOperation()); }
void LessOp::inferShape() { inferEltwiseOpShape(getOperation()); }
void GreaterEqualOp::inferShape() { inferEltwiseOpShape(getOperation()); }
void GreaterOp::inferShape() { inferEltwiseOpShape(getOperation()); }
void CompareOp::inferShape() { inferEltwiseOpShape(getOperation()); }
void AndOp::inferShape() { inferEltwiseOpShape(getOperation()); }
void MinimumOp::inferShape() { inferEltwiseOpShape(getOperation()); }
void MaximumOp::inferShape() { inferEltwiseOpShape(getOperation()); }

void ConcatOp::inferShape() {
  // 第一个操作数是 output，从第二个开始是 inputs
  SmallVector<Value> inputs;
  for (size_t i = 1; i < getNumOperands(); ++i) {
    inputs.push_back(getOperand(i));
  }
  (void)inferConcatLikeOpShape(getOperation(), inputs, getAxis());
}

void ConcatV2Op::inferShape() {
  auto axisType = dyn_cast<atir::TensorType>(getAxis().getType());
  if (!axisType || !axisType.getCacheData()) {
    return;
  }

  auto axisAttr = axisType.getCacheData();
  if (axisAttr.getNumElements() != 1) {
    emitError("ConcatV2 axis must be scalar");
    return;
  }

  auto axisValue = axisAttr.getValues<APInt>()[0].getSExtValue();
  (void)inferConcatLikeOpShape(getOperation(), getValues(), axisValue);
}

void PackOp::inferShape() {
  if (getInputs().empty()) {
    emitError("Pack requires at least one input");
    return;
  }

  auto inputType = dyn_cast<atir::TensorType>(getInputs().front().getType());
  if (!inputType) {
    emitError("Pack inputs must be atir::TensorType");
    return;
  }

  llvm::SmallVector<int64_t> outputShape(inputType.getShape().begin(),
                                         inputType.getShape().end());
  int64_t axis = getAxis();
  int64_t rank = static_cast<int64_t>(outputShape.size());
  if (axis < 0) {
    axis += rank + 1;
  }
  if (axis < 0 || axis > rank) {
    emitError("Pack axis out of range");
    return;
  }

  outputShape.insert(outputShape.begin() + axis,
                     static_cast<int64_t>(getInputs().size()));
  (void)setSingleResultShape(getOperation(), outputShape);
}

void MergeOp::inferShape() {
  auto inputs = getInputsAndControl();
  if (inputs.empty()) {
    emitError("Merge expects at least one input");
    return;
  }

  auto firstInputType = dyn_cast<atir::TensorType>(inputs.front().getType());
  auto outputType = dyn_cast<atir::TensorType>(getOutput().getType());
  auto valueIndexType = dyn_cast<atir::TensorType>(getValueIndex().getType());
  if (!firstInputType || !outputType || !valueIndexType) {
    emitError("Merge operands/results must be atir::TensorType");
    return;
  }

  getOutput().setType(cloneWithShape(outputType, firstInputType.getShape()));
  getValueIndex().setType(cloneWithShape(valueIndexType, {}));
}

void DynamicPartitionOp::inferShape() {
  auto dataType = dyn_cast<atir::TensorType>(getData().getType());
  auto partitionsType = dyn_cast<atir::TensorType>(getPartitions().getType());
  if (!dataType || !partitionsType) {
    emitError("DynamicPartition operands must be atir::TensorType");
    return;
  }

  auto dataShape = dataType.getShape();
  auto partitionsShape = partitionsType.getShape();
  if (dataShape.size() < partitionsShape.size()) {
    emitError("DynamicPartition data rank must be >= partitions rank");
    return;
  }
  for (size_t i = 0; i < partitionsShape.size(); ++i) {
    int64_t pd = partitionsShape[i];
    int64_t dd = dataShape[i];
    if (pd != ShapedType::kDynamic && dd != ShapedType::kDynamic && pd != dd) {
      emitError("DynamicPartition partitions shape must match data prefix");
      return;
    }
  }

  SmallVector<int64_t> outputShape;
  outputShape.push_back(ShapedType::kDynamic);
  outputShape.append(dataShape.begin() + partitionsShape.size(), dataShape.end());

  for (Value out : getOutputs()) {
    auto outType = dyn_cast<atir::TensorType>(out.getType());
    if (!outType) {
      emitError("DynamicPartition outputs must be atir::TensorType");
      return;
    }
    out.setType(cloneWithShape(outType, outputShape));
  }
}

void ParallelDynamicStitchOp::inferShape() {
  if (getIndices().empty() || getData().empty() ||
      getIndices().size() != getData().size()) {
    emitError("ParallelDynamicStitch expects non-empty matched indices/data lists");
    return;
  }

  auto firstDataType = dyn_cast<atir::TensorType>(getData().front().getType());
  if (!firstDataType) {
    emitError("ParallelDynamicStitch data operands must be atir::TensorType");
    return;
  }

  auto dataShape = firstDataType.getShape();
  if (dataShape.empty()) {
    emitError("ParallelDynamicStitch data rank must be >= 1");
    return;
  }

  int64_t firstDim = ShapedType::kDynamic;
  for (Value idx : getIndices()) {
    auto idxType = dyn_cast<atir::TensorType>(idx.getType());
    if (!idxType) {
      emitError("ParallelDynamicStitch indices operands must be atir::TensorType");
      return;
    }
    auto idxShape = idxType.getShape();
    if (idxShape.size() != 1) {
      emitError("ParallelDynamicStitch currently expects 1D indices");
      return;
    }
    if (DenseElementsAttr idxAttr = idxType.getCacheData()) {
      SmallVector<int64_t> idxVals;
      auto elemTy = idxAttr.getElementType();
      if (isa<IntegerType, IndexType>(elemTy)) {
        idxVals.reserve(idxAttr.getNumElements());
        for (const APInt &v : idxAttr.getValues<APInt>()) {
          idxVals.push_back(v.getSExtValue());
        }
      } else if (isa<FloatType>(elemTy)) {
        idxVals.reserve(idxAttr.getNumElements());
        for (const APFloat &v : idxAttr.getValues<APFloat>()) {
          idxVals.push_back(
              static_cast<int64_t>(annc::apFloatToDouble(v)));
        }
      }
      if (!idxVals.empty()) {
        int64_t localMax = *std::max_element(idxVals.begin(), idxVals.end());
        firstDim = std::max(firstDim == ShapedType::kDynamic ? -1 : firstDim,
                            localMax + 1);
      }
    }
  }

  SmallVector<int64_t> outShape;
  outShape.push_back(firstDim);
  outShape.append(dataShape.begin() + 1, dataShape.end());
  (void)setSingleResultShape(getOperation(), outShape);
}

void WhereOp::inferShape() {
  if (getInputs().size() == 1) {
    // 单输入情况：返回满足条件的索引
    auto condType = dyn_cast<atir::TensorType>(getInputs().front().getType());
    if (!condType) {
      emitError("Where condition must be atir::TensorType");
      return;
    }
    // 输出形状: [N, R]，N是满足条件的元素数量（动态），R是输入维度数
    int64_t rank = static_cast<int64_t>(condType.getShape().size());
    (void)setSingleResultShape(getOperation(), {ShapedType::kDynamic, rank});
    return;
  }
  if (getInputs().size() == 3) {
    // 三输入情况：condition, x, y
    inferEltwiseOpShape(getOperation());
    return;
  }
  emitError("Where expects either 1 input (indices) or 3 inputs (condition, x, y)");
}

void BufferOp::inferShape() {
  auto resultType = dyn_cast<atir::TensorType>(getOutput().getType());
  if (!resultType) {
    return;
  }
  (void)setSingleResultShape(getOperation(), resultType.getShape());
}

void CustomizeOp::inferShape() {}

void ConstantOp::inferShape() {
  auto resultType = dyn_cast<atir::TensorType>(getData().getType());
  if (!resultType) {
    return;
  }
  (void)setSingleResultShape(getOperation(), resultType.getShape());
}

void VariableOp::inferShape() {
  auto resultType = dyn_cast<atir::TensorType>(getOutput().getType());
  if (!resultType) {
    return;
  }
  (void)setSingleResultShape(getOperation(), resultType.getShape());
}

void ReturnOp::inferShape() {}

void IdentityOp::inferShape() { inferEltwiseOpShape(getOperation()); }

void ShapeOp::inferShape() {
  auto inputType = dyn_cast<atir::TensorType>(getInput().getType());
  if (!inputType) {
    emitError("Shape input must be atir::TensorType");
    return;
  }

  llvm::SmallVector<int64_t> outputShape = {
      static_cast<int64_t>(inputType.getShape().size())};
  (void)setSingleResultShape(getOperation(), outputShape);
}

void SizeOp::inferShape() { (void)setSingleResultShape(getOperation(), {}); }

void FillOp::inferShape() {
  auto shapeInTy = dyn_cast<atir::TensorType>(getShapeInput().getType());
  if (!shapeInTy) {
    emitError("Fill shapeInput must be atir::TensorType");
    return;
  }

  auto shapeInShape = shapeInTy.getShape();
  if (shapeInShape.size() != 1) {
    emitError("Fill shapeInput must be rank-1");
    return;
  }

  int64_t rank = shapeInShape[0];
  llvm::SmallVector<int64_t> outShape;
  if (rank == ShapedType::kDynamic) {
    return;
  }

  if (DenseElementsAttr shapeAttr = shapeInTy.getCacheData()) {
    auto elemTy = shapeAttr.getElementType();
    if (isa<IntegerType, IndexType>(elemTy)) {
      for (const APInt &v : shapeAttr.getValues<APInt>()) {
        outShape.push_back(v.getSExtValue());
      }
    } else if (isa<FloatType>(elemTy)) {
      for (const APFloat &v : shapeAttr.getValues<APFloat>()) {
        outShape.push_back(
            static_cast<int64_t>(annc::apFloatToDouble(v)));
      }
    }
    if (static_cast<int64_t>(outShape.size()) == rank) {
      (void)setSingleResultShape(getOperation(), outShape);
      return;
    }
  }

  outShape.assign(static_cast<size_t>(rank), ShapedType::kDynamic);
  (void)setSingleResultShape(getOperation(), outShape);
}

void RangeOp::inferShape() {
  auto startTy = dyn_cast<atir::TensorType>(getStart().getType());
  auto limitTy = dyn_cast<atir::TensorType>(getLimit().getType());
  auto deltaTy = dyn_cast<atir::TensorType>(getDelta().getType());
  if (!startTy || !limitTy || !deltaTy) {
    emitError("Range operands must be atir::TensorType");
    return;
  }

  auto readScalarDouble = [](DenseElementsAttr attr) -> std::optional<double> {
    if (!attr || attr.getNumElements() != 1) {
      return std::nullopt;
    }
    auto et = attr.getElementType();
    if (isa<FloatType>(et)) {
      return annc::apFloatToDouble(attr.getValues<APFloat>()[0]);
    }
    if (isa<IntegerType, IndexType>(et)) {
      return static_cast<double>(attr.getValues<APInt>()[0].getSExtValue());
    }
    return std::nullopt;
  };

  DenseElementsAttr sAttr = startTy.getCacheData();
  DenseElementsAttr lAttr = limitTy.getCacheData();
  DenseElementsAttr dAttr = deltaTy.getCacheData();
  if (!sAttr || !lAttr || !dAttr) {
    (void)setSingleResultShape(getOperation(), {ShapedType::kDynamic});
    return;
  }

  auto start = readScalarDouble(sAttr);
  auto limit = readScalarDouble(lAttr);
  auto delta = readScalarDouble(dAttr);
  if (!start || !limit || !delta) {
    (void)setSingleResultShape(getOperation(), {ShapedType::kDynamic});
    return;
  }

  int64_t len =
      inferStaticRangeOutputDim(*start, *limit, *delta, 1LL << 28);
  if (len < 0) {
    if (len == -1) {
      emitError("Range delta must be non-zero for shape inference");
    }
    (void)setSingleResultShape(getOperation(), {ShapedType::kDynamic});
    return;
  }

  (void)setSingleResultShape(getOperation(), {len});
}

void ParameterOp::inferShape() {}

void ScatterOp::inferShape() {
  auto baseTy =
      dyn_cast<atir::TensorType>(getOperation()->getOperand(0).getType());
  if (!baseTy) {
    emitError("Scatter operand must be atir::TensorType");
    return;
  }
  (void)setSingleResultShape(getOperation(), baseTy.getShape());
}

void SumOp::inferShape() {
  auto inputType = dyn_cast<atir::TensorType>(getInput().getType());
  auto indicesType = dyn_cast<atir::TensorType>(getIndices().getType());
  auto resultType = dyn_cast<atir::TensorType>(getOutput().getType());
  if (!inputType || !indicesType || !resultType) {
    return;
  }

  auto inputShape = inputType.getShape();
  llvm::SmallVector<int64_t> axes;
  auto indicesAttr = indicesType.getCacheData();
  if (indicesAttr && !indicesAttr.empty()) {
    for (const APInt &value : indicesAttr.getValues<APInt>()) {
      int64_t axis = value.getSExtValue();
      if (axis < 0) {
        axis += static_cast<int64_t>(inputShape.size());
      }
      axes.push_back(axis);
    }
  }

  if (axes.empty()) {
    auto inferredAxes =
        inferReductionAxesFromShapes(inputShape, resultType.getShape());
    axes.append(inferredAxes.begin(), inferredAxes.end());
  }

  llvm::SmallVector<int64_t> outputShape;
  bool keepDims = getKeepDims() ||
                  resultType.getShape().size() == inputShape.size();
  if (keepDims) {
    outputShape.assign(inputShape.begin(), inputShape.end());
    for (int64_t axis : axes) {
      if (axis >= 0 && axis < static_cast<int64_t>(outputShape.size())) {
        outputShape[axis] = 1;
      }
    }
  } else {
    if (axes.empty()) {
      for (int64_t axis = 0; axis < static_cast<int64_t>(inputShape.size()); ++axis) {
        axes.push_back(axis);
      }
    }
    for (size_t axis = 0; axis < inputShape.size(); ++axis) {
      if (!llvm::is_contained(axes, static_cast<int64_t>(axis))) {
        outputShape.push_back(inputShape[axis]);
      }
    }
  }

  (void)setSingleResultShape(getOperation(), outputShape);
}

void ProdOp::inferShape() {
  auto inputType = dyn_cast<atir::TensorType>(getInput().getType());
  auto indicesType = dyn_cast<atir::TensorType>(getIndices().getType());
  auto resultType = dyn_cast<atir::TensorType>(getOutput().getType());
  if (!inputType || !indicesType || !resultType) {
    return;
  }

  auto inputShape = inputType.getShape();
  llvm::SmallVector<int64_t> axes;
  auto indicesAttr = indicesType.getCacheData();
  if (indicesAttr && !indicesAttr.empty()) {
    for (const APInt &value : indicesAttr.getValues<APInt>()) {
      int64_t axis = value.getSExtValue();
      if (axis < 0) {
        axis += static_cast<int64_t>(inputShape.size());
      }
      axes.push_back(axis);
    }
  }

  if (axes.empty()) {
    auto inferredAxes =
        inferReductionAxesFromShapes(inputShape, resultType.getShape());
    axes.append(inferredAxes.begin(), inferredAxes.end());
  }

  llvm::SmallVector<int64_t> outputShape;
  bool keepDims = getKeepDims() ||
                  resultType.getShape().size() == inputShape.size();
  if (keepDims) {
    outputShape.assign(inputShape.begin(), inputShape.end());
    for (int64_t axis : axes) {
      if (axis >= 0 && axis < static_cast<int64_t>(outputShape.size())) {
        outputShape[axis] = 1;
      }
    }
  } else {
    if (axes.empty()) {
      for (int64_t axis = 0; axis < static_cast<int64_t>(inputShape.size()); ++axis) {
        axes.push_back(axis);
      }
    }
    for (size_t axis = 0; axis < inputShape.size(); ++axis) {
      if (!llvm::is_contained(axes, static_cast<int64_t>(axis))) {
        outputShape.push_back(inputShape[axis]);
      }
    }
  }

  (void)setSingleResultShape(getOperation(), outputShape);
}

void GatherOp::inferShape() {
  auto paramsType = dyn_cast<atir::TensorType>(getParams().getType());
  auto indicesType = dyn_cast<atir::TensorType>(getIndices().getType());
  auto axisType = dyn_cast<atir::TensorType>(getAxis().getType());
  if (!paramsType || !indicesType || !axisType || !axisType.getCacheData()) {
    return;
  }

  auto axisValues = axisType.getCacheData().getValues<APInt>();
  if (axisType.getCacheData().getNumElements() != 1) {
    emitError("Gather axis must be scalar");
    return;
  }

  int64_t axis = axisValues[0].getSExtValue();
  auto paramsShape = paramsType.getShape();
  auto indicesShape = indicesType.getShape();
  int64_t rank = static_cast<int64_t>(paramsShape.size());
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) {
    emitError("Gather axis out of range");
    return;
  }
  int64_t batchDims = getBatchDims();
  int64_t indicesRank = static_cast<int64_t>(indicesShape.size());
  if (batchDims < 0) {
    emitError("Gather batch_dims must be non-negative");
    return;
  }
  if (indicesRank == 0) {
    if (batchDims != 0) {
      emitError("Gather with scalar indices requires batch_dims == 0");
      return;
    }
  } else if (!(batchDims < indicesRank)) {
    emitError("Gather batch_dims must be less than indices rank");
    return;
  }
  if (batchDims > axis) {
    emitError("Gather batch_dims must not exceed axis");
    return;
  }
  for (int64_t i = 0; i < batchDims; ++i) {
    int64_t ps = paramsShape[i];
    int64_t is = indicesShape[i];
    if (ps >= 0 && is >= 0 && ps != is) {
      emitError(
          "Gather params and indices must have identical shapes in the batch "
          "dimensions");
      return;
    }
  }
  llvm::SmallVector<int64_t> outputShape;
  outputShape.append(paramsShape.begin(), paramsShape.begin() + axis);
  outputShape.append(indicesShape.begin() + batchDims, indicesShape.end());
  outputShape.append(paramsShape.begin() + axis + 1, paramsShape.end());
  (void)setSingleResultShape(getOperation(), outputShape);
}

void GatherNdOp::inferShape() {
  auto valuesType = dyn_cast<atir::TensorType>(getValues().getType());
  auto indicesType = dyn_cast<atir::TensorType>(getIndices().getType());
  if (!valuesType || !indicesType) {
    emitError("GatherNd inputs must be atir::TensorType");
    return;
  }
  
  auto valuesShape = valuesType.getShape();
  auto indicesShape = indicesType.getShape();
  
  if (indicesShape.empty()) {
    emitError("GatherNd indices must have at least one dimension");
    return;
  }
  
  // indices 的最后一个维度指定要收集的轴数
  int64_t indicesRank = static_cast<int64_t>(indicesShape.size());
  int64_t valuesRank = static_cast<int64_t>(valuesShape.size());
  int64_t lastIndexDim = indicesShape[indicesRank - 1];
  
  if (lastIndexDim > valuesRank) {
    emitError("GatherNd lastIndexDim must not exceed values rank");
    return;
  }
  
  // 输出形状: indicesShape[:-1] + valuesShape[lastIndexDim:]
  SmallVector<int64_t> outputShape;
  for (int64_t i = 0; i < indicesRank - 1; ++i) {
    outputShape.push_back(indicesShape[i]);
  }
  for (int64_t i = lastIndexDim; i < valuesRank; ++i) {
    outputShape.push_back(valuesShape[i]);
  }
  
  (void)setSingleResultShape(getOperation(), outputShape);
}

void SliceOp::inferShape() {
  auto inputType = dyn_cast<atir::TensorType>(getInput().getType());
  auto beginTypepe = dyn_cast<atir::TensorType>(getBegin().getType());
  auto sizeType = dyn_cast<atir::TensorType>(getSize().getType());

  if (!inputType || !beginTypepe || !sizeType) {
    emitError("Slice inputs must be atir::TensorType");
    return;
  }

  auto inputShape = inputType.getShape();
  auto beginShape = beginTypepe.getShape();
  auto sizeShape = sizeType.getShape();

  if (beginShape.size() != 1 || sizeShape.size() != 1) {
    emitError("Slice begin and size must be 1D tensors");
    return;
  }

  if (beginShape[0] != static_cast<int64_t>(inputShape.size()) ||
      sizeShape[0] != static_cast<int64_t>(inputShape.size())) {
    emitError("Slice begin and size length must match input rank");
    return;
  }

  SmallVector<int64_t> outputShape(inputShape.begin(), inputShape.end());
  (void)setSingleResultShape(getOperation(), outputShape);
}

void LoadOp::inferShape() {
  auto inputType = dyn_cast<atir::TensorType>(getInput().getType());
  if (!inputType) {
    emitError("Load input must be atir::TensorType");
    return;
  }
  auto inputShape = inputType.getShape();
  auto axes = getIntArrayAttrValues(getAxes());
  auto sizes = getIntArrayAttrValues(getSize());
  if (axes.size() != sizes.size()) {
    emitError("Load axes and size must have the same length");
    return;
  }
  llvm::SmallVector<int64_t> outputShape(inputShape.begin(), inputShape.end());
  for (auto [axis, size] : llvm::zip(axes, sizes)) {
    if (axis < 0 || axis >= static_cast<int64_t>(outputShape.size())) {
      emitError("Load axis out of range");
      return;
    }
    outputShape[axis] = size;
  }
  (void)setSingleResultShape(getOperation(), outputShape);
}

void StridedSliceOp::inferShape() {
  auto inputType = dyn_cast<atir::TensorType>(getInput().getType());
  auto beginType = dyn_cast<atir::TensorType>(getBegin().getType());
  auto endType = dyn_cast<atir::TensorType>(getEnd().getType());
  auto stridesType = dyn_cast<atir::TensorType>(getStrides().getType());

  if (!inputType || !beginType || !endType || !stridesType) {
    emitError("StridedSlice begin/end/strides must be atir::TensorType");
    return;
  }
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();

  int64_t Ltensor = beginType.getShape().empty() ? int64_t(-1) : beginType.getShape()[0];
  SmallVector<int64_t> beginVals, endVals, strideVals;
  bool haveIdx = readStridedSlice1DInts(beginType, beginVals) &&
                 readStridedSlice1DInts(endType, endVals) &&
                 readStridedSlice1DInts(stridesType, strideVals);
  auto runWithVectors = [&](llvm::ArrayRef<int64_t> b, llvm::ArrayRef<int64_t> e,
                            llvm::ArrayRef<int64_t> s) -> bool {
    StridedSliceDensePlan plan;
    std::string err;
    if (!buildStridedSliceDensePlan(inputShape, b, e, s, getBeginMask(),
                                    getEndMask(), getEllipsisMask(),
                                    getNewAxisMask(), getShrinkAxisMask(), plan,
                                    &err)) {
      emitError(err);
      return false;
    }
    SmallVector<int64_t> outputShape;
    inferStridedSliceShapes(inputShape, plan, outputShape);
    return succeeded(setSingleResultShape(getOperation(), outputShape));
  };

  if (haveIdx) {
    (void)runWithVectors(beginVals, endVals, strideVals);
    return;
  }

  if (Ltensor >= 0 && Ltensor < 32) {
    SmallVector<int64_t> zb(Ltensor, 0), ze(Ltensor, 0), zs(Ltensor, 1);
    StridedSliceDensePlan plan;
    std::string err;
    if (!buildStridedSliceDensePlan(inputShape, zb, ze, zs, getBeginMask(),
                                    getEndMask(), getEllipsisMask(),
                                    getNewAxisMask(), getShrinkAxisMask(), plan,
                                    &err)) {
      emitError(err);
      return;
    }
    int64_t outRank = 0;
    for (int64_t g : plan.final_gather) {
      if (g != kStridedSliceShrinkGather)
        ++outRank;
    }
    SmallVector<int64_t> dynamicShape(outRank, ShapedType::kDynamic);
    (void)setSingleResultShape(getOperation(), dynamicShape);
    return;
  }

  emitError(
      "StridedSlice needs static 1-D spec length or constant begin/end/strides");
}

void MatMulOp::inferShape() {
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();
  if (!lhsType || !rhsType) {
    emitError("Matmul operands must have defining ops");
    return;
  }
  auto lhsTensorType = dyn_cast<atir::TensorType>(lhsType);
  auto rhsTensorType = dyn_cast<atir::TensorType>(rhsType);

  llvm::ArrayRef<int64_t> lhsShape = lhsTensorType.getShape();
  llvm::ArrayRef<int64_t> rhsShape = rhsTensorType.getShape();
  if (lhsShape.size() != 2 || rhsShape.size() != 2) {
    emitError("Matmul inputs must have rank == 2, batchMatmul not yet supported");
    return;
  }

  int64_t lhsRows = getLeftTranspose() ? lhsShape[1] : lhsShape[0];
  int64_t lhsCols = getLeftTranspose() ? lhsShape[0] : lhsShape[1];
  int64_t rhsRows = getRightTranspose() ? rhsShape[1] : rhsShape[0];
  int64_t rhsCols = getRightTranspose() ? rhsShape[0] : rhsShape[1];
  if (lhsCols != rhsRows) {
    emitError("Matmul lhsShape[1] != rhsShape[0]");
    return;
  }
  llvm::SmallVector<int64_t, 2> outputShape = {lhsRows, rhsCols};
  if (getOutputTranspose()) { 
    outputShape = {rhsCols, lhsRows}; 
  }
  (void)setSingleResultShape(getOperation(), outputShape);
}

void BatchMatMulOp::inferShape() {}

void DotOp::inferShape() {
  auto lhsType = dyn_cast<atir::TensorType>(getLhs().getType());
  auto rhsType = dyn_cast<atir::TensorType>(getRhs().getType());
  if (!lhsType || !rhsType) {
    emitError("Dot operands must be atir::TensorType");
    return;
  }
  auto lsh = lhsType.getShape();
  auto rsh = rhsType.getShape();
  if (lsh.empty() || rsh.empty()) {
    emitError("Dot expects rank>=1 operands");
    return;
  }
  int64_t lLast = lsh.back();
  int64_t rFirst = rsh[0];
  if (lLast != rFirst && lLast != ShapedType::kDynamic &&
      rFirst != ShapedType::kDynamic) {
    emitError("Dot contracting dimension mismatch");
    return;
  }
  llvm::SmallVector<int64_t> outShape;
  outShape.append(lsh.begin(), lsh.end() - 1);
  outShape.append(rsh.begin() + 1, rsh.end());
  (void)setSingleResultShape(getOperation(), outShape);
}
void ReshapeOp::inferShape() {
  auto targetAttr = getTargetShapeAttr();
  if (!targetAttr) {
    return;
  }
  llvm::SmallVector<int64_t> outShape;
  for (Attribute a : targetAttr) {
    if (auto intAttr = dyn_cast<IntegerAttr>(a)) {
      outShape.push_back(intAttr.getInt());
      continue;
    }
    emitError("Reshape targetShape must be integer array");
    return;
  }
  (void)setSingleResultShape(getOperation(), outShape);
}
void TransposeOp::inferShape() {
  auto inputType = dyn_cast<atir::TensorType>(getInput().getType());
  if (!inputType) {
    emitError("Transpose input must be atir::TensorType");
    return;
  }
  auto inShape = inputType.getShape();
  int64_t rank = static_cast<int64_t>(inShape.size());
  auto permAttr = getPermutationAttr();
  if (!permAttr) {
    emitError("Transpose missing permutation");
    return;
  }
  llvm::SmallVector<int64_t> perm;
  perm.reserve(permAttr.size());
  for (Attribute a : permAttr) {
    auto ia = dyn_cast<IntegerAttr>(a);
    if (!ia) {
      emitError("Transpose permutation must be integer attributes");
      return;
    }
    perm.push_back(ia.getInt());
  }
  if (static_cast<int64_t>(perm.size()) != rank) {
    emitError("Transpose permutation length must match input rank");
    return;
  }
  llvm::SmallVector<int64_t> outShape(rank, ShapedType::kDynamic);
  llvm::SmallVector<bool> seen(rank, false);
  for (int64_t i = 0; i < rank; ++i) {
    int64_t p = perm[i];
    if (p < 0) p += rank;
    if (p < 0 || p >= rank) {
      emitError("Transpose permutation out of range");
      return;
    }
    if (seen[p]) {
      emitError("Transpose permutation must be bijective");
      return;
    }
    seen[p] = true;
    outShape[i] = inShape[p];
  }
  (void)setSingleResultShape(getOperation(), outShape);
}
void ExpandDimsOp::inferShape() {
  auto inputType = getInput().getType();
  if (!inputType) {
    emitError("ExpandDims input must have defining op");
    return;
  }
  auto inputTensorType = dyn_cast<atir::TensorType>(inputType);
  if (!inputTensorType) {
    emitError("Expected TensorType for input, got: ") << inputType;
    return;
  }
  llvm::ArrayRef<int64_t> inputShape = inputTensorType.getShape();
  int32_t axisAttr = getAxis();
  int64_t axis = axisAttr;
  int64_t rank = inputShape.size();
  if (axis < 0) axis = rank + axis + 1;
  if (axis < 0 || axis > rank) {
    emitError("axis must be in range [0, ") << rank << "]";
    return;
  }
  // 在指定axis位置插入大小为1的新维度
  llvm::SmallVector<int64_t> outputShape;
  for (int64_t i = 0; i <= rank; ++i) {
    if (i == axis) outputShape.push_back(1);  // 插入新维度
    if (i < rank) outputShape.push_back(inputShape[i]);  // 复制原始维度
  } 
  (void)setSingleResultShape(getOperation(), outputShape);
}
void TileOp::inferShape() {
  auto inputType = dyn_cast<atir::TensorType>(getInput().getType());
  auto multiplesType = dyn_cast<atir::TensorType>(getMultiples().getType());
  if (!inputType || !multiplesType || !multiplesType.getCacheData()) return;
  auto inputShape = inputType.getShape();
  auto multiplesAttr = multiplesType.getCacheData();
  llvm::SmallVector<int64_t> multiples;
  for (const APInt &value : multiplesAttr.getValues<APInt>()) {
    multiples.push_back(value.getSExtValue());
  }
  if (multiples.size() != inputShape.size()) {
    emitError("Tile multiples rank mismatch");
    return;
  }
  llvm::SmallVector<int64_t> outputShape;
  for (auto [dim, multiple] : llvm::zip(inputShape, multiples)) {
    outputShape.push_back(dim * multiple);
  }
  (void)setSingleResultShape(getOperation(), outputShape);
}

void BroadcastOp::inferShape() {
  auto outputType = dyn_cast<atir::TensorType>(getOutput().getType());
  auto inputType = dyn_cast<atir::TensorType>(getInput().getType());
  if (!outputType || !inputType) {
    emitError("broadcast input/output must be atir::TensorType");
    return;
  }
  SmallVector<int64_t> outShape;
  for (int64_t d : getBroadcastSizes()) {
    outShape.push_back(d);
  }
  if (outShape.empty()) {
    outShape.assign(outputType.getShape().begin(), outputType.getShape().end());
  }
  if (!outShape.empty()) {
    getOutput().setType(cloneWithShape(outputType, outShape));
  }
}

void PadOp::inferShape() {
  auto inputType = dyn_cast<atir::TensorType>(getInput().getType());
  auto paddingsType = dyn_cast<atir::TensorType>(getPaddings().getType());
  if (!inputType || !paddingsType) {
    emitError("Pad input/paddings must be atir::TensorType");
    return;
  }
  auto inputShape = inputType.getShape();
  auto outputType = dyn_cast<atir::TensorType>(getOutput().getType());
  if (!outputType) {
    emitError("Pad output must be atir::TensorType");
    return;
  }
  if (auto paddingsAttr = paddingsType.getCacheData()) {
    SmallVector<int64_t> paddings;
    auto elemTy = paddingsAttr.getElementType();
    if (isa<IntegerType, IndexType>(elemTy)) {
      for (const APInt &v : paddingsAttr.getValues<APInt>()) {
        paddings.push_back(v.getSExtValue());
      }
    } else if (isa<FloatType>(elemTy)) {
      for (const APFloat &v : paddingsAttr.getValues<APFloat>()) {
        paddings.push_back(
            static_cast<int64_t>(annc::apFloatToDouble(v)));
      }
    }
    if (paddings.size() == inputShape.size() * 2) {
      SmallVector<int64_t> outputShape;
      outputShape.reserve(inputShape.size());
      for (size_t i = 0; i < inputShape.size(); ++i) {
        int64_t d = inputShape[i];
        int64_t before = paddings[2 * i];
        int64_t after = paddings[2 * i + 1];
        if (d == ShapedType::kDynamic || before < 0 || after < 0) {
          outputShape.push_back(ShapedType::kDynamic);
        } else {
          outputShape.push_back(d + before + after);
        }
      }
      getOutput().setType(cloneWithShape(outputType, outputShape));
      return;
    }
  }
  getOutput().setType(cloneWithShape(
      outputType,
      SmallVector<int64_t>(inputShape.size(), ShapedType::kDynamic)));
}

void SparseToDenseOp::inferShape() {
  auto outputShapeType = dyn_cast<atir::TensorType>(getOutputShape().getType());
  auto outputType = dyn_cast<atir::TensorType>(getOutput().getType());
  if (!outputShapeType || !outputType) {
    emitError("SparseToDense expects tensor outputShape and output");
    return;
  }
  if (DenseElementsAttr shapeAttr = outputShapeType.getCacheData()) {
    SmallVector<int64_t> denseShape;
    auto elemTy = shapeAttr.getElementType();
    if (isa<IntegerType, IndexType>(elemTy)) {
      for (const APInt &v : shapeAttr.getValues<APInt>()) {
        denseShape.push_back(v.getSExtValue());
      }
    } else if (isa<FloatType>(elemTy)) {
      for (const APFloat &v : shapeAttr.getValues<APFloat>()) {
        denseShape.push_back(
            static_cast<int64_t>(annc::apFloatToDouble(v)));
      }
    }
    if (!denseShape.empty()) {
      getOutput().setType(cloneWithShape(outputType, denseShape));
    }
  }
}

void SparseReshapeOp::inferShape() {
  auto indicesType = dyn_cast<atir::TensorType>(getIndices().getType());
  auto newShapeType = dyn_cast<atir::TensorType>(getNewShape().getType());
  auto outIndicesType = dyn_cast<atir::TensorType>(getOutputIndices().getType());
  auto outShapeType = dyn_cast<atir::TensorType>(getOutputShape().getType());
  if (!indicesType || !newShapeType || !outIndicesType || !outShapeType) {
    emitError("SparseReshape operands/results must be atir::TensorType");
    return;
  }
  auto indicesShape = indicesType.getShape();
  auto newShapeShape = newShapeType.getShape();
  if (indicesShape.size() != 2 || newShapeShape.size() != 1) {
    emitError("SparseReshape expects indices rank-2 and newShape rank-1");
    return;
  }
  int64_t nnz = indicesShape[0];
  int64_t outRank = newShapeShape[0];
  SmallVector<int64_t> outIndicesShape = {nnz, outRank};
  SmallVector<int64_t> outShapeShape = {outRank};
  getResult(0).setType(cloneWithShape(outIndicesType, outIndicesShape));
  getResult(1).setType(cloneWithShape(outShapeType, outShapeShape));
}

void SparseFillEmptyRowsOp::inferShape() {
  auto indicesType = dyn_cast<atir::TensorType>(getIndices().getType());
  auto valuesType = dyn_cast<atir::TensorType>(getValues().getType());
  auto denseShapeType = dyn_cast<atir::TensorType>(getDenseShape().getType());
  if (!indicesType || !valuesType || !denseShapeType) {
    emitError("SparseFillEmptyRows operands must be atir::TensorType");
    return;
  }
  auto indicesShape = indicesType.getShape();
  if (indicesShape.size() != 2) {
    emitError("SparseFillEmptyRows indices must have rank-2");
    return;
  }
  int64_t idxRank = indicesShape[1];
  getOutputIndices().setType(
      cloneWithShape(getOutputIndices().getType(),
                      {ShapedType::kDynamic, idxRank}));
  getOutputValues().setType(
      cloneWithShape(getOutputValues().getType(), {ShapedType::kDynamic}));
  getEmptyRowIndicator().setType(
      cloneWithShape(getEmptyRowIndicator().getType(), {ShapedType::kDynamic}));
  getReverseIndexMap().setType(
      cloneWithShape(getReverseIndexMap().getType(), {indicesShape[0]}));
}

void SparseSegmentSumOp::inferShape() {
  auto inputType = dyn_cast<atir::TensorType>(getInput().getType());
  auto indicesType = dyn_cast<atir::TensorType>(getIndices().getType());
  auto segmentIdsType = dyn_cast<atir::TensorType>(getSegmentIds().getType());
  auto numSegType = dyn_cast<atir::TensorType>(getNumSegments().getType());
  if (!inputType || !indicesType || !segmentIdsType || !numSegType) {
    emitError("SparseSegmentSum operands must be atir::TensorType");
    return;
  }
  auto inputShape = inputType.getShape();
  if (inputShape.empty()) {
    emitError("SparseSegmentSum input rank must be >= 1");
    return;
  }
  auto inferRowsFromSegmentIds = [&](atir::TensorType segTy) -> int64_t {
    int64_t inferred = ShapedType::kDynamic;
    if (auto segmentIdsAttr = segTy.getCacheData();
        segmentIdsAttr && segmentIdsAttr.getNumElements() > 0) {
      int64_t maxSegment = -1;
      auto elemTy = segmentIdsAttr.getElementType();
      if (isa<IntegerType, IndexType>(elemTy)) {
        for (const APInt &v : segmentIdsAttr.getValues<APInt>()) {
          maxSegment = std::max(maxSegment, v.getSExtValue());
        }
        inferred = maxSegment + 1;
      } else if (isa<FloatType>(elemTy)) {
        for (const APFloat &v : segmentIdsAttr.getValues<APFloat>()) {
          maxSegment = std::max(
              maxSegment,
              static_cast<int64_t>(annc::apFloatToDouble(v)));
        }
        inferred = maxSegment + 1;
      }
    }
    return inferred;
  };
  int64_t inferredRows = inferRowsFromSegmentIds(segmentIdsType);
  int64_t explicitRows = ShapedType::kDynamic;
  if (auto nsAttr = numSegType.getCacheData();
      nsAttr && nsAttr.getNumElements() == 1) {
    int64_t v = 0;
    auto elemTy = nsAttr.getElementType();
    if (isa<IntegerType, IndexType>(elemTy)) {
      v = nsAttr.getValues<APInt>()[0].getSExtValue();
    } else if (isa<FloatType>(elemTy)) {
      v = static_cast<int64_t>(
          annc::apFloatToDouble(nsAttr.getValues<APFloat>()[0]));
    }
    if (v > 0) explicitRows = v;
  }
  int64_t numSegments = ShapedType::kDynamic;
  if (inferredRows != ShapedType::kDynamic &&
      explicitRows != ShapedType::kDynamic) {
    numSegments = std::max(inferredRows, explicitRows);
  } else if (explicitRows != ShapedType::kDynamic) {
    numSegments = explicitRows;
  } else {
    numSegments = inferredRows;
  }
  llvm::SmallVector<int64_t> outputShape;
  outputShape.push_back(numSegments);
  outputShape.append(inputShape.begin() + 1, inputShape.end());
  (void)setSingleResultShape(getOperation(), outputShape);
}

void SparseSegmentMeanOp::inferShape() {
  auto dataType = dyn_cast<atir::TensorType>(getData().getType());
  auto indicesType = dyn_cast<atir::TensorType>(getIndices().getType());
  auto segmentIdsType = dyn_cast<atir::TensorType>(getSegmentIds().getType());
  auto numSegType = dyn_cast<atir::TensorType>(getNumSegments().getType());
  if (!dataType || !indicesType || !segmentIdsType || !numSegType) {
    emitError("SparseSegmentMean operands must be atir::TensorType");
    return;
  }
  auto dataShape = dataType.getShape();
  if (dataShape.empty()) {
    emitError("SparseSegmentMean data rank must be >= 1");
    return;
  }
  auto inferRowsFromSegmentIds = [&](atir::TensorType segTy) -> int64_t {
    int64_t inferred = ShapedType::kDynamic;
    if (auto segmentIdsAttr = segTy.getCacheData();
        segmentIdsAttr && segmentIdsAttr.getNumElements() > 0) {
      int64_t maxSegment = -1;
      auto elemTy = segmentIdsAttr.getElementType();
      if (isa<IntegerType, IndexType>(elemTy)) {
        for (const APInt &v : segmentIdsAttr.getValues<APInt>()) {
          maxSegment = std::max(maxSegment, v.getSExtValue());
        }
        inferred = maxSegment + 1;
      } else if (isa<FloatType>(elemTy)) {
        for (const APFloat &v : segmentIdsAttr.getValues<APFloat>()) {
          maxSegment = std::max(
              maxSegment,
              static_cast<int64_t>(annc::apFloatToDouble(v)));
        }
        inferred = maxSegment + 1;
      }
    }
    return inferred;
  };
  int64_t inferredRows = inferRowsFromSegmentIds(segmentIdsType);
  int64_t explicitRows = ShapedType::kDynamic;
  if (auto nsAttr = numSegType.getCacheData();
      nsAttr && nsAttr.getNumElements() == 1) {
    int64_t v = 0;
    auto elemTy = nsAttr.getElementType();
    if (isa<IntegerType, IndexType>(elemTy)) {
      v = nsAttr.getValues<APInt>()[0].getSExtValue();
    } else if (isa<FloatType>(elemTy)) {
      v = static_cast<int64_t>(
          annc::apFloatToDouble(nsAttr.getValues<APFloat>()[0]));
    }
    if (v > 0) explicitRows = v;
  }
  int64_t numSegments = ShapedType::kDynamic;
  if (inferredRows != ShapedType::kDynamic &&
      explicitRows != ShapedType::kDynamic) {
    numSegments = std::max(inferredRows, explicitRows);
  } else if (explicitRows != ShapedType::kDynamic) {
    numSegments = explicitRows;
  } else {
    numSegments = inferredRows;
  }
  llvm::SmallVector<int64_t> outputShape;
  outputShape.push_back(numSegments);
  outputShape.append(dataShape.begin() + 1, dataShape.end());
  (void)setSingleResultShape(getOperation(), outputShape);
}
void SparseSegmentMinOp::inferShape() {
  auto inputType = dyn_cast<atir::TensorType>(getInput().getType());
  auto indicesType = dyn_cast<atir::TensorType>(getIndices().getType());
  auto segmentIdsType = dyn_cast<atir::TensorType>(getSegmentIds().getType());
  auto numSegType = dyn_cast<atir::TensorType>(getNumSegments().getType());
  if (!inputType || !indicesType || !segmentIdsType || !numSegType) {
    emitError("SparseSegmentMin operands must be atir::TensorType");
    return;
  }

  auto inputShape = inputType.getShape();
  if (inputShape.empty()) {
    emitError("SparseSegmentMin input rank must be >= 1");
    return;
  }

  auto inferRowsFromSegmentIds = [&](atir::TensorType segTy) -> int64_t {
    int64_t inferred = ShapedType::kDynamic;
    if (auto segmentIdsAttr = segTy.getCacheData();
        segmentIdsAttr && segmentIdsAttr.getNumElements() > 0) {
      int64_t maxSegment = -1;
      auto elemTy = segmentIdsAttr.getElementType();
      if (isa<IntegerType, IndexType>(elemTy)) {
        for (const APInt &v : segmentIdsAttr.getValues<APInt>()) {
          maxSegment = std::max(maxSegment, v.getSExtValue());
        }
        inferred = maxSegment + 1;
      } else if (isa<FloatType>(elemTy)) {
        for (const APFloat &v : segmentIdsAttr.getValues<APFloat>()) {
          maxSegment = std::max(
              maxSegment,
              static_cast<int64_t>(annc::apFloatToDouble(v)));
        }
        inferred = maxSegment + 1;
      }
    }
    return inferred;
  };

  int64_t inferredRows = inferRowsFromSegmentIds(segmentIdsType);
  int64_t explicitRows = ShapedType::kDynamic;
  if (auto nsAttr = numSegType.getCacheData();
      nsAttr && nsAttr.getNumElements() == 1) {
    int64_t v = 0;
    auto elemTy = nsAttr.getElementType();
    if (isa<IntegerType, IndexType>(elemTy)) {
      v = nsAttr.getValues<APInt>()[0].getSExtValue();
    } else if (isa<FloatType>(elemTy)) {
      v = static_cast<int64_t>(
          annc::apFloatToDouble(nsAttr.getValues<APFloat>()[0]));
    }
    if (v > 0) explicitRows = v;
  }

  int64_t numSegments = ShapedType::kDynamic;
  if (inferredRows != ShapedType::kDynamic &&
      explicitRows != ShapedType::kDynamic) {
    numSegments = std::max(inferredRows, explicitRows);
  } else if (explicitRows != ShapedType::kDynamic) {
    numSegments = explicitRows;
  } else {
    numSegments = inferredRows;
  }

  llvm::SmallVector<int64_t> outputShape;
  outputShape.push_back(numSegments);
  outputShape.append(inputShape.begin() + 1, inputShape.end());
  (void)setSingleResultShape(getOperation(), outputShape);
}
}  // namespace atir