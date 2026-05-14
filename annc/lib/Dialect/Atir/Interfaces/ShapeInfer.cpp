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


}  // namespace atir