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

}  // namespace atir