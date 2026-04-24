#include <llvm/Support/Debug.h>
#include "Dialect/Pimp/PimpOps.h"

#include "Dialect/Pimp/Interfaces/ShapeInfer.cpp.inc"

namespace pimp {
#define UNREACHABLE_OP(info) emitError(info);

llvm::SmallVector<int64_t> computeBroadcastShape(mlir::Operation* op) {
  // todo  type shape
  auto lhsType = op->getOperand(0).getType();
  auto lhsTensorType = dyn_cast<pimp::TensorType>(lhsType);

  auto lhs_shape = lhsTensorType.getShape();
  auto out_shape = llvm::SmallVector<int64_t>(lhs_shape);

  if (op->getNumOperands() > 1) {
    for (size_t i = 1; i < op->getNumOperands(); ++i) {
      auto operandType = op->getOperand(i).getType();
      auto operandTensorType = dyn_cast<pimp::TensorType>(operandType);
      // Opdyn_cast_or_nullnull
      if (llvm::dyn_cast_or_null<pimp::NoneOp>(
              op->getOperand(i).getDefiningOp())) {
        continue;
      }

      auto hs_shape = operandTensorType.getShape();
      auto tmp_shape = llvm::SmallVector<int64_t>();
      for (auto it : llvm::zip_longest(llvm::reverse(out_shape),
                                       llvm::reverse(hs_shape))) {
        if (std::get<0>(it) && std::get<0>(it) != 1) {
          tmp_shape.push_back(std::get<0>(it).value());
        } else {
          if (std::get<1>(it))
            tmp_shape.push_back(std::get<1>(it).value());
          else
            tmp_shape.push_back(std::get<0>(it).value());
        }
      }
      out_shape = llvm::SmallVector<int64_t>(llvm::reverse(tmp_shape));
    }
  }
  return out_shape;
}

// todo shapeInfer
void inferEltwiseOpShape(mlir::Operation* op) {}

void AddOp::inferShape() {
  inferEltwiseOpShape(getOperation());
  auto out_shape = computeBroadcastShape(getOperation());
  auto result = this->getResult();
  auto resultType = result.getType();
  // todo pimp::TensorTypecloneWithShape
  pimp::TensorType resultTensor = pimp::TensorType::get(
      out_shape, resultType.getElementType(), resultType.getName(),
      resultType.getEncoding(), resultType.getStride(), resultType.getLayout(),
      resultType.getMemType(), resultType.getAddress(),
      resultType.getDeviceParallel(), resultType.getOnchipParallel(),
      resultType.getCacheData());
  result.setType(resultTensor);
  auto resultShape = result.getType().getShape();
  llvm::dbgs() << "after Add inferShape, resultShape M: " << resultShape[0]
               << ", N: " << resultShape[1] << "\n";
}

void MatMulOp::inferShape() {
  inferEltwiseOpShape(getOperation());
  // todo batch matmul
  // todo 2
  // todo 
  auto lhsType = getOperand(0).getType();
  auto rhsType = getOperand(1).getType();
  if (!lhsType || !rhsType) {
    emitError("Matmul operands must have defining ops");
    return;
  }

  auto lhsTensorType = dyn_cast<pimp::TensorType>(lhsType);
  auto rhsTensorType = dyn_cast<pimp::TensorType>(rhsType);
  if (!lhsTensorType || !rhsTensorType) {
    emitError("Expected TensorType for operand, got: ")
        << "lhsType: " << lhsType
        << " (TensorType: " << isa<pimp::TensorType>(lhsType) << ")\n"
        << "rhsType: " << rhsType
        << " (TensorType: " << isa<pimp::TensorType>(rhsType) << ")\n";
    return;
  }

  llvm::ArrayRef<int64_t> lhsShape = lhsTensorType.getShape();  // (M, K)
  llvm::ArrayRef<int64_t> rhsShape = rhsTensorType.getShape();  // (K, N)
  llvm::dbgs() << "before inferShape, Matmul lhsShape M: " << lhsShape[0]
               << ", N: " << lhsShape[1] << "\n";
  llvm::dbgs() << "before inferShape, Matmul rhsShape M: " << rhsShape[0]
               << ", N: " << rhsShape[1] << "\n";
  if (lhsShape.size() != 2 || lhsShape.size() != 2) {
    emitError(
        "Matmul inputs must have rank == 2, batchMatmul not yet supported");
    return;
  }
  llvm::dbgs() << "lhsShape M: " << lhsShape[0] << ", N: " << lhsShape[1]
               << " ----------\n";
  llvm::dbgs() << "rhsShape M: " << rhsShape[0] << ", N: " << rhsShape[1]
               << " ----------\n";

  int64_t M = lhsShape[0];
  int64_t K_a = lhsShape[1];
  int64_t K_b = rhsShape[0];
  int64_t N = rhsShape[1];
  if (K_a != K_b) {
    emitError("Matmul lhsShape[1] != rhsShape[0]");
    return;
  }

  llvm::SmallVector<int64_t, 2> outputShape = {M, N};
  auto result = this->getResult();
  auto resultType = result.getType();
  // todo pimp::TensorTypecloneWithShape
  pimp::TensorType resultTensor = pimp::TensorType::get(
      outputShape, resultType.getElementType(), resultType.getName(),
      resultType.getEncoding(), resultType.getStride(), resultType.getLayout(),
      resultType.getMemType(), resultType.getAddress(),
      resultType.getDeviceParallel(), resultType.getOnchipParallel(),
      resultType.getCacheData());
  result.setType(resultTensor);
  auto resultShape = result.getType().getShape();
  llvm::dbgs() << "after inferShape, Matmul resultShape M: " << resultShape[0]
               << ", N: " << resultShape[1] << "\n";
}

void ReluOp::inferShape() {
  // 
  if (this->getOperation()->getNumOperands() != 1) {
    mlir::emitError(getLoc(), "ReluOp must have exactly one operand.");
    return;
  }
  auto type = getOperand().getType();
  if (!type) {
    emitError("Relu operands must have defining ops");
    return;
  }

  auto tensorType = dyn_cast<pimp::TensorType>(type);
  if (!tensorType) {
    emitError("Expected TensorType for operand, got: ")
        << "type: " << type << " (TensorType: " << isa<pimp::TensorType>(type)
        << ")\n";
    return;
  }

  // 2. 
  llvm::ArrayRef<int64_t> inputShape = tensorType.getShape();  // (M, K)

  // 
  auto result = this->getResult();
  auto resultType = result.getType();
  // todo pimp::TensorTypecloneWithShape
  pimp::TensorType resultTensor = pimp::TensorType::get(
      inputShape, resultType.getElementType(), resultType.getName(),
      resultType.getEncoding(), resultType.getStride(), resultType.getLayout(),
      resultType.getMemType(), resultType.getAddress(),
      resultType.getDeviceParallel(), resultType.getOnchipParallel(),
      resultType.getCacheData());
  result.setType(resultTensor);
  auto resultShape = result.getType().getShape();
  llvm::dbgs() << "after Relu inferShape, resultShape M: " << resultShape[0]
               << ", N: " << resultShape[1] << "\n";
  inferEltwiseOpShape(getOperation());
}

void ConcatOp::inferShape() { inferEltwiseOpShape(getOperation()); }

void LoadOp::inferShape() { inferEltwiseOpShape(getOperation()); }

}  // namespace pimp