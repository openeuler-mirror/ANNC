#include <llvm/Support/Debug.h>
#include "Dialect/Pimp/PimpOps.h"

#include "Dialect/Pimp/Interfaces/Interpret.cpp.inc"

namespace pimp {

void InterpretAddOp() {
}

void InterpretMatMulOp() {
}

void InterpretReluOp() {
}

void AddOp::Interpret() {
  InterpretAddOp();
  // shapeInfer
  this->inferShape();
  // matmul operands  type
  auto lhsType = getOperand(0).getType();
  auto rhsType = getOperand(1).getType();
  if (!lhsType || !rhsType) {
    emitOpError("Add operands must have type");
    return;
  }

  auto lhsTensorType = dyn_cast<pimp::TensorType>(lhsType);
  auto rhsTensorType = dyn_cast<pimp::TensorType>(rhsType);
  if (!lhsTensorType || !rhsTensorType) {
    emitError("Expected TensorType for operand, got: ")
            << "lhsType: " << lhsType << " (TensorType: " << isa<pimp::TensorType>(lhsType) << ")\n"
            << "rhsType: " << rhsType << " (TensorType: " << isa<pimp::TensorType>(rhsType) << ")\n";
    return;
  }

  // 2.  lhs  rhs  value
  auto lhsAttr = lhsTensorType.getCacheData();
  auto rhsAttr = rhsTensorType.getCacheData();
  if (!lhsAttr) {
    emitError("Missing Add lhsAttr input cacheData attr on operands");
    return;
  }
  if (!rhsAttr) {
    emitError("Missing Add rhsAttr input cacheData attr on operands");
    return;
  }

  //  shape 
  // todo other data type
  auto lhsVals = lhsAttr.getValues<float>();
  auto rhsVals = rhsAttr.getValues<float>();
  auto lhsShape = lhsTensorType.getShape();
  auto rhsShape = rhsTensorType.getShape();

  auto resultType = this->getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outSize = 1;
  for (int64_t d : outputShape) outSize *= d;

  std::vector<float> result(outSize, 0.0f);

  // 
  auto getIndex = [](ArrayRef<int64_t> shape, int64_t flatIndex) {
    SmallVector<int64_t> idx(shape.size());
    for (int64_t i = shape.size() - 1; i >= 0; --i) {
      idx[i] = flatIndex % shape[i];
      flatIndex /= shape[i];
    }
    return idx;
  };

  auto getFlatIndex = [](ArrayRef<int64_t> shape, ArrayRef<int64_t> idx) {
    int64_t flat = 0;
    for (int64_t i = 0; i < (int64_t)shape.size(); ++i)
      flat = flat * shape[i] + idx[i];
    return flat;
  };

  // 
  for (int64_t outIdx = 0; outIdx < outSize; ++outIdx) {
    auto idx = getIndex(outputShape, outIdx);

    // lhs 
    SmallVector<int64_t> lhsIdx(idx);
    if (lhsShape.size() < idx.size())
      lhsIdx = SmallVector<int64_t>(idx.begin() + (idx.size() - lhsShape.size()), idx.end());
    for (int64_t i = 0; i < (int64_t)lhsShape.size(); ++i)
      if (lhsShape[i] == 1) lhsIdx[i] = 0;

    // rhs 
    SmallVector<int64_t> rhsIdx(idx);
    if (rhsShape.size() < idx.size())
      rhsIdx = SmallVector<int64_t>(idx.begin() + (idx.size() - rhsShape.size()), idx.end());
    for (int64_t i = 0; i < (int64_t)rhsShape.size(); ++i)
      if (rhsShape[i] == 1) rhsIdx[i] = 0;

    float lhsV = lhsVals[getFlatIndex(lhsShape, lhsIdx)];
    float rhsV = rhsVals[getFlatIndex(rhsShape, rhsIdx)];

    result[outIdx] = lhsV + rhsV;
  }

  //  op 
  mlir::Type elementType = resultType.getElementType();
  mlir::RankedTensorType standardTensorType = mlir::RankedTensorType::get(
          outputShape,
          elementType
  );
  auto resultAttr = mlir::DenseElementsAttr::get(standardTensorType, llvm::ArrayRef(result));
  resultType.setCacheData(resultAttr);
  llvm::dbgs() << "Add results:\n";
  resultType.getCacheData().print(llvm::dbgs());
  llvm::dbgs() << "\n";
}

void MatMulOp::Interpret() {
  // todo >50
  InterpretMatMulOp();
  // shapeInfer
  this->inferShape();
  auto lhsType = getOperand(0).getType();
  auto rhsType = getOperand(1).getType();
  if (!lhsType || !rhsType) {
    emitOpError("Matmul operands must have type");
    return;
  }

  auto lhsTensorType = dyn_cast<pimp::TensorType>(lhsType);
  auto rhsTensorType = dyn_cast<pimp::TensorType>(rhsType);
  if (!lhsTensorType || !rhsTensorType) {
    emitError("Expected TensorType for operand, got: ")
            << "lhsType: " << lhsType << " (TensorType: " << isa<pimp::TensorType>(lhsType) << ")\n"
            << "rhsType: " << rhsType << " (TensorType: " << isa<pimp::TensorType>(rhsType) << ")\n";
    return;
  }

  auto lhsAttr = lhsTensorType.getCacheData();
  auto rhsAttr = rhsTensorType.getCacheData();
  if (!lhsAttr || !rhsAttr) {
    emitError("Missing Matmul input cacheData attr on operands");
    return;
  }

  // todo other data type
  auto lhsVals = lhsAttr.getValues<float>();
  auto rhsVals = rhsAttr.getValues<float>();

  auto lhsShape = lhsAttr.getType().getShape();
  auto rhsShape = rhsAttr.getType().getShape();

  int64_t M = lhsShape[0], K = lhsShape[1], N = rhsShape[1];
  std::vector<float> result(M * N, 0.0f);

  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j)
      for (int64_t k = 0; k < K; ++k)
        result[i * N + j] += lhsVals[i * K + k] * rhsVals[k * N + j];

  auto resultType = this->getResult().getType();

  auto shape = resultType.getShape();
  llvm::dbgs() << "resultShape M: " << shape[0] << ", N: " << shape[1] << "\n";

  auto outputShape = this->getResult().getType().getShape();
  mlir::Type elementType = resultType.getElementType();
  mlir::RankedTensorType standardTensorType = mlir::RankedTensorType::get(
          outputShape,
          elementType
  );
  auto resultAttr = mlir::DenseElementsAttr::get(standardTensorType, llvm::ArrayRef(result));
  resultType.setCacheData(resultAttr);
  llvm::dbgs() << "Matmul results:\n";
  resultType.getCacheData().dump();
}

void ReluOp::Interpret() {
  // shapeInfer
  this->inferShape();
  InterpretReluOp();
  //  operand  type
  auto inputType = getOperand().getType();
  if (!inputType) {
    emitOpError("Relu operand must have type");
    return;
  }

  // 
  auto inputTensor = dyn_cast<pimp::TensorType>(inputType);
  auto inputAttr = inputTensor.getCacheData();
  if (!inputAttr) {
    emitError("Missing Relu input cacheData attr on operands");
    return;
  }

  // todo other dtype
  auto inputVals = inputAttr.getValues<float>();

  //  ReLU
  std::vector<float> result(inputVals.size());
  for (size_t i = 0; i < result.size(); ++i)
    result[i] = std::max(0.0f, inputVals[i]);

  //  op 
  auto resultType = this->getResult().getType();
  auto outputShape = resultType.getShape();
  mlir::Type elementType = resultType.getElementType();
  mlir::RankedTensorType standardTensorType = mlir::RankedTensorType::get(
          outputShape,
          elementType
  );
  auto resultAttr = mlir::DenseElementsAttr::get(standardTensorType, llvm::ArrayRef(result));
  resultType.setCacheData(resultAttr);
  llvm::dbgs() << "Relu results:\n";
  resultType.getCacheData().dump();
}

void ConstantOp::Interpret() {
  auto result = getResult();
  auto data = result.getType().getCacheData();

  if (data.empty()) {
    emitError("Error: current constant op data is empty! result: ");
    return;
  }
}

void ConcatOp::Interpret() {}

void LoadOp::Interpret() {}

void ExpandDimsOp::Interpret() {}

void IdentityOp::Interpret() {}

void ShapeOp::Interpret() {}

void NotEqualOp::Interpret() {}

void CastOp::Interpret() {}

void WhereOp::Interpret() {}

void StridedSliceOp::Interpret() {}

void PackOp::Interpret() {}

void GatherNdOp::Interpret() {}

void StringToHashBucketFastOp::Interpret() {}

void SparseReshapeOp::Interpret() {}

void SliceOp::Interpret() {}

void GatherV2Op::Interpret() {}

void ProdOp::Interpret() {}

void GreaterEqualOp::Interpret() {}

void SparseFillEmptyRowsOp::Interpret() {}

void UniqueOp::Interpret(){}

void SparseSegmentSumOp::Interpret() {}

void TileOp::Interpret() {}

void ZerosLikeOp::Interpret() {}

void SelectOp::Interpret() {}

void ConcatV2Op::Interpret() {}

void BatchMatMulV2Op::Interpret() {}

}  // namespace pimp