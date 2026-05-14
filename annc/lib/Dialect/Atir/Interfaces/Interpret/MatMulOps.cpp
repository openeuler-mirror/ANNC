#include "Common.h"
#include "Dialect/Atir/AtirOps.h"

using namespace atir::interpret;

namespace atir {
void MatMulOp::Interpret() {
  this->inferShape();
  atir::TensorType lhsTensorType;
  atir::TensorType rhsTensorType;
  DenseElementsAttr lhsAttr;
  DenseElementsAttr rhsAttr;
  if (failed(getTensorTypeAndData(getOperation(), getLhs(), "MatMul lhs",
                               lhsTensorType, lhsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getRhs(), "MatMul rhs",
                               rhsTensorType, rhsAttr))) {
    return;
  }
  auto lhsValsOr = getFloatValues(lhsAttr);
  auto rhsValsOr = getFloatValues(rhsAttr);
  if (failed(lhsValsOr) || failed(rhsValsOr)) {
    emitOpError("MatMul only supports numeric input cacheData");
    return;
  }
  auto lhsShape = lhsTensorType.getShape();
  auto rhsShape = rhsTensorType.getShape();
  if (lhsShape.size() != 2 || rhsShape.size() != 2) {
    emitOpError("MatMul interpreter currently only supports rank-2 tensors");
    return;
  }
  bool lhsTranspose = getLeftTranspose();
  bool rhsTranspose = getRightTranspose();
  bool outputTranspose = getOutputTranspose();
  bool doRelu = getDoRelu();
  float reluLimit =
      annc::apFloatToFloat(cast<FloatAttr>(getReluLimitAttr()).getValue());
  int64_t lhsRows = lhsTranspose ? lhsShape[1] : lhsShape[0];
  int64_t lhsCols = lhsTranspose ? lhsShape[0] : lhsShape[1];
  int64_t rhsRows = rhsTranspose ? rhsShape[1] : rhsShape[0];
  int64_t rhsCols = rhsTranspose ? rhsShape[0] : rhsShape[1];
  if (lhsCols != rhsRows) {
    emitOpError("MatMul K dimension mismatch");
    return;
  }
  const auto &lhsVals = *lhsValsOr;
  const auto &rhsVals = *rhsValsOr;
  auto lhsAt = [&](int64_t i, int64_t j) -> float {
    return lhsTranspose ? lhsVals[j * lhsShape[1] + i]
                        : lhsVals[i * lhsShape[1] + j];
  };
  auto rhsAt = [&](int64_t i, int64_t j) -> float {
    return rhsTranspose ? rhsVals[j * rhsShape[1] + i]
                        : rhsVals[i * rhsShape[1] + j];
  };
  int64_t M = lhsRows;
  int64_t K = lhsCols;
  int64_t N = rhsCols;
  std::vector<float> result(M * N, 0.0f);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float value = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        value += lhsAt(i, k) * rhsAt(k, j);
      }
      result[i * N + j] = value;
    }
  }
  if (getWithBias() && getNumOperands() > 3 &&
      !isa<atir::NoneOp>(getOperand(3).getDefiningOp())) {
    atir::TensorType biasType;
    DenseElementsAttr biasAttr;
    if (failed(getTensorTypeAndData(getOperation(), getOperand(3), "MatMul bias",
                                    biasType, biasAttr))) {
      return;
    }
    auto biasValsOr = getFloatValues(biasAttr);
    if (failed(biasValsOr)) {
      emitOpError("MatMul bias must be numeric");
      return;
    }
    const auto &biasVals = *biasValsOr;
    auto biasShape = biasType.getShape();
    if (biasShape.empty()) {
      for (float &value : result) {
        value += biasVals[0];
      }
    } else if (biasShape.size() == 1 && biasShape[0] == N) {
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          result[i * N + j] += biasVals[j];
        }
      }
    } else if (biasShape.size() == 2 && biasShape[0] == M && biasShape[1] == N) {
      for (int64_t i = 0; i < M * N; ++i) {
        result[i] += biasVals[i];
      }
    } else {
      emitOpError("MatMul bias shape is not supported");
      return;
    }
  }
  for (float &value : result) {
    value = applyReluLimit(value, doRelu, reluLimit);
  }
  if (outputTranspose) {
    std::vector<float> transposed(M * N, 0.0f);
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        transposed[j * M + i] = result[i * N + j];
      }
    }
    result.swap(transposed);
  }
  auto resultType = this->getResult().getType();
  auto outputShape = resultType.getShape();
  (void)setDenseResult(resultType, outputShape, result);
}

void BatchMatMulOp::Interpret() {}

void DotOp::Interpret() {
  this->inferShape();
  atir::TensorType lhsType;
  atir::TensorType rhsType;
  DenseElementsAttr lhsAttr;
  DenseElementsAttr rhsAttr;
  if (failed(getTensorTypeAndData(getOperation(), getLhs(), "Dot lhs", lhsType,
                                  lhsAttr)) ||
      failed(getTensorTypeAndData(getOperation(), getRhs(), "Dot rhs", rhsType,
                                  rhsAttr))) {
    return;
  }
  auto lhsValsOr = getFloatValues(lhsAttr);
  auto rhsValsOr = getFloatValues(rhsAttr);
  if (failed(lhsValsOr) || failed(rhsValsOr)) {
    emitOpError("Dot only supports numeric input cacheData");
    return;
  }
  auto lsh = lhsType.getShape();
  auto rsh = rhsType.getShape();
  if (lsh.empty() || rsh.empty()) {
    emitOpError("Dot expects rank>=1 operands");
    return;
  }
  int64_t kDim = lsh.back();
  if (kDim < 0) {
    emitOpError("Dot contracting dimension must be static for interpretation");
    return;
  }
  auto resultType = getResult().getType();
  auto outputShape = resultType.getShape();
  int64_t outputSize = getElementCount(outputShape);
  if (outputSize < 0) {
    emitOpError("Dot output shape must be static for interpretation");
    return;
  }
  const auto &lhsVals = *lhsValsOr;
  const auto &rhsVals = *rhsValsOr;
  int64_t rankL = static_cast<int64_t>(lsh.size());
  int64_t lhsPrefix = rankL - 1;
  std::vector<float> result(outputSize, 0.0f);
  for (int64_t outFlat = 0; outFlat < outputSize; ++outFlat) {
    auto outMulti = getMultiIndex(outputShape, outFlat);
    float acc = 0.0f;
    for (int64_t k = 0; k < kDim; ++k) {
      SmallVector<int64_t, 8> li(lsh.begin(), lsh.end());
      li[rankL - 1] = k;
      SmallVector<int64_t, 8> ri(rsh.begin(), rsh.end());
      ri[0] = k;
      for (size_t t = 1; t < rsh.size(); ++t)
        ri[t] = outMulti[lhsPrefix + static_cast<int64_t>(t) - 1];
      for (int64_t d = 0; d < lhsPrefix; ++d)
        li[d] = outMulti[d];
      acc += lhsVals[getFlatIndex(lsh, li)] * rhsVals[getFlatIndex(rsh, ri)];
    }
    result[outFlat] = acc;
  }
  (void)setDenseResult(resultType, outputShape, result);
}
} // namespace atir