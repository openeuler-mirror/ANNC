#include "Patterns/BMMAdd.h"

#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "HloGraph.h"
#include "Patterns/ShapeProof.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
namespace annc::stablehlo {
namespace {
bool proveBMMAdd(Value data, Value weight, Value bias, Value product,
                 Value output) {
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType || (dataType.getRank() != 3 && dataType.getRank() != 4))
    return false;
  const int64_t rank = dataType.getRank();
  const int64_t batchRank = rank - 2;
  for (Value v : {data, weight, bias, product, output}) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t || t.getRank() != rank || !t.getElementType().isF32() ||
        llvm::is_contained(t.getShape(), 0))
      return false;
  }
  auto weightType = cast<RankedTensorType>(weight.getType());
  auto biasType = cast<RankedTensorType>(bias.getType());
  auto productType = cast<RankedTensorType>(product.getType());
  auto outputType = cast<RankedTensorType>(output.getType());
  // Existing BMMAdd kernels accept dynamic M, but batch extents and K/N
  // determine indexing, constants, and output allocation and stay static.
  if (!weightType.hasStaticShape() || !biasType.hasStaticShape() ||
      weightType.getDimSize(rank - 2) <= 0 ||
      weightType.getDimSize(rank - 1) <= 0)
    return false;
  for (int64_t axis = 0; axis < batchRank; ++axis) {
    if (dataType.isDynamicDim(axis) || productType.isDynamicDim(axis) ||
        outputType.isDynamicDim(axis))
      return false;
  }
  if (dataType.isDynamicDim(rank - 1) ||
      dataType.getDimSize(rank - 1) != weightType.getDimSize(rank - 2) ||
      productType.getDimSize(rank - 1) != weightType.getDimSize(rank - 1) ||
      outputType.getDimSize(rank - 1) != weightType.getDimSize(rank - 1) ||
      (!productType.isDynamicDim(rank - 2) &&
       !dataType.isDynamicDim(rank - 2) &&
       productType.getDimSize(rank - 2) != dataType.getDimSize(rank - 2)) ||
      (!outputType.isDynamicDim(rank - 2) &&
       !productType.isDynamicDim(rank - 2) &&
       outputType.getDimSize(rank - 2) != productType.getDimSize(rank - 2)))
    return false;
  for (int64_t axis = 0; axis < batchRank; ++axis) {
    const int64_t batch = outputType.getDimSize(axis);
    if (batch <= 0 || biasType.getDimSize(axis) != batch ||
        (dataType.getDimSize(axis) != 1 &&
         dataType.getDimSize(axis) != batch) ||
        (weightType.getDimSize(axis) != 1 &&
         weightType.getDimSize(axis) != batch))
      return false;
  }
  if (biasType.getDimSize(rank - 2) != 1 ||
      biasType.getDimSize(rank - 1) != weightType.getDimSize(rank - 1))
    return false;
  auto dot = product.getDefiningOp<mlir::stablehlo::DotGeneralOp>();
  auto add = output.getDefiningOp<mlir::stablehlo::AddOp>();
  auto w = weight.getDefiningOp<mlir::stablehlo::ConstantOp>();
  auto b = bias.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!dot || !add || !w || !b) return false;
  auto wData = dyn_cast<DenseElementsAttr>(w.getValue());
  auto bData = dyn_cast<DenseElementsAttr>(b.getValue());
  if (!wData || !bData) return false;
  auto dimensions = dot.getDotDimensionNumbers();
  SmallVector<int64_t> expectedBatchDims;
  for (int64_t axis = 0; axis < batchRank; ++axis)
    expectedBatchDims.push_back(axis);
  if (!llvm::equal(dimensions.getLhsBatchingDimensions(), expectedBatchDims) ||
      !llvm::equal(dimensions.getRhsBatchingDimensions(), expectedBatchDims) ||
      dimensions.getLhsContractingDimensions() != ArrayRef<int64_t>{rank - 1} ||
      dimensions.getRhsContractingDimensions() != ArrayRef<int64_t>{rank - 2} ||
      !proveStablehloBroadcast(dot.getLhs(), data) ||
      !proveStablehloBroadcast(dot.getRhs(), weight) ||
      !((proveStablehloBroadcast(add.getLhs(), product) &&
         proveStablehloBroadcast(add.getRhs(), bias)) ||
        (proveStablehloBroadcast(add.getRhs(), product) &&
         proveStablehloBroadcast(add.getLhs(), bias))))
    return false;
  // Require default precision and no algorithm override for this first path.
  if (dot.getAlgorithm() ||
      (dot.getPrecisionConfig() &&
       llvm::any_of(*dot.getPrecisionConfig(), [](Attribute attr) {
         return cast<mlir::stablehlo::PrecisionAttr>(attr).getValue() !=
                mlir::stablehlo::Precision::DEFAULT;
       })))
    return false;
  return true;
}

// Mirror the legacy pre-affine preparation: fold immutable scale/offset into
// private weight/bias payloads, while keeping the original ONNX boundary slots.
// This uses the legacy FP32 folding policy, not bitwise reassociation
// equivalence.
struct PreparedBMM {
  Value input;
  DenseElementsAttr weight;
  DenseElementsAttr bias;
};

// Called after proveBMMAdd has validated the weight and bias constants.
template <typename Unwrap>
std::optional<PreparedBMM> prepareBMM(Value data, Value weight, Value bias,
                                      Unwrap unwrap) {
  Value scaled = data, offset;
  if (auto add = data.getDefiningOp<mlir::stablehlo::AddOp>()) {
    for (unsigned side = 0; side < 2; ++side) {
      Value lhs = unwrap(add->getOperand(side));
      Value rhs = unwrap(add->getOperand(1 - side));
      if (lhs && rhs && lhs.getDefiningOp<mlir::stablehlo::MulOp>() &&
          rhs.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
        scaled = lhs;
        offset = rhs;
        break;
      }
    }
    if (!offset) return std::nullopt;
  }
  auto mul = scaled.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!mul) return std::nullopt;
  auto w = cast<mlir::stablehlo::ConstantOp>(weight.getDefiningOp());
  auto b = cast<mlir::stablehlo::ConstantOp>(bias.getDefiningOp());
  auto wt = cast<RankedTensorType>(weight.getType());
  auto bt = cast<RankedTensorType>(bias.getType());
  auto shape = wt.getShape();
  auto rank = shape.size();
  int64_t batches = 1;
  const int64_t kSize = shape[rank - 2], nSize = shape[rank - 1];
  for (size_t axis = 0; axis + 2 < rank; ++axis) {
    if (shape[axis] <= 0 || shape[axis] != bt.getDimSize(axis))
      return std::nullopt;
    if (batches > std::numeric_limits<int64_t>::max() / shape[axis])
      return std::nullopt;
    batches *= shape[axis];
  }
  auto acceptable = [&](DenseFPElementsAttr values) {
    if (!values || !values.getElementType().isF32() ||
        !llvm::all_of(values.getValues<float>(),
                      [](float v) { return std::isfinite(v); }))
      return false;
    auto dims = values.getType().getShape();
    if (dims.size() > rank) return false;
    for (size_t axis = 0; axis < dims.size(); ++axis) {
      size_t logical = rank - dims.size() + axis;
      int64_t expected = 1;
      if (logical + 2 < rank) expected = shape[logical];
      if (logical + 1 == rank) expected = kSize;
      if (dims[axis] != 1 && dims[axis] != expected) return false;
    }
    return true;
  };
  auto offsetOp =
      offset ? offset.getDefiningOp<mlir::stablehlo::ConstantOp>() : nullptr;
  auto offsetData = offsetOp
                        ? dyn_cast<DenseFPElementsAttr>(offsetOp.getValue())
                        : DenseFPElementsAttr{};
  if (offset && !acceptable(offsetData)) return std::nullopt;
  for (unsigned side = 0; side < 2; ++side) {
    Value input = unwrap(mul->getOperand(side));
    Value scale = unwrap(mul->getOperand(1 - side));
    auto op =
        scale ? scale.getDefiningOp<mlir::stablehlo::ConstantOp>() : nullptr;
    auto values = op ? dyn_cast<DenseFPElementsAttr>(op.getValue())
                     : DenseFPElementsAttr{};
    if (!input || input.getType() != data.getType() ||
        scaled.getType() != data.getType() ||
        input.getDefiningOp<mlir::stablehlo::ConstantOp>() ||
        !acceptable(values))
      continue;
    std::vector<float> weights(w.getValue().getValues<float>().begin(),
                               w.getValue().getValues<float>().end());
    std::vector<float> biases(b.getValue().getValues<float>().begin(),
                              b.getValue().getValues<float>().end());
    std::vector<float> scales(values.getValues<float>().begin(),
                              values.getValues<float>().end());
    std::vector<float> offsets;
    if (offsetData)
      offsets.assign(offsetData.getValues<float>().begin(),
                     offsetData.getValues<float>().end());
    auto index = [&](DenseFPElementsAttr data, int64_t batch, int64_t k) {
      SmallVector<int64_t> coord(rank, 0);
      coord[rank - 1] = k;
      for (int axis = int(rank) - 3; axis >= 0; --axis) {
        coord[axis] = batch % shape[axis];
        batch /= shape[axis];
      }
      auto dims = data.getType().getShape();
      int64_t result = 0;
      for (size_t axis = 0; axis < dims.size(); ++axis)
        result = result * dims[axis] +
                 (dims[axis] == 1 ? 0 : coord[rank - dims.size() + axis]);
      return result;
    };
    if (!llvm::all_of(weights, [](float v) { return std::isfinite(v); }) ||
        !llvm::all_of(biases, [](float v) { return std::isfinite(v); }))
      return std::nullopt;
    auto original = weights;
    for (int64_t batch = 0; batch < batches; ++batch)
      for (int64_t n = 0; n < nSize; ++n) {
        float value = biases[batch * nSize + n];
        for (int64_t k = 0; k < kSize; ++k) {
          auto j = (batch * kSize + k) * nSize + n;
          weights[j] = scales[index(values, batch, k)] * original[j];
          if (offsetData)
            value += offsets[index(offsetData, batch, k)] * original[j];
        }
        biases[batch * nSize + n] = value;
      }
    if (!llvm::all_of(weights, [](float v) { return std::isfinite(v); }) ||
        !llvm::all_of(biases, [](float v) { return std::isfinite(v); }))
      return std::nullopt;
    return PreparedBMM{
        input, DenseElementsAttr::get(wt, ArrayRef<float>(weights)),
        offsetData ? DenseElementsAttr::get(bt, ArrayRef<float>(biases))
                   : DenseElementsAttr{}};
  }
  return std::nullopt;
}
}  // namespace

void partitionBMMAdd(HloGraph &graph, SmallVectorImpl<Attribute> &rejections) {
  auto unwrap = [](Value value) {
    while (auto *op = value.getDefiningOp()) {
      if (!isa<mlir::stablehlo::BroadcastInDimOp,
               mlir::stablehlo::DynamicBroadcastInDimOp>(op))
        break;
      Value input = op->getOperand(0);
      if (!proveStablehloBroadcast(value, input)) return Value();
      value = input;
    }
    return value;
  };
  for (const auto &[id, op] : graph.operations) {
    auto add = dyn_cast<mlir::stablehlo::AddOp>(op);
    if (!add) continue;
    for (unsigned i = 0; i < 2; ++i) {
      Value product = unwrap(add->getOperand(i));
      auto dot = product
                     ? product.getDefiningOp<mlir::stablehlo::DotGeneralOp>()
                     : nullptr;
      Value bias = unwrap(add->getOperand(1 - i));
      if (!dot || !bias) continue;
      Value data = unwrap(dot.getLhs()), weight = unwrap(dot.getRhs());
      if (!data || !weight) continue;
      if (!proveBMMAdd(data, weight, bias, product, add.getResult())) continue;
      auto rank = cast<RankedTensorType>(data.getType()).getRank();
      StringRef builtin = rank == 4 ? "BatchedMatMulAdd4" : "BatchedMatMulAdd";
      if (auto prepared = prepareBMM(data, weight, bias, unwrap)) {
        SmallVector<Attribute> preparationRejections;
        if (graph.fuse("batched_matmul_add", builtin,
                       {prepared->input, weight, bias}, {add.getResult()}, {},
                       preparationRejections,
                       {{}, prepared->weight, prepared->bias}))
          break;
      }
      graph.fuse("batched_matmul_add", builtin, {data, weight, bias},
                 {add.getResult()}, {}, rejections);
      break;
    }
  }
}
}  // namespace annc::stablehlo
