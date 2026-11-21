#include "Patterns/ShapeProof.h"

#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace {
// The pinned producer uses integer shape programs even for static models.
// Evaluate only the pure, bounded subset needed to prove broadcast extents.
std::optional<SmallVector<int64_t>> shapeValue(Value value,
                                               unsigned depth = 0) {
  if (depth > 64) return std::nullopt;
  Operation *op = value.getDefiningOp();
  if (!op) return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getNumElements() > 64 ||
      !isa<IntegerType>(type.getElementType()))
    return std::nullopt;
  auto integer = cast<IntegerType>(type.getElementType());
  if (integer.isUnsigned() || integer.getWidth() > 64) return std::nullopt;
  if (auto c = dyn_cast<stablehlo::ConstantOp>(op)) {
    auto ints = dyn_cast<DenseIntElementsAttr>(c.getValue());
    if (!ints) return std::nullopt;
    SmallVector<int64_t> result;
    for (APInt x : ints.getValues<APInt>()) {
      result.push_back(x.getBitWidth() == 1 ? x.getZExtValue()
                                            : x.getSExtValue());
    }
    return result;
  }
  if (auto dim = dyn_cast<stablehlo::GetDimensionSizeOp>(op)) {
    auto input = dyn_cast<RankedTensorType>(dim.getOperand().getType());
    if (input && !input.isDynamicDim(dim.getDimension()) &&
        llvm::isIntN(integer.getWidth(), input.getDimSize(dim.getDimension())))
      return SmallVector<int64_t>{input.getDimSize(dim.getDimension())};
    auto *def = dim.getOperand().getDefiningOp();
    if (def &&
        isa<stablehlo::DynamicReshapeOp, stablehlo::DynamicBroadcastInDimOp>(
            def)) {
      auto shape = shapeValue(def->getOperand(1), depth + 1);
      if (shape && dim.getDimension() < shape->size() &&
          llvm::isIntN(integer.getWidth(), (*shape)[dim.getDimension()]))
        return SmallVector<int64_t>{(*shape)[dim.getDimension()]};
    }
    return std::nullopt;
  }
  auto name = op->getName().getStringRef();
  auto operand = [&](unsigned i) {
    return shapeValue(op->getOperand(i), depth + 1);
  };
  if (name == "stablehlo.reshape" || name == "stablehlo.dynamic_reshape") {
    auto values = operand(0);
    if (!values || values->size() != static_cast<size_t>(type.getNumElements()))
      return std::nullopt;
    if (name == "stablehlo.dynamic_reshape") {
      auto target = operand(1);
      if (!target || !llvm::equal(*target, type.getShape()))
        return std::nullopt;
    }
    return values;
  }
  if (name == "stablehlo.convert") {
    auto values = operand(0);
    unsigned width = integer.getWidth();
    // Boolean conversion uses truth-value semantics, not signed truncation.
    if (!values || width == 1) return std::nullopt;
    for (int64_t x : *values)
      if (!llvm::isIntN(width, x)) return std::nullopt;
    return values;
  }
  if (auto cat = dyn_cast<stablehlo::ConcatenateOp>(op)) {
    if (cat.getDimension() != 0 || type.getRank() != 1) return std::nullopt;
    SmallVector<int64_t> result;
    for (Value v : op->getOperands()) {
      auto part = shapeValue(v, depth + 1);
      if (!part) return std::nullopt;
      llvm::append_range(result, *part);
    }
    return result;
  }
  if (auto slice = dyn_cast<stablehlo::DynamicSliceOp>(op)) {
    auto values = operand(0), start = operand(1);
    if (!values || !start || start->size() != 1 || type.getRank() != 1 ||
        op->getNumOperands() != 2)
      return std::nullopt;
    int64_t count = type.getNumElements();
    if (count > static_cast<int64_t>(values->size())) return std::nullopt;
    int64_t begin = std::max<int64_t>(
        0, std::min<int64_t>((*start)[0], values->size() - count));
    return SmallVector<int64_t>(values->begin() + begin,
                                values->begin() + begin + count);
  }
  if (auto slice = dyn_cast<stablehlo::SliceOp>(op)) {
    auto values = operand(0);
    if (!values || type.getRank() != 1 ||
        slice.getStrides() != ArrayRef<int64_t>{1})
      return std::nullopt;
    auto begin = slice.getStartIndices()[0], end = slice.getLimitIndices()[0];
    if (begin < 0 || end < begin || end > static_cast<int64_t>(values->size()))
      return std::nullopt;
    return SmallVector<int64_t>(values->begin() + begin, values->begin() + end);
  }
  if (auto compare = dyn_cast<stablehlo::CompareOp>(op)) {
    auto a = operand(0), b = operand(1);
    if (!a || !b || a->size() != b->size()) return std::nullopt;
    SmallVector<int64_t> result;
    for (auto [x, y] : llvm::zip(*a, *b)) {
      switch (compare.getComparisonDirection()) {
        case stablehlo::ComparisonDirection::EQ:
          result.push_back(x == y);
          break;
        case stablehlo::ComparisonDirection::NE:
          result.push_back(x != y);
          break;
        case stablehlo::ComparisonDirection::LT:
          result.push_back(x < y);
          break;
        case stablehlo::ComparisonDirection::LE:
          result.push_back(x <= y);
          break;
        case stablehlo::ComparisonDirection::GT:
          result.push_back(x > y);
          break;
        case stablehlo::ComparisonDirection::GE:
          result.push_back(x >= y);
          break;
      }
    }
    return result;
  }
  if (name == "stablehlo.select") {
    auto cond = operand(0);
    if (!cond || cond->empty()) return std::nullopt;
    if (llvm::all_of(*cond, [&](int64_t x) { return x == cond->front(); }))
      return operand(cond->front() ? 1 : 2);
    auto a = operand(1), b = operand(2);
    if (!a || !b || a->size() != cond->size() || b->size() != cond->size())
      return std::nullopt;
    SmallVector<int64_t> result;
    for (size_t i = 0; i < cond->size(); ++i)
      result.push_back((*cond)[i] ? (*a)[i] : (*b)[i]);
    return result;
  }
  if (name == "stablehlo.broadcast_in_dim" ||
      name == "stablehlo.dynamic_broadcast_in_dim") {
    auto values = operand(0);
    if (!values || values->empty()) return std::nullopt;
    if (values->size() == 1)
      return SmallVector<int64_t>(type.getNumElements(), values->front());
    if (values->size() == static_cast<size_t>(type.getNumElements()))
      return values;
    return std::nullopt;
  }
  if (auto select = dyn_cast<stablehlo::TorchIndexSelectOp>(op)) {
    if (select.getDim() != 0 || select.getBatchDims() != 0 ||
        type.getRank() != 1)
      return std::nullopt;
    auto values = operand(0), indices = operand(1);
    if (!values || !indices) return std::nullopt;
    SmallVector<int64_t> result;
    for (int64_t i : *indices) {
      if (i < 0 || i >= static_cast<int64_t>(values->size()))
        return std::nullopt;
      result.push_back((*values)[i]);
    }
    return result;
  }
  if (auto reduce = dyn_cast<stablehlo::ReduceOp>(op)) {
    if (op->getNumOperands() != 2 ||
        reduce.getDimensions() != ArrayRef<int64_t>{0} ||
        !llvm::hasSingleElement(reduce.getBody()))
      return std::nullopt;
    auto &block = reduce.getBody().front();
    if (block.getNumArguments() != 2 || block.getOperations().size() != 2)
      return std::nullopt;
    auto mul = dyn_cast<stablehlo::MulOp>(block.front());
    auto ret = dyn_cast<stablehlo::ReturnOp>(block.back());
    if (!mul || !ret || mul.getLhs() != block.getArgument(0) ||
        mul.getRhs() != block.getArgument(1) ||
        ret->getOperand(0) != mul.getResult())
      return std::nullopt;
    auto values = operand(0), init = operand(1);
    if (!values || !init || init->size() != 1) return std::nullopt;
    APInt result(integer.getWidth(), init->front(), true);
    for (int64_t x : *values) result *= APInt(integer.getWidth(), x, true);
    return SmallVector<int64_t>{result.getSExtValue()};
  }
  if (op->getNumOperands() == 2) {
    auto a = operand(0), b = operand(1);
    if (!a || !b || a->size() != b->size()) return std::nullopt;
    SmallVector<int64_t> result;
    for (auto [x, y] : llvm::zip(*a, *b)) {
      APInt lhs(integer.getWidth(), x, true), rhs(integer.getWidth(), y, true),
          r;
      if (name == "stablehlo.add")
        r = lhs + rhs;
      else if (name == "stablehlo.subtract")
        r = lhs - rhs;
      else if (name == "stablehlo.multiply")
        r = lhs * rhs;
      else if (name == "stablehlo.divide" && !rhs.isZero() &&
               !(lhs.isMinSignedValue() && rhs.isAllOnes()))
        r = lhs.sdiv(rhs);
      else if (name == "stablehlo.remainder" && !rhs.isZero() &&
               !(lhs.isMinSignedValue() && rhs.isAllOnes()))
        r = lhs.srem(rhs);
      else if (name == "stablehlo.and")
        r = lhs & rhs;
      else if (name == "stablehlo.or")
        r = lhs | rhs;
      else if (name == "stablehlo.xor")
        r = lhs ^ rhs;
      else
        return std::nullopt;
      result.push_back(r.getBitWidth() == 1 ? r.getZExtValue()
                                            : r.getSExtValue());
    }
    return result;
  }
  return std::nullopt;
}

// This proves trailing-dimension broadcasting. Dynamic extents are accepted
// only when the source dimension is one or both types leave it unknown.
bool broadcasts(Value actual, Value original) {
  if (actual == original) return true;
  Operation *op = actual.getDefiningOp();
  if (!op) return false;
  auto name = op->getName().getStringRef();
  if (name != "stablehlo.broadcast_in_dim" &&
      name != "stablehlo.dynamic_broadcast_in_dim")
    return false;
  if (op->getOperand(0) != original) return false;
  auto src = dyn_cast<RankedTensorType>(original.getType());
  auto dst = dyn_cast<RankedTensorType>(actual.getType());
  auto dims = op->getAttrOfType<DenseI64ArrayAttr>("broadcast_dimensions");
  // In the pinned MLIR revision inherent attributes can be stored as
  // properties.
  if (auto b = dyn_cast<stablehlo::BroadcastInDimOp>(op))
    dims = b.getBroadcastDimensionsAttr();
  if (auto b = dyn_cast<stablehlo::DynamicBroadcastInDimOp>(op))
    dims = b.getBroadcastDimensionsAttr();
  if (!src || !dst || src.getRank() > dst.getRank() || !dims ||
      static_cast<int64_t>(dims.size()) != src.getRank())
    return false;
  for (auto [i, axis] : llvm::enumerate(dims.asArrayRef())) {
    if (axis != dst.getRank() - src.getRank() + static_cast<int64_t>(i) ||
        (src.getDimSize(i) != 1 && src.getDimSize(i) != ShapedType::kDynamic &&
         dst.getDimSize(axis) != ShapedType::kDynamic &&
         src.getDimSize(i) != dst.getDimSize(axis)))
      return false;
  }
  if (name == "stablehlo.dynamic_broadcast_in_dim") {
    auto shape = shapeValue(op->getOperand(1));
    if (shape) {
      if (shape->size() != static_cast<size_t>(dst.getRank())) return false;
      for (auto [i, extent] : llvm::enumerate(*shape))
        if (!dst.isDynamicDim(i) && extent != dst.getDimSize(i)) return false;
    } else if (dst.hasStaticShape()) {
      return false;
    }
  }
  return true;
}
}  // namespace

namespace annc {
std::optional<SmallVector<int64_t>> evaluateStablehloShape(Value value) {
  return shapeValue(value);
}
bool proveStablehloBroadcast(Value actual, Value original) {
  return broadcasts(actual, original);
}
}  // namespace annc
