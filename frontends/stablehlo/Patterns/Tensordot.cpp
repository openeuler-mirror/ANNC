#include "Patterns/Tensordot.h"

#include <algorithm>
#include <cstdint>
#include <optional>

#include "HloGraph.h"
#include "Patterns/ShapeProof.h"
#include "llvm/ADT/DenseMap.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
namespace {
// A small abstract interpreter for integer shape programs. A term is c, c*d0,
// c*d1 or c*d0*d1. As in legacy Tensordot matching, symbolic simplification
// assumes ordinary non-overflowing shape arithmetic. This is not a runtime
// size contract; the original kernel's shape/error handling is unchanged.
struct Extent {
  int64_t coefficient;
  unsigned axes = 0;
  int64_t variableBound = INT32_MAX;
  bool operator==(const Extent &other) const {
    return coefficient == other.coefficient && axes == other.axes;
  }
  int64_t minimum() const { return coefficient; }
  std::optional<int64_t> maximum() const {
    if (!axes || !coefficient) return coefficient;
    if (coefficient < 0 || coefficient > INT64_MAX / variableBound)
      return std::nullopt;
    return coefficient * variableBound;
  }
  bool fits(unsigned width) const {
    auto upper = maximum();
    return upper && llvm::isIntN(width, minimum()) &&
           llvm::isIntN(width, *upper);
  }
};
using Extents = SmallVector<Extent>;
using ShapeResult = std::optional<Extents>;
class TensordotShapeProof {
 public:
  explicit TensordotShapeProof(Value data) : data(data) {}
  Extent dimension(unsigned axis) {
    auto t = cast<RankedTensorType>(data.getType());
    if (!t.isDynamicDim(axis)) return {t.getDimSize(axis)};
    unsigned other = axis == 0 ? 1 : 0;
    int64_t bound =
        t.isDynamicDim(other) ? INT32_MAX : INT32_MAX / t.getDimSize(other);
    return {1, 1u << axis, bound};
  }
  ShapeResult evaluate(Value v, unsigned depth = 0) {
    if (depth > 128) return std::nullopt;
    auto found = cache.find(v);
    if (found != cache.end()) return found->second;
    auto result = compute(v, depth);
    cache.try_emplace(v, result);
    return result;
  }
  bool shape(Value v, ArrayRef<Extent> expected) {
    auto actual = evaluate(v);
    return actual && llvm::equal(*actual, expected);
  }
  bool reshape(Value v, ArrayRef<Extent> expected) {
    auto *op = v.getDefiningOp();
    if (!op) return false;
    if (isa<stablehlo::DynamicReshapeOp>(op))
      return shape(op->getOperand(1), expected);
    auto t = cast<RankedTensorType>(v.getType());
    if (!isa<stablehlo::ReshapeOp>(op) || !t.hasStaticShape() ||
        static_cast<size_t>(t.getRank()) != expected.size())
      return false;
    for (auto [d, e] : llvm::zip(t.getShape(), expected))
      if (!(e == Extent{d})) return false;
    return true;
  }
  Value identity(Value v, ArrayRef<Extent> expected) {
    auto *op = v.getDefiningOp();
    if (!op ||
        !isa<stablehlo::BroadcastInDimOp, stablehlo::DynamicBroadcastInDimOp>(
            op))
      return v;
    Value input = op->getOperand(0);
    if (static_cast<size_t>(
            cast<RankedTensorType>(input.getType()).getRank()) !=
            expected.size() ||
        !annc::proveStablehloBroadcast(v, input))
      return {};
    if (isa<stablehlo::DynamicBroadcastInDimOp>(op)) {
      if (!shape(op->getOperand(1), expected)) return {};
    } else {
      auto t = cast<RankedTensorType>(v.getType());
      for (auto [d, e] : llvm::zip(t.getShape(), expected))
        if (!(e == Extent{d})) return {};
    }
    return input;
  }

 private:
  Value data;
  llvm::DenseMap<Value, ShapeResult> cache;
  ShapeResult compute(Value v, unsigned depth) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    auto *op = v.getDefiningOp();
    if (!t || !op || !t.hasStaticShape() || t.getNumElements() > 64 ||
        !isa<IntegerType>(t.getElementType()))
      return std::nullopt;
    auto integer = cast<IntegerType>(t.getElementType());
    unsigned width = integer.getWidth();
    if (integer.isUnsigned() || width > 64) return std::nullopt;
    auto arg = [&](unsigned i) {
      return evaluate(op->getOperand(i), depth + 1);
    };
    auto name = op->getName().getStringRef();
    if (auto c = dyn_cast<stablehlo::ConstantOp>(op)) {
      auto values = dyn_cast<DenseIntElementsAttr>(c.getValue());
      if (!values) return std::nullopt;
      Extents result;
      for (APInt value : values.getValues<APInt>())
        result.push_back({width == 1 ? int64_t(value.getZExtValue())
                                     : value.getSExtValue()});
      return result;
    }
    if (auto dim = dyn_cast<stablehlo::GetDimensionSizeOp>(op)) {
      Extent value{0};
      auto input = cast<RankedTensorType>(dim.getOperand().getType());
      if (annc::stablehlo::HloGraph::identitySource(dim.getOperand()) ==
          annc::stablehlo::HloGraph::identitySource(data))
        value = dimension(dim.getDimension());
      else if (!input.isDynamicDim(dim.getDimension()))
        value = {input.getDimSize(dim.getDimension())};
      else {
        auto *source = dim.getOperand().getDefiningOp();
        ShapeResult dims;
        if (source && isa<stablehlo::DynamicReshapeOp,
                          stablehlo::DynamicBroadcastInDimOp>(source))
          dims = evaluate(source->getOperand(1), depth + 1);
        else if (auto dot = dyn_cast_or_null<stablehlo::DotOp>(source)) {
          auto side = dim.getDimension() == 0 ? dot.getLhs() : dot.getRhs();
          // Dot's dynamic row extent comes from its lhs shape program.
          auto *broadcast = side.getDefiningOp();
          if (dim.getDimension() != 0 || !broadcast ||
              !isa<stablehlo::DynamicBroadcastInDimOp>(broadcast))
            return std::nullopt;
          dims = evaluate(broadcast->getOperand(1), depth + 1);
        }
        if (!dims || dim.getDimension() >= dims->size()) return std::nullopt;
        value = (*dims)[dim.getDimension()];
      }
      return value.fits(width) ? ShapeResult(Extents{value}) : std::nullopt;
    }
    if (isa<stablehlo::ConvertOp, stablehlo::ReshapeOp,
            stablehlo::DynamicReshapeOp>(op)) {
      auto values = arg(0);
      if (!values || values->size() != static_cast<size_t>(t.getNumElements()))
        return std::nullopt;
      if (isa<stablehlo::ConvertOp>(op) &&
          (width == 1 ||
           llvm::any_of(*values, [&](Extent e) { return !e.fits(width); })))
        return std::nullopt;
      if (isa<stablehlo::DynamicReshapeOp>(op)) {
        Extents expected;
        for (int64_t d : t.getShape()) expected.push_back({d});
        if (!shape(op->getOperand(1), expected)) return std::nullopt;
      }
      return values;
    }
    if (auto cat = dyn_cast<stablehlo::ConcatenateOp>(op)) {
      if (t.getRank() != 1 || cat.getDimension() != 0) return std::nullopt;
      Extents result;
      for (Value part : cat.getOperands()) {
        auto values = evaluate(part, depth + 1);
        if (!values) return std::nullopt;
        llvm::append_range(result, *values);
      }
      return result;
    }
    if (isa<stablehlo::DynamicSliceOp>(op)) {
      auto values = arg(0), start = arg(1);
      if (!values || !start || start->size() != 1 || (*start)[0].axes ||
          op->getNumOperands() != 2 || t.getRank() != 1 ||
          static_cast<size_t>(t.getNumElements()) > values->size())
        return std::nullopt;
      int64_t begin = std::clamp<int64_t>((*start)[0].coefficient, 0,
                                          values->size() - t.getNumElements());
      return Extents(values->begin() + begin,
                     values->begin() + begin + t.getNumElements());
    }
    if (auto gather = dyn_cast<stablehlo::TorchIndexSelectOp>(op)) {
      auto values = arg(0), indices = arg(1);
      if (!values || !indices || gather.getDim() != 0 ||
          gather.getBatchDims() != 0 || t.getRank() != 1)
        return std::nullopt;
      Extents result;
      for (Extent index : *indices) {
        if (index.axes || index.coefficient < 0 ||
            static_cast<uint64_t>(index.coefficient) >= values->size())
          return std::nullopt;
        result.push_back((*values)[index.coefficient]);
      }
      return result;
    }
    if (auto select = dyn_cast<stablehlo::SelectOp>(op)) {
      auto cond = arg(0);
      if (!cond || cond->empty() ||
          llvm::any_of(*cond, [](Extent e) { return e.axes != 0; }))
        return std::nullopt;
      if (llvm::all_of(*cond, [&](Extent e) { return e == cond->front(); }))
        return arg(cond->front().coefficient ? 1 : 2);
      auto a = arg(1), b = arg(2);
      if (!a || !b || a->size() != cond->size() || b->size() != cond->size())
        return std::nullopt;
      Extents result;
      for (size_t i = 0; i < cond->size(); ++i)
        result.push_back((*cond)[i].coefficient ? (*a)[i] : (*b)[i]);
      return result;
    }
    auto multiply = [&](Extent a, Extent b) -> std::optional<Extent> {
      if ((a.axes & b.axes) || a.coefficient < 0 || b.coefficient < 0 ||
          (b.coefficient && a.coefficient > INT64_MAX / b.coefficient))
        return std::nullopt;
      unsigned axes = a.axes | b.axes;
      int64_t bound =
          axes == 3 ? INT32_MAX : (a.axes ? a.variableBound : b.variableBound);
      Extent result{a.coefficient * b.coefficient, axes, bound};
      if (!result.coefficient) result.axes = 0;
      return result.fits(width) ? std::optional<Extent>(result) : std::nullopt;
    };
    if (auto reduce = dyn_cast<stablehlo::ReduceOp>(op)) {
      if (op->getNumOperands() != 2 ||
          reduce.getDimensions() != ArrayRef<int64_t>{0} ||
          !llvm::hasSingleElement(reduce.getBody()))
        return std::nullopt;
      auto &body = reduce.getBody().front();
      if (body.getNumArguments() != 2 || body.getOperations().size() != 2)
        return std::nullopt;
      auto mul = dyn_cast<stablehlo::MulOp>(body.front());
      auto ret = dyn_cast<stablehlo::ReturnOp>(body.back());
      if (!mul || !ret || mul.getLhs() != body.getArgument(0) ||
          mul.getRhs() != body.getArgument(1) ||
          ret->getOperand(0) != mul.getResult())
        return std::nullopt;
      auto values = arg(0), init = arg(1);
      if (!values || !init || init->size() != 1) return std::nullopt;
      Extent result = init->front();
      for (Extent value : *values) {
        auto next = multiply(result, value);
        if (!next) return std::nullopt;
        result = *next;
      }
      return Extents{result};
    }
    if (op->getNumOperands() != 2) return std::nullopt;
    auto a = arg(0), b = arg(1);
    if (!a || !b || a->size() != b->size()) return std::nullopt;
    Extents result;
    for (auto [x, y] : llvm::zip(*a, *b)) {
      if (auto compare = dyn_cast<stablehlo::CompareOp>(op)) {
        auto hiX = x.maximum(), hiY = y.maximum();
        if (!hiX || !hiY ||
            compare.getCompareType() == stablehlo::ComparisonType::UNSIGNED)
          return std::nullopt;
        auto direction = compare.getComparisonDirection();
        bool equal = x == y;
        bool disjoint = *hiX < y.minimum() || *hiY < x.minimum();
        if (direction == stablehlo::ComparisonDirection::EQ &&
            (equal || disjoint))
          result.push_back({equal});
        else if (direction == stablehlo::ComparisonDirection::NE &&
                 (equal || disjoint))
          result.push_back({!equal});
        else if (direction == stablehlo::ComparisonDirection::LT &&
                 (equal || *hiX < y.minimum() || x.minimum() >= *hiY))
          result.push_back({*hiX < y.minimum()});
        else
          return std::nullopt;
      } else if (name == "stablehlo.multiply") {
        auto product = multiply(x, y);
        if (!product) return std::nullopt;
        result.push_back(*product);
      } else if (!x.axes && !y.axes) {
        // Delegate constant bit-vector arithmetic to the existing evaluator.
        auto constants = annc::evaluateStablehloShape(v);
        if (!constants) return std::nullopt;
        Extents folded;
        for (int64_t c : *constants) folded.push_back({c});
        return folded;
      } else
        return std::nullopt;
    }
    return result;
  }
};
Value identityBroadcast(Value v) {
  auto *op = v.getDefiningOp();
  if (!op ||
      !isa<stablehlo::BroadcastInDimOp, stablehlo::DynamicBroadcastInDimOp>(op))
    return v;
  auto input = op->getOperand(0);
  auto a = cast<RankedTensorType>(input.getType());
  auto z = cast<RankedTensorType>(v.getType());
  if (a.getRank() != z.getRank() || !annc::proveStablehloBroadcast(v, input))
    return {};
  if (isa<stablehlo::DynamicBroadcastInDimOp>(op)) {
    auto actual = annc::evaluateStablehloShape(op->getOperand(1));
    auto *def = input.getDefiningOp();
    std::optional<SmallVector<int64_t>> source;
    if (a.hasStaticShape())
      source = SmallVector<int64_t>(a.getShape());
    else if (def && isa<stablehlo::DynamicReshapeOp>(def))
      source = annc::evaluateStablehloShape(def->getOperand(1));
    if (!actual || !source || *actual != *source) return {};
  } else if (a.getShape() != z.getShape())
    return {};
  return input;
}
bool reshapesTo(Value v, ArrayRef<int64_t> shape) {
  auto *op = v.getDefiningOp();
  if (!op) return false;
  if (isa<stablehlo::ReshapeOp>(op))
    return llvm::equal(cast<RankedTensorType>(v.getType()).getShape(), shape);
  if (!isa<stablehlo::DynamicReshapeOp>(op)) return false;
  auto actual = annc::evaluateStablehloShape(op->getOperand(1));
  return actual && llvm::equal(*actual, shape);
}
}  // namespace
namespace {
bool proveTensordot(Value output, Value data, Value weight) {
  auto x = dyn_cast<RankedTensorType>(data.getType());
  auto w = dyn_cast<RankedTensorType>(weight.getType());
  auto y = dyn_cast<RankedTensorType>(output.getType());
  auto c = weight.getDefiningOp<stablehlo::ConstantOp>();
  if (!x || !w || !y || x.getRank() != 3 || w.getRank() != 2 ||
      y.getRank() != 3 || !w.hasStaticShape() || !x.getElementType().isF32() ||
      !w.getElementType().isF32() || !y.getElementType().isF32() || !c ||
      llvm::any_of(
          x.getShape(),
          [](int64_t d) { return !ShapedType::isDynamic(d) && d <= 0; }) ||
      llvm::any_of(w.getShape(), [](int64_t d) { return d <= 0; }) ||
      x.getDimSize(2) != w.getDimSize(0) ||
      (!x.isDynamicDim(0) && !x.isDynamicDim(1) &&
       x.getDimSize(0) > INT64_MAX / x.getDimSize(1)))
    return {};
  bool dynamic = !x.hasStaticShape();
  if (x.isDynamicDim(2) ||
      (dynamic &&
       llvm::any_of(w.getShape(), [](int64_t d) { return d > INT32_MAX; })))
    return {};
  int64_t rows =
      dynamic ? ShapedType::kDynamic : x.getDimSize(0) * x.getDimSize(1);
  int64_t k = w.getDimSize(0), n = w.getDimSize(1);
  SmallVector<int64_t> outShape{x.getDimSize(0), x.getDimSize(1), n};
  for (auto [actual, expected] : llvm::zip(y.getShape(), outShape))
    if (!ShapedType::isDynamic(actual) && actual != expected) return {};
  if (!dynamic && !reshapesTo(output, outShape)) return {};
  auto dot =
      output.getDefiningOp()->getOperand(0).getDefiningOp<stablehlo::DotOp>();
  if (!dot) return {};
  if (dot.getPrecisionConfig() &&
      llvm::any_of(*dot.getPrecisionConfig(), [](Attribute attr) {
        return cast<stablehlo::PrecisionAttr>(attr).getValue() !=
               stablehlo::Precision::DEFAULT;
      }))
    return {};
  Value flat, rhs;
  if (dynamic) {
    TensordotShapeProof proof(data);
    Extent a = proof.dimension(0), z = proof.dimension(1);
    if (a.coefficient > INT32_MAX || z.coefficient > INT32_MAX ||
        a.coefficient > INT32_MAX / z.coefficient)
      return {};
    Extent product{a.coefficient * z.coefficient, a.axes | z.axes};
    Extents matrix{product, {k}}, result{a, z, {n}};
    flat = proof.identity(dot.getLhs(), matrix);
    rhs = proof.identity(dot.getRhs(), Extents{{k}, {n}});
    if (!flat || rhs != weight || !proof.reshape(flat, matrix) ||
        !proof.reshape(output, result))
      return {};
  } else {
    flat = identityBroadcast(dot.getLhs());
    rhs = identityBroadcast(dot.getRhs());
    if (!flat || rhs != weight || !reshapesTo(flat, {rows, k})) return {};
  }
  if (flat.getDefiningOp()->getOperand(0) != data) return {};
  auto lhsType = cast<RankedTensorType>(dot.getLhs().getType());
  auto rhsType = cast<RankedTensorType>(dot.getRhs().getType());
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2) return {};

  return true;
}
}  // namespace

namespace annc::stablehlo {
void partitionTensordot(HloGraph &graph,
                        SmallVectorImpl<Attribute> &rejections) {
  auto unwrap = [](Value v) {
    auto *op = v.getDefiningOp();
    return op && isa<mlir::stablehlo::BroadcastInDimOp,
                     mlir::stablehlo::DynamicBroadcastInDimOp>(op)
               ? op->getOperand(0)
               : v;
  };
  for (const auto &[id, op] : graph.operations) {
    if (!isa<mlir::stablehlo::ReshapeOp, mlir::stablehlo::DynamicReshapeOp>(op))
      continue;
    auto dot = op->getOperand(0).getDefiningOp<mlir::stablehlo::DotOp>();
    if (!dot) continue;
    Value flat = unwrap(dot.getLhs()), weight = unwrap(dot.getRhs());
    auto *flatten = flat.getDefiningOp();
    if (!flatten ||
        !isa<mlir::stablehlo::ReshapeOp, mlir::stablehlo::DynamicReshapeOp>(
            flatten))
      continue;
    Value data = flatten->getOperand(0), output = op->getResult(0);
    if (!proveTensordot(output, data, weight)) continue;
    graph.fuse("tensordot_matmul", "TensordotMatMul", {data, weight}, {output},
               {}, rejections);
  }
}
}  // namespace annc::stablehlo
