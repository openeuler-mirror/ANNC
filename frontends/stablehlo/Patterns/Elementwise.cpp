#include "Patterns/Elementwise.h"

#include <algorithm>

#include "Patterns/ShapeProof.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
using namespace mlir;

namespace {
Value unbroadcast(Value v) {
  while (auto *op = v.getDefiningOp()) {
    auto name = op->getName().getStringRef();
    if (name != "stablehlo.broadcast_in_dim" &&
        name != "stablehlo.dynamic_broadcast_in_dim")
      break;
    auto input = op->getOperand(0);
    if (!annc::proveStablehloBroadcast(v, input)) return {};
    v = input;
  }
  return v;
}
bool splat(Value value, double expected) {
  value = unbroadcast(value);
  if (!value) return false;
  auto op = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
  auto data = op ? dyn_cast<DenseFPElementsAttr>(op.getValue()) : nullptr;
  return data && data.isSplat() &&
         data.getSplatValue<APFloat>().convertToDouble() == expected;
}
// A structural dimension proof, rather than assigning sample values to dynamic
// extents. Only shape-preserving wrappers and explicit shape-vector operations
// are traversed; arithmetic on unknown dimensions is deliberately rejected.
bool dimension(Value shape, unsigned index, Value like, unsigned axis,
               unsigned depth = 0) {
  if (depth > 64) return false;
  auto type = dyn_cast<RankedTensorType>(like.getType());
  if (!type || axis >= type.getRank()) return false;
  if (auto reduce = like.getDefiningOp<mlir::stablehlo::ReduceOp>()) {
    unsigned inputAxis = 0, kept = 0;
    while (llvm::is_contained(reduce.getDimensions(), inputAxis) ||
           kept++ < axis)
      ++inputAxis;
    return dimension(shape, index, reduce->getOperand(0), inputAxis, depth + 1);
  }
  if (!type.isDynamicDim(axis)) {
    auto values = annc::evaluateStablehloShape(shape);
    if (values && index < values->size())
      return (*values)[index] == type.getDimSize(axis);
  }
  auto *op = shape.getDefiningOp();
  if (!op) return false;
  auto recurse = [&](Value v, unsigned i) {
    return dimension(v, i, like, axis, depth + 1);
  };
  if (auto convert = dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
    auto from =
        cast<RankedTensorType>(convert.getOperand().getType()).getElementType();
    auto to = cast<RankedTensorType>(shape.getType()).getElementType();
    if (!isa<IntegerType>(from) || !isa<IntegerType>(to) ||
        from.getIntOrFloatBitWidth() > to.getIntOrFloatBitWidth())
      return false;
    return recurse(convert.getOperand(), index);
  }
  if (isa<mlir::stablehlo::ReshapeOp>(op))
    return recurse(op->getOperand(0), index);
  if (auto slice = dyn_cast<mlir::stablehlo::SliceOp>(op)) {
    auto result = dyn_cast<RankedTensorType>(shape.getType());
    if (!result || result.getRank() != 1 || !result.hasStaticShape() ||
        index >= result.getNumElements() || slice.getStartIndices().size() != 1)
      return false;
    return recurse(slice.getOperand(),
                   slice.getStartIndices()[0] + index * slice.getStrides()[0]);
  }
  if (auto mul = dyn_cast<mlir::stablehlo::MulOp>(op)) {
    // Multiplication by one preserves every integer bit pattern. Do not
    // generalize this to arithmetic on unknown extents or narrowing casts.
    for (unsigned slot = 0; slot != 2; ++slot) {
      auto ones = annc::evaluateStablehloShape(mul->getOperand(slot));
      if (ones && index < ones->size() && (*ones)[index] == 1)
        return recurse(mul->getOperand(1 - slot), index);
    }
    return false;
  }
  if (auto cat = dyn_cast<mlir::stablehlo::ConcatenateOp>(op)) {
    if (cat.getDimension() != 0) return false;
    for (Value part : cat.getOperands()) {
      auto t = dyn_cast<RankedTensorType>(part.getType());
      if (!t || !t.hasStaticShape()) return false;
      if (index < t.getNumElements()) return recurse(part, index);
      index -= t.getNumElements();
    }
  }
  if (auto slice = dyn_cast<mlir::stablehlo::DynamicSliceOp>(op)) {
    auto start = annc::evaluateStablehloShape(op->getOperand(1));
    auto source = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto result = dyn_cast<RankedTensorType>(shape.getType());
    if (op->getNumOperands() != 2 || !start || start->size() != 1 || !source ||
        !result || source.getRank() != 1 || result.getRank() != 1 ||
        !source.hasStaticShape() || !result.hasStaticShape() ||
        index >= result.getNumElements())
      return false;
    int64_t maximum = source.getNumElements() - result.getNumElements();
    if (maximum < 0) return false;
    int64_t begin = std::clamp<int64_t>((*start)[0], 0, maximum);
    return recurse(op->getOperand(0), index + begin);
  }
  if (auto dim = dyn_cast<mlir::stablehlo::GetDimensionSizeOp>(op)) {
    if (index != 0) return false;
    if (dim.getOperand() == like && dim.getDimension() == axis) return true;
    auto *source = dim.getOperand().getDefiningOp();
    if (source && isa<mlir::stablehlo::DynamicBroadcastInDimOp,
                      mlir::stablehlo::DynamicReshapeOp>(source))
      return recurse(source->getOperand(1), dim.getDimension());
  }
  return false;
}
bool reshape(Value output, Value input) {
  auto *op = output.getDefiningOp();
  if (!op || op->getOperand(0) != input) return false;
  auto dst = dyn_cast<RankedTensorType>(output.getType());
  if (!dst) return false;
  if (isa<mlir::stablehlo::ReshapeOp>(op)) return dst.hasStaticShape();
  if (!isa<mlir::stablehlo::DynamicReshapeOp>(op)) return false;
  auto shape = annc::evaluateStablehloShape(op->getOperand(1));
  if (shape && dst.hasStaticShape()) return llvm::equal(*shape, dst.getShape());
  auto src = dyn_cast<RankedTensorType>(input.getType());
  if (!src || dst.getRank() != src.getRank() + 1 ||
      dst.getDimSize(dst.getRank() - 1) != 1)
    return false;
  for (unsigned i = 0; i < src.getRank(); ++i)
    if (!dimension(op->getOperand(1), i, input, i)) return false;
  // Last dimension must be a literal one, even for a dynamic leading shape.
  Value target = op->getOperand(1);
  auto cat = target.getDefiningOp<mlir::stablehlo::ConcatenateOp>();
  if (!cat) return false;
  auto last = annc::evaluateStablehloShape(cat.getOperands().back());
  return last && *last == SmallVector<int64_t>{1};
}
bool sumReduction(mlir::stablehlo::ReduceOp reduce) {
  if (!reduce || reduce->getNumOperands() != 2 ||
      !splat(reduce->getOperand(1), 0) ||
      reduce.getDimensions() != ArrayRef<int64_t>{1} ||
      !llvm::hasSingleElement(reduce.getBody()))
    return false;
  auto &block = reduce.getBody().front();
  if (block.getNumArguments() != 2 || block.getOperations().size() != 2)
    return false;
  auto add = dyn_cast<mlir::stablehlo::AddOp>(block.front());
  auto ret = dyn_cast<mlir::stablehlo::ReturnOp>(block.back());
  return add && ret && add.getLhs() == block.getArgument(0) &&
         add.getRhs() == block.getArgument(1) &&
         ret->getOperand(0) == add.getResult();
}
bool meanDivisor(Value value, Value input) {
  auto type = cast<RankedTensorType>(input.getType());
  if (splat(value, type.getDimSize(1))) return true;
  auto *reshapeOp = value.getDefiningOp();
  if (!reshapeOp ||
      !isa<mlir::stablehlo::ReshapeOp, mlir::stablehlo::DynamicReshapeOp>(
          reshapeOp) ||
      !reshape(value, reshapeOp->getOperand(0)))
    return false;
  auto count =
      reshapeOp->getOperand(0).getDefiningOp<mlir::stablehlo::ReduceOp>();
  if (!sumReduction(count)) return false;
  auto ones = count->getOperand(0)
                  .getDefiningOp<mlir::stablehlo::DynamicBroadcastInDimOp>();
  if (!ones || !splat(ones.getOperand(), 1) ||
      !ones.getBroadcastDimensions().empty())
    return false;
  auto onesType = cast<RankedTensorType>(ones.getType());
  if (onesType.getRank() != 2 || onesType.getDimSize(1) != type.getDimSize(1))
    return false;
  return dimension(ones.getOutputDimensions(), 0, input, 0) &&
         dimension(ones.getOutputDimensions(), 1, input, 1);
}
}  // namespace

namespace annc::stablehlo {
std::optional<HloElementwise> analyzeElementwise(Value output) {
  auto *op = output.getDefiningOp();
  auto type = dyn_cast<RankedTensorType>(output.getType());
  if (!op || op->getNumResults() != 1 || !type ||
      (!isa<FloatType>(type.getElementType()) &&
       !type.getElementType().isInteger(1)))
    return std::nullopt;
  HloElementwise fragment;
  auto &kind = fragment.kind;
  if (isa<mlir::stablehlo::AddOp>(op))
    kind = "Add";
  else if (isa<mlir::stablehlo::SubtractOp>(op))
    kind = "Sub";
  else if (isa<mlir::stablehlo::MulOp>(op))
    kind = "Mul";
  else if (isa<mlir::stablehlo::DivOp>(op)) {
    auto *reshapeOp = op->getOperand(0).getDefiningOp();
    auto reduce = reshapeOp && isa<mlir::stablehlo::ReshapeOp,
                                   mlir::stablehlo::DynamicReshapeOp>(reshapeOp)
                      ? reshapeOp->getOperand(0)
                            .getDefiningOp<mlir::stablehlo::ReduceOp>()
                      : nullptr;
    if (sumReduction(reduce) &&
        reshape(op->getOperand(0), reduce->getResult(0)) &&
        cast<RankedTensorType>(reduce->getOperand(0).getType()).getRank() ==
            2 &&
        !cast<RankedTensorType>(reduce->getOperand(0).getType())
             .isDynamicDim(1) &&
        type.getRank() == 2 && type.getDimSize(1) == 1 &&
        meanDivisor(op->getOperand(1), reduce->getOperand(0))) {
      fragment.kind = "ReduceMean";
      fragment.inputs.push_back(reduce->getOperand(0));
      return fragment;
    }
    if (splat(op->getOperand(0), 1)) {
      fragment.kind = "Reciprocal";
      fragment.inputs.push_back(unbroadcast(op->getOperand(1)));
      return fragment;
    }
    kind = "Div";
  } else if (isa<mlir::stablehlo::SqrtOp>(op))
    kind = "Sqrt";
  else if (isa<mlir::stablehlo::ConvertOp>(op))
    kind = "Cast";
  else if (auto cmp = dyn_cast<mlir::stablehlo::CompareOp>(op)) {
    if (cmp.getComparisonDirection() !=
        mlir::stablehlo::ComparisonDirection::EQ)
      return std::nullopt;
    kind = "Equal";
  } else if (isa<mlir::stablehlo::NotOp>(op) &&
             type.getElementType().isInteger(1))
    kind = "Not";
  else if (isa<mlir::stablehlo::ReshapeOp, mlir::stablehlo::DynamicReshapeOp>(
               op)) {
    auto broad = op->getOperand(0)
                     .getDefiningOp<mlir::stablehlo::DynamicBroadcastInDimOp>();
    if (isa<mlir::stablehlo::DynamicReshapeOp>(op) && broad &&
        broad.getBroadcastDimensions() == ArrayRef<int64_t>{1, 3, 5}) {
      auto src = dyn_cast<RankedTensorType>(broad.getOperand().getType());
      if (!src || src.getRank() != 3 || src.getDimSize(2) != 1 ||
          type.getRank() != 3 || type.isDynamicDim(2))
        return std::nullopt;
      int64_t width = type.getDimSize(2);
      // Tile expands [B,S,1] to [1,B,1,S,W,1], then collapses to
      // [B,S,W]. Prove both shape programs against the same input;
      // equal '?' dimensions alone are not evidence of equality.
      if (width < 1 || type.getDimSize(0) != src.getDimSize(0) ||
          type.getDimSize(1) != src.getDimSize(1))
        return std::nullopt;
      auto literal = [&](Value shape, unsigned index, int64_t expected) {
        // dimension() reads type extents without requiring a defining op.
        Value witness = broad.getOperand();
        if (expected == 1) return dimension(shape, index, witness, 2);
        // W is a static dimension of the final Tile result.
        return dimension(shape, index, output, 2);
      };
      Value expanded = broad.getOutputDimensions(), target = op->getOperand(1);
      if (!literal(expanded, 0, 1) ||
          !dimension(expanded, 1, broad.getOperand(), 0) ||
          !literal(expanded, 2, 1) ||
          !dimension(expanded, 3, broad.getOperand(), 1) ||
          !literal(expanded, 4, width) || !literal(expanded, 5, 1) ||
          !dimension(target, 0, broad.getOperand(), 0) ||
          !dimension(target, 1, broad.getOperand(), 1) ||
          !literal(target, 2, width))
        return std::nullopt;
      fragment.kind = "Tile";
      fragment.inputs = {broad.getOperand(), Value()};
      return fragment;
    }
    auto src = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!src || type.getRank() != src.getRank() + 1 ||
        type.getDimSize(type.getRank() - 1) != 1 ||
        !llvm::equal(src.getShape(), type.getShape().drop_back()) ||
        !reshape(output, op->getOperand(0)))
      return std::nullopt;
    fragment.kind = "Unsqueeze";
    fragment.inputs.push_back(op->getOperand(0));
    return fragment;
  } else
    return std::nullopt;
  for (Value input : op->getOperands()) {
    Value leaf = (kind == "Sqrt" || kind == "Cast" || kind == "Not")
                     ? input
                     : unbroadcast(input);
    if (!leaf) return std::nullopt;
    fragment.inputs.push_back(leaf);
  }
  // The MaskedMul contract accepts variadic Sum. Collapse only its existing
  // left spine, with a single data use at every absorbed result.
  if (kind == "Add" && type.getRank() == 3) {
    Value left = fragment.inputs[0];
    SmallVector<Value> tail{fragment.inputs[1]};
    while (auto add = left.getDefiningOp<mlir::stablehlo::AddOp>()) {
      // Producer shape programs also read this result. They do not observe
      // its data and must not prevent preserving the existing add spine.
      if (llvm::count_if(left.getUses(), [](OpOperand &use) {
            return !isa<mlir::stablehlo::GetDimensionSizeOp>(use.getOwner());
          }) != 1)
        break;
      Value lhs = unbroadcast(add.getLhs()), rhs = unbroadcast(add.getRhs());
      if (!lhs || !rhs) break;
      fragment.absorbed.push_back(add);
      tail.push_back(rhs);
      left = lhs;
    }
    if (!fragment.absorbed.empty()) {
      fragment.kind = "Sum";
      fragment.inputs.assign(1, left);
      llvm::append_range(fragment.inputs, llvm::reverse(tail));
    }
  }
  return fragment;
}

}  // namespace annc::stablehlo
