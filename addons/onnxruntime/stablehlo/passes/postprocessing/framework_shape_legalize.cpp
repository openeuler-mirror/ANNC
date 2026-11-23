#include "Dialect/Atir/AtirOps.h"
// Legalize the shape/index plumbing emitted by upstream ONNX-MLIR without
// lowering to buffers or depending on the ANNC compiler.
#include <algorithm>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace {
class ShapeLegalizer {
 public:
  explicit ShapeLegalizer(MLIRContext *context) : builder(context) {}

  Value make(StringRef name, Type type, ValueRange values = {},
             ArrayRef<NamedAttribute> attrs = {}) {
    OperationState state(source->getLoc(), name);
    state.addOperands(values);
    state.addTypes(type);
    state.addAttributes(attrs);
    if (auto ids = source->getAttr("source_ids"))
      state.addAttribute("source_ids", ids);
    return builder.create(state)->getResult(0);
  }
  RankedTensorType scalarType() {
    return RankedTensorType::get({}, builder.getI64Type());
  }
  Value integer(int64_t value) {
    return make("stablehlo.constant", scalarType(), {},
                {builder.getNamedAttr(
                    "value", DenseIntElementsAttr::get(scalarType(), value))});
  }
  Value binary(StringRef name, Value lhs, Value rhs) {
    return make(name, lhs.getType(), {lhs, rhs});
  }
  Value compare(Value lhs, Value rhs, stablehlo::ComparisonDirection direction,
                bool unsignedCompare = false) {
    return make(
        "stablehlo.compare", RankedTensorType::get({}, builder.getI1Type()),
        {lhs, rhs},
        {builder.getNamedAttr("comparison_direction",
                              stablehlo::ComparisonDirectionAttr::get(
                                  builder.getContext(), direction)),
         builder.getNamedAttr(
             "compare_type",
             stablehlo::ComparisonTypeAttr::get(
                 builder.getContext(),
                 unsignedCompare ? stablehlo::ComparisonType::UNSIGNED
                                 : stablehlo::ComparisonType::SIGNED))});
  }
  Value select(Value condition, Value lhs, Value rhs) {
    return make("stablehlo.select", lhs.getType(), {condition, lhs, rhs});
  }
  Value quotient(Value lhs, Value rhs, bool ceil = false) {
    using D = stablehlo::ComparisonDirection;
    Value zero = integer(0), one = integer(1);
    Value q = binary("stablehlo.divide", lhs, rhs);
    Value r = binary("stablehlo.remainder", lhs, rhs);
    Value lhsNegative = compare(lhs, zero, D::LT);
    Value rhsNegative = compare(rhs, zero, D::LT);
    Value signsDiffer = binary("stablehlo.xor", lhsNegative, rhsNegative);
    Value needsAdjustment =
        binary("stablehlo.and", compare(r, zero, D::NE),
               ceil ? make("stablehlo.not", signsDiffer.getType(), signsDiffer)
                    : signsDiffer);
    return binary(ceil ? "stablehlo.add" : "stablehlo.subtract", q,
                  select(needsAdjustment, one, zero));
  }
  Value reshape(Value value, ArrayRef<int64_t> shape) {
    auto type = cast<RankedTensorType>(value.getType());
    return make("stablehlo.reshape",
                RankedTensorType::get(shape, type.getElementType()), value);
  }
  Value vector(ValueRange values) {
    auto type = RankedTensorType::get({static_cast<int64_t>(values.size())},
                                      builder.getI64Type());
    if (values.empty())
      return make(
          "stablehlo.constant", type, {},
          {builder.getNamedAttr(
              "value", DenseIntElementsAttr::get(type, ArrayRef<int64_t>{}))});
    SmallVector<Value> parts;
    for (Value value : values) parts.push_back(reshape(value, {1}));
    return make(
        "stablehlo.concatenate", type, parts,
        {builder.getNamedAttr("dimension", builder.getI64IntegerAttr(0))});
  }
  Value extent(Value shape, Value dimension) {
    auto type = RankedTensorType::get({1}, builder.getI64Type());
    Value slice = make("stablehlo.dynamic_slice", type, {shape, dimension},
                       {builder.getNamedAttr(
                           "slice_sizes", builder.getDenseI64ArrayAttr({1}))});
    return reshape(slice, {});
  }
  Value shapeOf(Value tensor) {
    auto type = dyn_cast<RankedTensorType>(tensor.getType());
    if (!type) return {};
    SmallVector<Value> extents;
    for (int64_t i = 0; i < type.getRank(); ++i) {
      auto i32 = RankedTensorType::get({}, builder.getI32Type());
      Value dimension = make(
          "stablehlo.get_dimension_size", i32, tensor,
          {builder.getNamedAttr("dimension", builder.getI64IntegerAttr(i))});
      extents.push_back(make("stablehlo.convert", scalarType(), dimension));
    }
    return vector(extents);
  }
  Value affineExpr(AffineExpr expr, ValueRange operands, unsigned dims) {
    if (auto constant = dyn_cast<AffineConstantExpr>(expr))
      return integer(constant.getValue());
    if (auto dim = dyn_cast<AffineDimExpr>(expr))
      return operands[dim.getPosition()];
    if (auto symbol = dyn_cast<AffineSymbolExpr>(expr))
      return operands[dims + symbol.getPosition()];
    auto binaryExpr = dyn_cast<AffineBinaryOpExpr>(expr);
    if (!binaryExpr) return {};
    Value lhs = affineExpr(binaryExpr.getLHS(), operands, dims);
    Value rhs = affineExpr(binaryExpr.getRHS(), operands, dims);
    if (!lhs || !rhs) return {};
    switch (expr.getKind()) {
      case AffineExprKind::Add:
        return binary("stablehlo.add", lhs, rhs);
      case AffineExprKind::Mul:
        return binary("stablehlo.multiply", lhs, rhs);
      case AffineExprKind::FloorDiv:
        return quotient(lhs, rhs);
      case AffineExprKind::CeilDiv:
        return quotient(lhs, rhs, true);
      case AffineExprKind::Mod:
        return binary("stablehlo.subtract", lhs,
                      binary("stablehlo.multiply", quotient(lhs, rhs), rhs));
      default:
        return {};
    }
  }
  Type converted(Type type, Operation *op) {
    if (type.isIndex()) return scalarType();
    if (isa<IntegerType, FloatType>(type))
      return RankedTensorType::get({}, type);
    if (auto ranked = dyn_cast<RankedTensorType>(type);
        ranked && ranked.getElementType().isIndex())
      return RankedTensorType::get(ranked.getShape(), builder.getI64Type());
    if (isa<shape::ShapeType>(type)) {
      if (op->getName().getStringRef() == "shape.from_extents")
        return RankedTensorType::get(
            {static_cast<int64_t>(op->getNumOperands())}, builder.getI64Type());
      if (op->getName().getStringRef() == "shape.broadcast") {
        int64_t rank = 0;
        for (Value operand : op->getOperands()) {
          auto shape = dyn_cast<RankedTensorType>(operand.getType());
          if (!shape || !shape.hasStaticShape() ||
              shape.getElementType() != builder.getI64Type())
            return type;
          rank = std::max(rank, shape.getNumElements());
        }
        return RankedTensorType::get({rank}, builder.getI64Type());
      }
    }
    return type;
  }
  LogicalResult run(ModuleOp module) {
    SmallVector<Operation *> ops;
    module.walk<WalkOrder::PreOrder>([&](Operation *op) { ops.push_back(op); });
    for (Operation *op : ops) {
      for (OpResult result : op->getResults())
        result.setType(converted(result.getType(), op));
    }
    for (Operation *op : ops) {
      StringRef ns = op->getName().getDialectNamespace();
      if (ns == "builtin" || ns == "func" || ns == "stablehlo" ||
          isa<atir::OpaqueOp>(op))
        continue;
      source = op;
      builder.setInsertionPoint(op);
      StringRef name = op->getName().getStringRef();
      Value result;
      if (name == "arith.constant") {
        auto value = op->getAttrOfType<TypedAttr>("value");
        auto target = dyn_cast<RankedTensorType>(op->getResult(0).getType());
        Attribute dense;
        if (auto attr = dyn_cast<IntegerAttr>(value))
          dense = DenseIntElementsAttr::get(
              target,
              attr.getValue().sextOrTrunc(target.getElementTypeBitWidth()));
        else if (auto attr = dyn_cast<FloatAttr>(value))
          dense = DenseFPElementsAttr::get(target, attr.getValue());
        else if (auto attr = dyn_cast<DenseIntElementsAttr>(value)) {
          SmallVector<APInt> values;
          for (auto v : attr.getValues<APInt>())
            values.push_back(v.sextOrTrunc(target.getElementTypeBitWidth()));
          dense = DenseIntElementsAttr::get(target, values);
        }
        if (dense)
          result = make("stablehlo.constant", target, {},
                        {builder.getNamedAttr("value", dense)});
      } else if (name == "shape.shape_of") {
        result = shapeOf(op->getOperand(0));
      } else if (name == "shape.get_extent") {
        result = extent(op->getOperand(0), op->getOperand(1));
      } else if (name == "tensor.dim") {
        Value shape = shapeOf(op->getOperand(0));
        if (shape) result = extent(shape, op->getOperand(1));
      } else if (name == "shape.from_extents" ||
                 name == "tensor.from_elements") {
        result = vector(op->getOperands());
      } else if (name == "shape.to_extent_tensor" ||
                 name == "arith.index_cast") {
        Value value = op->getOperand(0);
        result =
            value.getType() == op->getResult(0).getType()
                ? value
                : make("stablehlo.convert", op->getResult(0).getType(), value);
      } else if (name == "shape.broadcast") {
        auto target = dyn_cast<RankedTensorType>(op->getResult(0).getType());
        if (target && target.hasStaticShape()) {
          SmallVector<Value> values;
          int64_t rank = target.getNumElements();
          for (int64_t i = 0; i < rank; ++i) {
            Value current = integer(1);
            for (Value shape : op->getOperands()) {
              auto shapeType = dyn_cast<RankedTensorType>(shape.getType());
              if (!shapeType || !shapeType.hasStaticShape())
                return op->emitError("broadcast shape rank must be known");
              int64_t offset = rank - shapeType.getNumElements();
              if (i < offset) continue;
              Value next = extent(shape, integer(i - offset));
              current = select(compare(current, integer(1),
                                       stablehlo::ComparisonDirection::EQ),
                               next, current);
            }
            values.push_back(current);
          }
          result = vector(values);
        }
      } else if (name == "affine.apply" || name == "affine.max" ||
                 name == "affine.min") {
        if (auto mapAttr = op->getAttrOfType<AffineMapAttr>("map")) {
          AffineMap map = mapAttr.getValue();
          for (AffineExpr expr : map.getResults()) {
            Value value = affineExpr(expr, op->getOperands(), map.getNumDims());
            if (!value)
              return op->emitError(
                  "unsupported affine expression at StableHLO boundary");
            result = !result
                         ? value
                         : binary(name == "affine.min" ? "stablehlo.minimum"
                                                       : "stablehlo.maximum",
                                  result, value);
          }
        }
      } else if (name == "arith.cmpi") {
        using P = arith::CmpIPredicate;
        using D = stablehlo::ComparisonDirection;
        auto predicate = static_cast<P>(
            cast<IntegerAttr>(op->getAttr("predicate")).getInt());
        D direction = D::EQ;
        switch (predicate) {
          case P::eq:
            direction = D::EQ;
            break;
          case P::ne:
            direction = D::NE;
            break;
          case P::slt:
          case P::ult:
            direction = D::LT;
            break;
          case P::sle:
          case P::ule:
            direction = D::LE;
            break;
          case P::sgt:
          case P::ugt:
            direction = D::GT;
            break;
          case P::sge:
          case P::uge:
            direction = D::GE;
            break;
        }
        result = compare(op->getOperand(0), op->getOperand(1), direction,
                         predicate >= P::ult);
      } else if (name == "arith.select") {
        result =
            select(op->getOperand(0), op->getOperand(1), op->getOperand(2));
      } else if (name == "arith.floordivsi") {
        result = quotient(op->getOperand(0), op->getOperand(1));
      } else if (name == "arith.muli" || name == "arith.addi" ||
                 name == "arith.subi") {
        result = binary(name == "arith.muli"   ? "stablehlo.multiply"
                        : name == "arith.addi" ? "stablehlo.add"
                                               : "stablehlo.subtract",
                        op->getOperand(0), op->getOperand(1));
      }
      if (!result || op->getNumResults() != 1)
        return op->emitError(
            "operation cannot be expressed by framework StableHLO shape "
            "legalization");
      builder.replaceOp(op, result);
    }
    return success();
  }

 private:
  IRRewriter builder;
  Operation *source = nullptr;
};

class FrameworkShapePass
    : public PassWrapper<FrameworkShapePass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FrameworkShapePass)
  StringRef getArgument() const final { return "framework-shape-to-stablehlo"; }
  StringRef getDescription() const final {
    return "Legalize ONNX-MLIR shape plumbing to pure StableHLO";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<stablehlo::StablehloDialect>();
  }
  void runOnOperation() override {
    if (failed(ShapeLegalizer(&getContext()).run(getOperation())))
      signalPassFailure();
  }
};
}  // namespace

void registerFrameworkShapePass() { PassRegistration<FrameworkShapePass>(); }
