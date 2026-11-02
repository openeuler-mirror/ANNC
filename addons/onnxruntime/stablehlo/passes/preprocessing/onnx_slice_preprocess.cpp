#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "src/Dialect/ONNX/ONNXDialect.hpp"

using namespace mlir;

namespace {

Operation *create(IRRewriter &builder, Operation *source, Type type,
                  ValueRange operands = {},
                  ArrayRef<NamedAttribute> attrs = {}) {
  OperationState state(source->getLoc(), "onnx.Constant");
  state.addTypes(type);
  state.addOperands(operands);
  state.addAttributes(attrs);
  if (auto ids = source->getAttr("source_ids"))
    state.addAttribute("source_ids", ids);
  return builder.create(state);
}

Value constant(IRRewriter &builder, Operation *source,
               RankedTensorType type, ArrayRef<int64_t> values) {
  return create(builder, source, type, {},
                {builder.getNamedAttr(
                    "value", DenseIntElementsAttr::get(type, values))})
      ->getResult(0);
}

Value indexI64(IRRewriter &builder, Operation *slice, Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.getElementType().isInteger(32)) return value;
  auto targetType = RankedTensorType::get(type.getShape(), builder.getI64Type());
  Operation *def = value.getDefiningOp();
  if (def && def->getName().getStringRef() == "onnx.Constant") {
    if (auto attr = def->getAttrOfType<ElementsAttr>("value")) {
      SmallVector<int64_t> values;
      for (IntegerAttr element : attr.getValues<IntegerAttr>())
        values.push_back(element.getInt());
      return constant(builder, slice, targetType, values);
    }
  }
  OperationState state(slice->getLoc(), "onnx.Cast");
  state.addOperands(value);
  state.addTypes(targetType);
  state.addAttribute("to", TypeAttr::get(builder.getI64Type()));
  if (auto ids = slice->getAttr("source_ids"))
    state.addAttribute("source_ids", ids);
  return builder.create(state)->getResult(0);
}

LogicalResult normalize(Operation *op, IRRewriter &builder) {
  builder.setInsertionPoint(op);
  for (unsigned i = 1; i < op->getNumOperands(); ++i)
    op->setOperand(i, indexI64(builder, op, op->getOperand(i)));
  auto startsType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!startsType || !startsType.hasStaticShape())
    return op->emitError("Slice preprocessing requires a known index count");
  int64_t count = startsType.getNumElements();
  auto vectorType = RankedTensorType::get({count}, builder.getI64Type());
  if (isa<NoneType>(op->getOperand(3).getType())) {
    SmallVector<int64_t> axes;
    for (int64_t i = 0; i < count; ++i) axes.push_back(i);
    op->setOperand(3, constant(builder, op, vectorType, axes));
  }
  if (isa<NoneType>(op->getOperand(4).getType())) {
    SmallVector<int64_t> steps(count, 1);
    op->setOperand(4, constant(builder, op, vectorType, steps));
  }
  auto axes = op->getOperand(3).getDefiningOp();
  if (!axes || axes->getName().getStringRef() != "onnx.Constant" ||
      !axes->getAttrOfType<ElementsAttr>("value"))
    return op->emitError("StableHLO Slice lowering requires constant axes");
  return success();
}

class OnnxSlicePreprocessPass
    : public PassWrapper<OnnxSlicePreprocessPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OnnxSlicePreprocessPass)
  StringRef getArgument() const final { return "onnx-slice-preprocess"; }
  StringRef getDescription() const final {
    return "Normalize ONNX Slice indices for StableHLO lowering";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ONNXDialect>();
  }
  void runOnOperation() override {
    SmallVector<Operation *> slices;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "onnx.Slice") slices.push_back(op);
    });
    IRRewriter builder(&getContext());
    for (Operation *op : slices)
      if (failed(normalize(op, builder))) {
        signalPassFailure();
        return;
      }
    SmallVector<Operation *> unused;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "onnx.NoValue" && op->use_empty())
        unused.push_back(op);
    });
    for (Operation *op : unused) builder.eraseOp(op);
  }
};

}  // namespace

void registerOnnxSlicePreprocessPass() {
  PassRegistration<OnnxSlicePreprocessPass>();
}
