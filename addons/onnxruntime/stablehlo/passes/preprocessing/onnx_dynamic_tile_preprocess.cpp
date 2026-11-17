#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

class OnnxDynamicTilePreprocessPass
    : public PassWrapper<OnnxDynamicTilePreprocessPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OnnxDynamicTilePreprocessPass)
  StringRef getArgument() const final {
    return "onnx-dynamic-tile-preprocess";
  }
  StringRef getDescription() const final {
    return "Mark dynamic ONNX Tile for residual StableHLO lowering";
  }
  void runOnOperation() override {
    SmallVector<Operation *> tiles;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() == "onnx.Tile") tiles.push_back(op);
    });
    IRRewriter builder(&getContext());
    for (Operation *op : tiles) {
      auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      if (!inputType) {
        op->emitError("Tile preprocessing requires a ranked input");
        signalPassFailure();
        return;
      }
      if (inputType.hasStaticShape()) continue;
      auto repeatsType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
      auto outputType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      int64_t rank = inputType.getRank();
      if (!repeatsType || repeatsType.getRank() != 1 ||
          repeatsType.getDimSize(0) != rank ||
          !repeatsType.getElementType().isInteger(64) || !outputType ||
          outputType.getRank() != rank) {
        op->emitError("Tile preprocessing requires rank-sized i64 repeats "
                       "and a ranked result of the same rank as the input");
        signalPassFailure();
        return;
      }
      builder.setInsertionPoint(op);
      OperationState marker(op->getLoc(), "annc_preprocess.dynamic_tile");
      marker.addOperands(op->getOperands());
      marker.addTypes(outputType);
      if (auto ids = op->getAttr("source_ids"))
        marker.addAttribute("source_ids", ids);
      builder.replaceOp(op, builder.create(marker)->getResults());
    }
  }
};

}  // namespace

void registerOnnxDynamicTilePreprocessPass() {
  PassRegistration<OnnxDynamicTilePreprocessPass>();
}
