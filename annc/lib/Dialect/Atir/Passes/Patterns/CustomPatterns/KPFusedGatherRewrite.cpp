#include "Dialect/Atir/Passes/Patterns/CustomFusionPatternBase.h"
#include "Dialect/Atir/Passes/Patterns/CustomPatterns/KPFusedGatherMatch.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistryMacros.h"

using namespace mlir;
using namespace atir;

namespace {

static bool definesAny(Operation *op, llvm::ArrayRef<Value> values) {
  if (!op) return false;
  for (Value result : op->getResults())
    if (llvm::is_contained(values, result)) return true;
  return false;
}

static void collectDefiningOpsPostOrder(
    Operation *op, llvm::ArrayRef<Value> boundaryValues,
    llvm::SmallPtrSetImpl<Operation *> &visited,
    llvm::SmallVectorImpl<Operation *> &ops) {
  if (!op || visited.contains(op) || isa<VariableOp>(op)) return;
  if (definesAny(op, boundaryValues)) return;
  visited.insert(op);
  for (Value operand : op->getOperands()) {
    if (llvm::is_contained(boundaryValues, operand)) continue;
    collectDefiningOpsPostOrder(operand.getDefiningOp(), boundaryValues,
                                visited, ops);
  }
  ops.push_back(op);
}

struct KPFusedGatherRewrite : public CustomFusionPatternBase<GatherOp> {
  KPFusedGatherRewrite(MLIRContext *context,
                       const CustomOpTypeFilter &customOpFilter,
                       PatternBenefit benefit = 9)
      : CustomFusionPatternBase<GatherOp>(context, customOpFilter, benefit) {}

  LogicalResult matchFusion(
      GatherOp anchor,
      llvm::SmallVectorImpl<Operation *> &fusedOps) const override {
    auto match = matchKPFusedGather(anchor);
    if (failed(match)) return failure();
    SmallVector<Value, 3> boundaryInputs = {match->data, match->keys,
                                            match->begin};
    SmallPtrSet<Operation *, 32> visited;
    for (Value output : match->boundaryOutputs) {
      collectDefiningOpsPostOrder(output.getDefiningOp(), boundaryInputs,
                                  visited, fusedOps);
    }
    return fusedOps.empty() ? failure() : success();
  }

  std::string getCustomOpName(GatherOp,
                              llvm::ArrayRef<Operation *>) const override {
    return "KPFusedGather";
  }

  CustomOpSchema getCustomOpSchema(GatherOp,
                                   llvm::ArrayRef<Operation *>) const override {
    return CustomOpSchema::get("KPFusedGather")
        .TypeVar("T")
        .TypeVar("Tkeys")
        .TypeVar("Tbegin")
        .TypeVar("Tindices")
        .MemRefArg("data", 2, "T")
        .MemRefArg("keys", 2, "Tkeys")
        .MemRefArg("begin", 1, "Tbegin")
        .Result("unique_values", 1, "Tkeys")
        .Result("unique_indices", 1, "Tindices")
        .Result("gathered", 2, "T");
  }

  void getOrderedBoundaryInputs(
      GatherOp anchor, llvm::ArrayRef<Operation *>,
      llvm::SmallVectorImpl<Value> &inputs) const override {
    auto match = matchKPFusedGather(anchor);
    if (succeeded(match)) {
      inputs.push_back(match->data);
      inputs.push_back(match->keys);
      inputs.push_back(match->begin);
    }
  }
};

REGISTER_CUSTOM_PATTERN(KPFusedGatherRewrite);

}  // namespace
