#ifndef ANNC_CUSTOMFUSIONPATTERNBASE_H
#define ANNC_CUSTOMFUSIONPATTERNBASE_H
#include "Dialect/Atir//AtirOps.h"
#include "Dialect/Atir/CustomOpSchema.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Dialect/Atir/Passes/Patterns/FusionBoundaryUtils.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistry.h"
#include "Kernel/KernelPriorityResolver.h"
#include "Support/Log.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace atir {

// Base class for fusion patterns that lower a matched subgraph into a single
// atir::CustomizeOp.  Subclasses implement matchFusion / getCustomOpName /
// getCustomOpSchema.
//
// ATIR ops use a hybrid buffer-style ABI: the first operand is an output
// buffer (typically defined by an atir::BufferOp placeholder) that the op
// writes to in place, and the op also yields an SSA result aliasing the same
// tensor.  When fusing such ops into a CustomizeOp, the base class:
//
//   1. Collects boundary outputs: SSA results of fusedOps that have at least
//      one external user.  An op is an "output op" if any of its results is
//      collected this way.
//   2. Collects input operands in two groups, both deduplicated via SetVector:
//        a) Output buffers of output ops (the first operand of each output
//           op when it is a BufferOp result), ordered by the output op's
//           position in fusedOps.  These become the CustomizeOp's output
//           memref arguments.
//        b) Real boundary inputs: operands whose defining op is neither in
//           fusedOps nor a BufferOp, in fusedOps/operand-index order.
//      The final inputValues is [output buffers..., real inputs...]; the
//      schema memref args MUST be listed in this order — the downstream
//      lowering consumes CustomizeOp operands positionally against the schema.
//   3. Erases fusedOps in reverse block order, plus any BufferOp whose result
//      is the output-buffer operand of a non-output fused op (intermediate
//      buffer placeholders that become dead after fusion).  BufferOps tied to
//      output ops are kept, since their results remain live as CustomizeOp
//      operands.
template <typename AnchorOp>
struct CustomFusionPatternBase : public mlir::OpRewritePattern<AnchorOp> {
  CustomFusionPatternBase(MLIRContext *context, PatternBenefit benefit = 9)
      : OpRewritePattern<AnchorOp>(context, benefit) {}

  CustomFusionPatternBase(MLIRContext *context,
                          const CustomOpTypeFilter &customOpFilter,
                          PatternBenefit benefit = 9)
      : OpRewritePattern<AnchorOp>(context, benefit),
        customOpFilter(customOpFilter) {}

 public:
  virtual mlir::LogicalResult matchFusion(
      AnchorOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const = 0;

  virtual std::string getCustomOpName(
      AnchorOp anchor, llvm::ArrayRef<mlir::Operation *> fusedOps) const = 0;

  virtual CustomOpSchema getCustomOpSchema(
      AnchorOp anchor, llvm::ArrayRef<mlir::Operation *> fusedOps) const = 0;

  virtual void getOrderedBoundaryInputs(
      AnchorOp anchor, llvm::ArrayRef<mlir::Operation *> fusedOps,
      llvm::SmallVectorImpl<mlir::Value> &inputs) const {
    for (mlir::Value input : collectBoundaryInputs(fusedOps)) {
      mlir::Operation *defOp = input.getDefiningOp();
      if (defOp && mlir::isa<BufferOp, ConstantOp>(defOp)) continue;
      inputs.push_back(input);
    }
  }

 public:
  mlir::LogicalResult matchAndRewrite(
      AnchorOp anchor, mlir::PatternRewriter &rewriter) const override {
    SmallVector<Operation *> fusedOps;
    auto matchres = matchFusion(anchor, fusedOps);
    if (failed(matchres)) {
      return failure();
    }

    auto customOpName = getCustomOpName(anchor, fusedOps);
    if (!customOpFilter.isEnabled(customOpName)) {
      ANNC_LOG_WARN("fusion-pattern")
          << "Custom op type '" << customOpName
          << "' filtered by FastCodegen type policy, skipping rewrite\n";
      return failure();
    }

    auto schema = getCustomOpSchema(anchor, fusedOps);
    bool usesOutputBuffers = schema.results().empty();

    llvm::SmallDenseSet<Operation *> fusedSet(fusedOps.begin(), fusedOps.end());

    // Step 1: identify escaping SSA results and v1 output-buffer owners.
    //
    // Escaping results are collected independently of internal users.  For
    // the legacy buffer ABI, a leaf op's first operand is also retained as
    // the output buffer.  Result-based v2 schemas omit these buffer operands.
    SmallVector<Value> outputValues = collectEscapingResults(fusedOps);
    llvm::SmallDenseSet<Operation *> outputOps;
    for (Operation *op : fusedOps) {
      bool hasInternalUser = false;
      for (const auto &res : op->getResults()) {
        for (Operation *useOp : res.getUsers()) {
          if (fusedSet.contains(useOp)) {
            hasInternalUser = true;
            break;
          }
        }
        if (hasInternalUser) break;
      }
      if (!hasInternalUser) {
        outputOps.insert(op);
      }
    }

    // Step 2a: for v1 schemas, collect output buffers in fused-op order.
    // BufferOp definitions stay live because CustomizeOp consumes them.
    llvm::SetVector<mlir::Value> inputSet;
    SmallVector<Operation *> keptBufferOps;
    if (usesOutputBuffers) {
      for (Operation *op : fusedOps) {
        if (!outputOps.contains(op)) continue;
        if (op->getNumOperands() == 0) continue;
        Value outOperand = op->getOperand(0);
        inputSet.insert(outOperand);
        Operation *defOp = outOperand.getDefiningOp();
        if (defOp && isa<BufferOp>(defOp)) {
          keptBufferOps.push_back(defOp);
        }
      }
    }

    // Step 2b: collect real boundary inputs through the ordering hook.  The
    // default uses first occurrence and excludes BufferOp/ConstantOp values;
    // patterns with a stricter ABI order can override it.
    SmallVector<Value> orderedBoundaryInputs;
    getOrderedBoundaryInputs(anchor, fusedOps, orderedBoundaryInputs);
    for (Value input : orderedBoundaryInputs) inputSet.insert(input);
    SmallVector<Value> inputValues(inputSet.begin(), inputSet.end());

    SmallVector<Type> resultTypes;
    resultTypes.reserve(outputValues.size());
    for (const auto &value : outputValues) {
      resultTypes.push_back(value.getType());
    }

    auto metadata = schema.toMetadata(rewriter.getContext());

    auto module = anchor->template getParentOfType<mlir::ModuleOp>();
    auto attr =
        module->template getAttrOfType<mlir::BoolAttr>("annc.enable_kdnn");
    bool enableKdnn = attr && attr.getValue();
    annc::kernels::KernelResolveRequest req;
    req.op_type = customOpName;
    if (auto parentFunc =
            anchor->template getParentOfType<mlir::func::FuncOp>()) {
      if (auto fusionMetadata =
              parentFunc->template getAttrOfType<mlir::DictionaryAttr>(
                  "fusion.metadata")) {
        if (auto abi = mlir::dyn_cast_or_null<mlir::StringAttr>(
                fusionMetadata.get("abi"))) {
          req.abi = abi.getValue().str();
        }
      }
    }
    SmallVector<Type> inputTypes;
    inputTypes.reserve(inputValues.size());
    for (Value value : inputValues) inputTypes.push_back(value.getType());
    TypeRange schemaResultTypes =
        schema.results().empty() ? TypeRange{} : TypeRange(resultTypes);
    auto inferredTypes = inferTypeConstraintsFromSchema(
        metadata, TypeRange(inputTypes), schemaResultTypes);
    if (!inferredTypes) {
      llvm::consumeError(inferredTypes.takeError());
      ANNC_LOG_WARN("fusion-pattern")
          << "Invalid type bindings for '" << customOpName
          << "', skipping CustomizeOp rewrite\n";
      return failure();
    }
    req.type_constraints = std::move(*inferredTypes);
    if (auto rhsFormat =
            anchor->template getAttrOfType<mlir::StringAttr>("rhs_format")) {
      req.rhs_format = rhsFormat.getValue().str();
    }
    if (!annc::kernels::hasAnyAvailableKernel(req, enableKdnn)) {
      ANNC_LOG_WARN("fusion-pattern")
          << "No builtin kernel available for '" << customOpName
          << "', skipping CustomizeOp rewrite\n";
      return failure();
    }

    auto callee = StringAttr::get(rewriter.getContext(), customOpName);

    // Create the CustomizeOp after all its inputs are defined to preserve
    // dominance.  Inputs may include ops defined after the anchor (e.g.
    // constants feeding ops later in the block); placing the custom op at
    // the anchor would violate SSA dominance.  Insert after the latest
    // defining op among inputValues.
    Operation *latestDef = nullptr;
    for (Value v : inputValues) {
      if (auto *defOp = v.getDefiningOp()) {
        if (!latestDef || latestDef->isBeforeInBlock(defOp)) {
          latestDef = defOp;
        }
      }
    }
    if (latestDef) {
      rewriter.setInsertionPointAfter(latestDef);
    } else {
      rewriter.setInsertionPoint(anchor);
    }

    auto customCallOp = rewriter.create<CustomizeOp>(
        anchor.getLoc(), resultTypes, inputValues, callee, metadata);
    if (auto rhsFormat =
            anchor->template getAttrOfType<mlir::StringAttr>("rhs_format")) {
      customCallOp->setAttr("rhs_format", rhsFormat);
    }

    // Replace escaping SSA results with the CustomizeOp's results.
    for (auto [oldV, newV] :
         llvm::zip(outputValues, customCallOp.getResults())) {
      rewriter.replaceAllUsesWith(oldV, newV);
    }

    // Step 3: erase fusedOps plus intermediate BufferOps, in reverse block
    // order to respect def-use.  Kept BufferOps (tied to output ops) are
    // excluded since their results remain live as CustomizeOp operands.
    // Use SetVector to deduplicate — a BufferOp may already be in fusedOps
    // (collected by the pattern) and also be re-added here as an intermediate
    // buffer of a non-output fused op; erasing twice would be UB.
    llvm::SetVector<Operation *> allEraseSet(fusedOps.begin(), fusedOps.end());
    for (Operation *op : fusedOps) {
      if (usesOutputBuffers && outputOps.contains(op)) continue;
      if (op->getNumOperands() == 0) continue;
      Operation *defOp = op->getOperand(0).getDefiningOp();
      if (defOp && isa<BufferOp>(defOp) &&
          !llvm::is_contained(keptBufferOps, defOp)) {
        allEraseSet.insert(defOp);
      }
    }
    SmallVector<Operation *, 64> allEraseOps(allEraseSet.begin(),
                                             allEraseSet.end());
    llvm::sort(allEraseOps, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    for (Operation *op : llvm::reverse(allEraseOps)) {
      rewriter.eraseOp(op);
    }

    return success();
  }

 private:
  CustomOpTypeFilter customOpFilter;
};

}  // namespace atir
#endif  // ANNC_CUSTOMFUSIONPATTERNBASE_H
