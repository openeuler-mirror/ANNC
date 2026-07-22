#ifndef ANNC_CUSTOMFUSIONPATTERNBASE_H
#define ANNC_CUSTOMFUSIONPATTERNBASE_H
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir//AtirOps.h"
#include "Dialect/Atir/CustomOpSchema.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "Kernel/KernelPriorityResolver.h"

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

  CustomFusionPatternBase(MLIRContext* context, PatternBenefit benefit = 9)
      :OpRewritePattern<AnchorOp>(context, benefit) {}

 public:
  virtual mlir::LogicalResult matchFusion(
      AnchorOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const = 0;

  virtual std::string getCustomOpName(
      AnchorOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const = 0;

  virtual CustomOpSchema getCustomOpSchema(
      AnchorOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const = 0;

 public:
  mlir::LogicalResult matchAndRewrite(
      AnchorOp anchor,
      mlir::PatternRewriter &rewriter) const override {
    SmallVector<Operation *> fusedOps;
    auto matchres = matchFusion(anchor, fusedOps);
    if (failed(matchres)) {
      return failure();
    }

    llvm::SmallDenseSet<Operation *> fusedSet(fusedOps.begin(), fusedOps.end());

    // Step 1: identify output ops and boundary outputs.
    //
    // An "output op" is a fused op whose SSA results are not consumed by any
    // other fused op — i.e. a leaf in the fused def-use chain.  Its first
    // operand (the output buffer) must be preserved as a CustomizeOp input
    // so the kernel has somewhere to write.  This covers two cases:
    //   - SSA results with external users (SSA-style escape)
    //   - SSA results with no users at all (buffer-style: the output is
    //     written through the output buffer operand, the SSA result is dead)
    //
    // "Boundary outputs" (outputSet) is the subset of output-op results that
    // have external users and thus need replaceAllUsesWith.
    llvm::SetVector<mlir::Value> outputSet;
    llvm::SmallDenseSet<Operation *> outputOps;
    for (Operation *op : fusedOps) {
      bool hasInternalUser = false;
      for (const auto &res : op->getResults()) {
        for (Operation *useOp : res.getUsers()) {
          if (fusedSet.contains(useOp)) {
            hasInternalUser = true;
            break;
          }
          outputSet.insert(res);
        }
        if (hasInternalUser) break;
      }
      if (!hasInternalUser) {
        outputOps.insert(op);
      }
    }
    SmallVector<Value> outputValues(outputSet.begin(), outputSet.end());

    // Step 2a: collect output buffers (the first operand of each output op),
    // ordered by the output op's position in fusedOps.  These become the
    // CustomizeOp's output memref arguments.  When the operand is a BufferOp
    // result, the BufferOp is kept live (NOT erased) since its result remains
    // a CustomizeOp operand.
    llvm::SetVector<mlir::Value> inputSet;
    SmallVector<Operation *> keptBufferOps;
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

    // Step 2b: real boundary inputs — operands whose defining op is neither
    // in fusedOps nor a BufferOp.  BufferOp results that are not output
    // buffers belong to intermediate fused ops and are skipped here (they
    // will be erased together with their host ops in step 3).  Output
    // buffers collected in step 2a are already in inputSet and skipped by
    // the SetVector's deduplication.
    for (Operation *op : fusedOps) {
      for (const auto &operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (fusedSet.contains(defOp)) continue;
        if (defOp && isa<BufferOp>(defOp)) continue;
        inputSet.insert(operand);
      }
    }
    SmallVector<Value> inputValues(inputSet.begin(), inputSet.end());

    SmallVector<Type> resultTypes;
    resultTypes.reserve(outputValues.size());
    for (const auto &value : outputValues) {
      resultTypes.push_back(value.getType());
    }

    auto customOpName = getCustomOpName(anchor, fusedOps);
    auto schema = getCustomOpSchema(anchor, fusedOps);
    auto metadata = schema.toMetadata(rewriter.getContext());

    auto module = anchor->template getParentOfType<mlir::ModuleOp>();
    auto attr = module->template getAttrOfType<mlir::BoolAttr>("annc.enable_kdnn");
    bool enableKdnn = attr && attr.getValue();
    annc::kernels::KernelResolveRequest req;
    req.op_type = customOpName;
    req.type_constraints = inferTypeConstraintsFromSchema(metadata, inputValues);
    if (auto rhsFormat = anchor->template getAttrOfType<mlir::StringAttr>("rhs_format")) {
      req.rhs_format = rhsFormat.getValue().str();
    }
    if (!annc::kernels::hasAnyAvailableKernel(req, enableKdnn)) {
      llvm::dbgs() << "ANNC: No builtin kernel available for '" << customOpName
                   << "', skipping CustomizeOp rewrite\n";
      return failure();
    }

    auto callee = StringAttr::get(rewriter.getContext(), customOpName);

    auto customCallOp = rewriter.create<CustomizeOp>(
        anchor.getLoc(), resultTypes, inputValues, callee, metadata);
    if (auto rhsFormat = anchor->template getAttrOfType<mlir::StringAttr>("rhs_format")) {
      customCallOp->setAttr("rhs_format", rhsFormat);
    }

    // Replace escaping SSA results with the CustomizeOp's results.
    for (auto [oldV, newV] : llvm::zip(outputValues, customCallOp.getResults())) {
      rewriter.replaceAllUsesWith(oldV, newV);
    }

    // Step 3: erase fusedOps plus intermediate BufferOps, in reverse block
    // order to respect def-use.  Kept BufferOps (tied to output ops) are
    // excluded since their results remain live as CustomizeOp operands.
    SmallVector<Operation *> allEraseOps(fusedOps.begin(), fusedOps.end());
    for (Operation *op : fusedOps) {
      if (outputOps.contains(op)) continue;
      if (op->getNumOperands() == 0) continue;
      Operation *defOp = op->getOperand(0).getDefiningOp();
      if (defOp && isa<BufferOp>(defOp) &&
          !llvm::is_contained(keptBufferOps, defOp)) {
        allEraseOps.push_back(defOp);
      }
    }
    llvm::sort(allEraseOps, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    for (Operation *op : llvm::reverse(allEraseOps)) {
      rewriter.eraseOp(op);
    }

    return success();
  }
};

}// namespace atir
#endif  // ANNC_CUSTOMFUSIONPATTERNBASE_H
