#ifndef ANNC_CUSTOMFUSIONPATTERNBASE_H
#define ANNC_CUSTOMFUSIONPATTERNBASE_H
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir//AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"
#include "Kernel/KernelPriorityResolver.h"

namespace atir {

template <typename AnchorOp>
struct CustomFusionPatternBase : public mlir::OpRewritePattern<AnchorOp> {

  CustomFusionPatternBase(MLIRContext* context, PatternBenefit benefit = 9)
      :OpRewritePattern<AnchorOp>(context, benefit) {}

 public:
  virtual mlir::LogicalResult matchFusion(
      AnchorOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const = 0;

  virtual std::string getKernelName(
      AnchorOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const {
    return anchor->getName().getStringRef().str();
  }

 public:
  mlir::LogicalResult matchAndRewrite(
      AnchorOp anchor,
      mlir::PatternRewriter &rewriter) const override {
    SmallVector<Operation *> fusedOps;
    auto matchres = matchFusion(anchor, fusedOps);
    if (failed(matchres)) {
      return failure();
    }

    // Check whether any builtin kernel is available for this op.
    // If no kernel exists for any backend, the pattern should not fire —
    // the op will take its default lowering path instead.
    auto kernelName = getKernelName(anchor, fusedOps);
    auto module = anchor->template getParentOfType<mlir::ModuleOp>();
    auto attr = module->template getAttrOfType<mlir::BoolAttr>("annc.enable_kdnn");
    bool enableKdnn = attr && attr.getValue();
    annc::kernels::KernelResolveRequest req;
    req.op_type = kernelName;
    if (!annc::kernels::hasAnyAvailableKernel(req, enableKdnn)) {
      llvm::dbgs() << "ANNC: No builtin kernel available for '" << kernelName
                   << "', skipping CustomizeOp rewrite\n";
      return failure();
    }

    SmallVector<Value> inputValues;

    llvm::SmallDenseSet<Operation *> fusedSet(fusedOps.begin(), fusedOps.end());
    //inputs
    for (const auto &op: fusedOps) {
      for (const auto &operand: op->getOperands()) {
        auto defOp = operand.getDefiningOp();
        if (!fusedSet.contains(defOp)) {
          inputValues.push_back(operand);
        }
      }
    }
    //output
    SmallVector<Value> outputValues;
    for (const auto &op: fusedOps) {
      for (const auto &res: op->getResults()) {
        for (const auto &useOp: res.getUsers()) {
          if (!useOp || !fusedSet.contains(useOp)) {
            outputValues.push_back(res);
          }
        }
      }
    }
    if (outputValues.empty()) {
      llvm::dbgs() << "outputValues is empty\n";
    }
    //result types
    SmallVector<Type> resultTypes;
    for (const auto &value: outputValues) {
      resultTypes.push_back(value.getType());
    }
    //
    auto callee = StringAttr::get(rewriter.getContext(), kernelName);

    // metadata is initialized as an empty DictionaryAttr here; runtime
    // attributes (kernel_name, shared_lib_path, Nconstants, etc.) will be
    // populated by downstream passes (e.g. the ANNC compilation pipeline).
    auto customCallOp = rewriter.create<CustomizeOp>(
        anchor.getLoc(), resultTypes, inputValues, callee, DictionaryAttr());

    //replace
    for (auto [oldV, newV] : llvm::zip(outputValues, customCallOp.getResults())) {
      rewriter.replaceAllUsesWith(oldV,newV);
    }

    //erase
    for (Operation *op : llvm::reverse(fusedOps)) {
      rewriter.eraseOp(op);
    }

    return success();
  }
};

}// namespace atir
#endif  // ANNC_CUSTOMFUSIONPATTERNBASE_H
