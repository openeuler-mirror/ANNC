#ifndef ANNC_CUSTOMFUSIONPATTERNBASE_H
#define ANNC_CUSTOMFUSIONPATTERNBASE_H
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir//AtirOps.h"
#include "Dialect/Atir/CustomOpSchema.h"
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
