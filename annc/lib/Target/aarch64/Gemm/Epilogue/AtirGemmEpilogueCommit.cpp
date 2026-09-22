#include <utility>

#include "../GemmPlan.h"
#include "Dialect/Atir/Passes/GemmEpilogueCandidate.h"
#include "Target/aarch64/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
namespace annc {
namespace {
class AArch64AtirGemmEpilogueCommit
    : public AArch64AtirGemmEpilogueCommitBase<AArch64AtirGemmEpilogueCommit> {
 public:
  using Base::Base;

  void runOnOperation() override {
    llvm::SmallVector<atir::MatMulOp> candidates;
    getOperation().walk([&](atir::MatMulOp matmul) {
      if (matmul->hasAttr(atir::kGemmEpilogueAttr)) return;
      candidates.push_back(matmul);
    });
    for (atir::MatMulOp matmul : candidates)
      if (mlir::failed(commitMatmul(matmul))) {
        signalPassFailure();
        return;
      }
  }

 private:
  mlir::LogicalResult commitMatmul(atir::MatMulOp matmul) {
    auto committedAttr = matmul->getAttrOfType<mlir::DictionaryAttr>(
        atir::kGemmEpilogueCandidateAttr);
    if (!committedAttr) return mlir::success();
    if (matmul.getWithBias() || matmul.getDoRelu())
      return matmul.emitOpError("legacy MatMul Bias/Relu attributes conflict with annc.gemm.epilogue");

    auto parsed = atir::parseEpilogueCandidatePlan(committedAttr);
    if (mlir::failed(parsed) ||
        mlir::failed(atir::bindEpilogueProgramToSource(matmul, *parsed)))
      return matmul.emitOpError(
          "epilogue candidate does not match the current source chain");
    atir::EpilogueProgram plan = std::move(*parsed);

    llvm::SmallVector<mlir::Value> epilogueInputs;
    for (const atir::EpilogueStep &step : plan.steps) {
      if (!atir::isBinaryEpilogueOpcode(step.opcode)) continue;
      mlir::Operation *source = step.source;
      epilogueInputs.push_back(step.inputIndex == 1 ? source->getOperand(2)
                                                    : source->getOperand(1));
    }

    mlir::Value terminalDestination = plan.steps.back().source->getOperand(0);
    mlir::OpBuilder builder(matmul);
    builder.setInsertionPoint(plan.steps.back().source);
    auto newMatmul = builder.create<atir::MatMulOp>(
        matmul.getLoc(), matmul.getOutput().getType(), terminalDestination,
        matmul.getLhs(), matmul.getRhs(), epilogueInputs,
        builder.getBoolAttr(false),
        builder.getBoolAttr(matmul.getRightTranspose()),
        builder.getBoolAttr(matmul.getLeftTranspose()),
        builder.getBoolAttr(matmul.getOutputTranspose()),
        builder.getBoolAttr(false), builder.getF32FloatAttr(-1.0f),
        matmul.getMStartAttr(), matmul.getNStartAttr(), matmul.getKStartAttr(),
        matmul.getMSizeAttr(), matmul.getNSizeAttr(), matmul.getKSizeAttr(),
        matmul->getAttrOfType<mlir::StringAttr>("rhs_format"));
    newMatmul->setAttr(atir::kGemmEpilogueAttr, committedAttr);

    plan.steps.back().result.replaceAllUsesWith(newMatmul.getOutput());
    for (auto it = plan.steps.rbegin(); it != plan.steps.rend(); ++it)
      it->source->erase();
    matmul.erase();
    return mlir::success();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAArch64AtirGemmEpilogueCommit() {
  return std::make_unique<AArch64AtirGemmEpilogueCommit>();
}

}  // namespace annc
