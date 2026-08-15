#include "GemmPlan.h"
#include "Target/aarch64/Passes.h"

namespace annc {
namespace {

class AArch64GemmEpilogueLowering
    : public AArch64GemmEpilogueLoweringBase<AArch64GemmEpilogueLowering> {
 public:
  using Base::Base;

  void runOnOperation() override {
    WalkResult result = getOperation().walk([](Operation *op) -> WalkResult {
      if (!op->hasAttr(aarch64::gemm::kEpilogueAttrName))
        return WalkResult::advance();
      op->emitOpError(
          "AArch64 GEMM epilogue lowering is not implemented");
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAArch64GemmEpilogueLowering() {
  return std::make_unique<AArch64GemmEpilogueLowering>();
}

}  // namespace annc
