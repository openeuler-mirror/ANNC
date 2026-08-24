#include "GemmPlan.h"
#include "Target/aarch64/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace annc {
namespace {

class AArch64VerifyGemmSchedule
    : public AArch64VerifyGemmScheduleBase<AArch64VerifyGemmSchedule> {
 public:
  using Base::Base;

  void runOnOperation() override {
    bool invalid = false;
    getOperation().walk([&](Operation *op) {
      if (aarch64::gemm::isGemmAnchor(op) &&
          op->hasAttr(aarch64::gemm::kPlanAttrName)) {
        op->emitOpError("was not consumed by GEMM lowering");
        invalid = true;
      }
      if (auto call = llvm::dyn_cast<func::CallOp>(op)) {
        StringRef callee = call.getCallee();
        if (callee == aarch64::gemm::kPackBLeafName ||
            callee == aarch64::gemm::kMicrokernelLeafName ||
            callee == aarch64::gemm::kMicrokernelRmLeafName ||
            callee == aarch64::gemm::kSvePackedBElementsAsmSymbol ||
            callee == aarch64::gemm::kSvePackedBOffsetAsmSymbol) {
          call.emitOpError("target-private GEMM leaf escaped ABI lowering");
          invalid = true;
        }
      }
    });
    if (invalid) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAArch64VerifyGemmSchedule() {
  return std::make_unique<AArch64VerifyGemmSchedule>();
}

}  // namespace annc
