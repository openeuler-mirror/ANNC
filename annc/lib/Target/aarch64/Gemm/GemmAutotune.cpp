#include "Target/aarch64/Passes.h"

namespace annc {
namespace {

class AArch64AutotuneGemmPlan
    : public AArch64AutotuneGemmPlanBase<AArch64AutotuneGemmPlan> {
 public:
  using Base::Base;

  // Preserve the baseline candidate when no tuning policy is configured; a
  // configured tuner may replace it before finalization.
  void runOnOperation() override {}
};

}  // namespace

std::unique_ptr<mlir::Pass> createAArch64AutotuneGemmPlan() {
  return std::make_unique<AArch64AutotuneGemmPlan>();
}

}  // namespace annc
