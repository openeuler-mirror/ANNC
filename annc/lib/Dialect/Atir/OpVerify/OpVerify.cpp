#include "Dialect/Atir/OpVerify/OpVerify.h"
#include "Dialect/Atir/OpVerify/IoDef.h"
#include "Dialect/Atir/OpVerify/compare.h"
#include "Dialect/Atir/OpVerify/PerformanceMetrics.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <map>
#include <vector>

using namespace llvm;
using namespace mlir;

namespace atir {

class AtirOpVerifyPass : public AtirOpVerifyBase<AtirOpVerifyPass> {
 public:
  AtirOpVerifyPass() = default;

  void runOnOperation() override { }
};

std::unique_ptr<OperationPass<ModuleOp>> createAtirOpVerifyPass() {
  return std::make_unique<AtirOpVerifyPass>();
}
}  // namespace atir