#include "GemmMicrokernelSelection.h"
#include "GemmPlan.h"
#include "Target/aarch64/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace annc {
namespace {

class AArch64GemmMicrokernelLowering
    : public AArch64GemmMicrokernelLoweringBase<
          AArch64GemmMicrokernelLowering> {
 public:
  using Base::Base;

  void runOnOperation() override {
    Builder builder(&getContext());
    getOperation().walk([&](func::CallOp call) {
      const bool isFused =
          aarch64::gemm::isFusedMicrokernelLeaf(call.getCallee());
      if (call.getCallee() != aarch64::gemm::kMicrokernelLeafName &&
          call.getCallee() != aarch64::gemm::kMicrokernelRmLeafName && !isFused)
        return;
      if (!aarch64::gemm::hasStage(call, aarch64::gemm::kPackedStage)) {
        call.emitOpError("requires the packed GEMM leaf stage");
        signalPassFailure();
        return;
      }
      FailureOr<aarch64::gemm::FinalMicrokernelSelection> selection =
          aarch64::gemm::readFinalMicrokernelSelection(call);
      if (failed(selection)) {
        signalPassFailure();
        return;
      }
      std::string symbol = selection->symbol;
      if (isFused) {
        auto generated = call->getAttrOfType<DictionaryAttr>(
            aarch64::gemm::kGeneratedMicrokernelAttrName);
        auto generatedSymbol =
            generated ? generated.getAs<StringAttr>("symbol") : StringAttr();
        if (!generatedSymbol) {
          call.emitOpError("fused leaf has no generated microkernel metadata");
          signalPassFailure();
          return;
        }
        symbol = generatedSymbol.getValue().str();
      }
      call->setDiscardableAttr(aarch64::gemm::kAsmSymbolAttrName,
                               builder.getStringAttr(symbol));
      aarch64::gemm::setStage(call, builder,
                              aarch64::gemm::kMicrokernelLoweredStage);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAArch64GemmMicrokernelLowering() {
  return std::make_unique<AArch64GemmMicrokernelLowering>();
}

}  // namespace annc
