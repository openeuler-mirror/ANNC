#include "GemmPlan.h"
#include "Target/aarch64/Passes.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace annc {
namespace {

FailureOr<int64_t> getStaticMicrokernelDimension(func::CallOp call,
                                                 StringRef name) {
  auto value = call->getAttrOfType<IntegerAttr>(name);
  if (!value || value.getInt() <= 0) {
    call.emitOpError() << "requires positive static attribute " << name;
    return failure();
  }
  return value.getInt();
}

FailureOr<std::string> selectMicrokernelSymbol(func::CallOp call) {
  FailureOr<int64_t> m =
      getStaticMicrokernelDimension(call, aarch64::gemm::kMicrokernelMAttrName);
  FailureOr<int64_t> n =
      getStaticMicrokernelDimension(call, aarch64::gemm::kMicrokernelNAttrName);
  FailureOr<int64_t> k =
      getStaticMicrokernelDimension(call, aarch64::gemm::kMicrokernelKAttrName);
  if (failed(m) || failed(n) || failed(k)) return failure();
  FailureOr<aarch64::gemm::GemmPlan> plan = aarch64::gemm::readPlan(call);
  if (failed(plan)) return failure();
  FailureOr<int64_t> nr = aarch64::gemm::getGemmNr(
      plan->kernelTile, plan->vectorLengthBytes, plan->dataType);
  if (failed(nr)) {
    call.emitOpError("has an invalid kernel N tile");
    return failure();
  }
  if (*m > plan->kernelTile.mr || *n > *nr) {
    call.emitOpError("has a microtile outside the static kernel family");
    return failure();
  }

  auto kcMode = call->getAttrOfType<StringAttr>(aarch64::gemm::kKcModeAttrName);
  if (!kcMode ||
      (kcMode.getValue() != "overwrite" && kcMode.getValue() != "accumulate")) {
    call.emitOpError("requires a valid static KC mode");
    return failure();
  }

  if (plan->isa == aarch64::gemm::GemmIsa::kSve) {
    const int64_t vectorLanes =
        plan->vectorLengthBytes / static_cast<int64_t>(sizeof(float));
    const int64_t nGroups = (*n + vectorLanes - 1) / vectorLanes;
    if (nGroups < 1 || nGroups > plan->kernelTile.panelLanes) {
      call.emitOpError("has an invalid SVE N-vector group count");
      return failure();
    }
    return (llvm::Twine("annc_aarch64_sve_kernel_mr") + llvm::Twine(*m) +
            "_n" + llvm::Twine(nGroups) + "vl" +
            (kcMode.getValue() == "accumulate" ? "_acc_f32" : "_f32"))
        .str();
  }

  const aarch64::gemm::GemmKernelABI &abi =
      aarch64::gemm::getGemmKernelABI(plan->target, plan->isa,
                                      plan->dataType);
  if (abi.kUnroll <= 0) {
    call.emitOpError("requires a positive NEON kernel ABI K unroll");
    return failure();
  }
  const int64_t groups = *k / abi.kUnroll;
  const int64_t residue = *k % abi.kUnroll;
  std::string kVariant;
  if (groups == 0) {
    kVariant = (llvm::Twine("k") + llvm::Twine(residue)).str();
  } else {
    kVariant = (llvm::Twine("kg_r") + llvm::Twine(residue)).str();
  }
  return (llvm::Twine("annc_aarch64_neon_kernel_mr") + llvm::Twine(*m) + "_n" +
          llvm::Twine(*n) + "_" + kVariant +
          (kcMode.getValue() == "accumulate" ? "_acc_f32" : "_f32"))
      .str();
}

class AArch64GemmMicrokernelLowering
    : public AArch64GemmMicrokernelLoweringBase<
          AArch64GemmMicrokernelLowering> {
 public:
  using Base::Base;

  void runOnOperation() override {
    Builder builder(&getContext());
    getOperation().walk([&](func::CallOp call) {
      if (call.getCallee() != aarch64::gemm::kMicrokernelLeafName) return;
      if (!aarch64::gemm::hasStage(call, aarch64::gemm::kPackedStage)) {
        call.emitOpError("requires the packed GEMM leaf stage");
        signalPassFailure();
        return;
      }
      FailureOr<std::string> symbol = selectMicrokernelSymbol(call);
      if (failed(symbol)) {
        signalPassFailure();
        return;
      }
      call->setDiscardableAttr(aarch64::gemm::kAsmSymbolAttrName,
                               builder.getStringAttr(*symbol));
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
