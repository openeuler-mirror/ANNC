#include "GemmPlan.h"
#include "Target/aarch64/Gemm/Epilogue/EpilogueEmitter.h"
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

FailureOr<std::string> selectMicrokernelSymbol(func::CallOp call,
                                               bool hasRowMajorCallee) {
  FailureOr<int64_t> m =
      getStaticMicrokernelDimension(call, aarch64::gemm::kMicrokernelMAttrName);
  FailureOr<int64_t> n =
      getStaticMicrokernelDimension(call, aarch64::gemm::kMicrokernelNAttrName);
  FailureOr<int64_t> k =
      getStaticMicrokernelDimension(call, aarch64::gemm::kMicrokernelKAttrName);
  if (failed(m) || failed(n) || failed(k)) return failure();
  FailureOr<aarch64::gemm::GemmPlan> plan = aarch64::gemm::readPlan(call);
  if (failed(plan)) return failure();
  const bool isRowMajor = aarch64::gemm::usesLdbAbi(
    aarch64::gemm::getGemmLeafKind(plan->executionKind, plan->rhsPacking));
  if (isRowMajor != hasRowMajorCallee) {
    call.emitOpError("microkernel leaf does not match the GEMM plan");
    return failure();
  }
  FailureOr<int64_t> nr = aarch64::gemm::getGemmNr(
      plan->kernelTile, plan->vectorLengthBytes, plan->dataType,
      plan->executionKind);
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

  llvm::StringRef prefix = isRowMajor ? "annc_aarch64_sve_kernel_rm_mr"
                                      : "annc_aarch64_sve_kernel_mr";
  if (plan->isa == aarch64::gemm::GemmIsa::kSve) {
    const int64_t vectorLanes =
        plan->vectorLengthBytes / static_cast<int64_t>(sizeof(float));
    const int64_t nGroups = (*n + vectorLanes - 1) / vectorLanes;
    if (nGroups < 1 || nGroups > plan->kernelTile.panelLanes) {
      call.emitOpError("has an invalid SVE N-vector group count");
      return failure();
    }
    return (llvm::Twine(prefix) + llvm::Twine(*m) + "_n" +
            llvm::Twine(nGroups) + "vl" +
            (kcMode.getValue() == "accumulate" ? "_acc_f32" : "_f32"))
        .str();
  }

  const aarch64::gemm::GemmKernelABI &abi =
      aarch64::gemm::getGemmKernelABI(plan->target, plan->isa,
                                      plan->dataType, plan->executionKind);
  FailureOr<int64_t> kScalarUnroll =
      aarch64::gemm::getGemmKScalarUnroll(abi, plan->dataType);
  if (failed(kScalarUnroll)) {
    call.emitOpError("requires a valid NEON kernel ABI vector K unroll");
    return failure();
  }
  const int64_t groups = *k / *kScalarUnroll;
  const int64_t residue = *k % *kScalarUnroll;
  std::string kVariant;
  if (groups == 0) {
    kVariant = (llvm::Twine("k") + llvm::Twine(residue)).str();
  } else {
    kVariant = (llvm::Twine("kg_r") + llvm::Twine(residue)).str();
  }
  if (plan->executionKind ==
      aarch64::gemm::GemmExecutionKind::kMatrixVector) {
    if (*n != 1 || *m > 4 || abi.kVectorUnroll != 4) {
      call.emitOpError("has an invalid matrix-vector microtile");
      return failure();
    }
    return (llvm::Twine("annc_aarch64_neon_matvec_mr") + llvm::Twine(*m) +
            "_" + kVariant +
            (kcMode.getValue() == "accumulate" ? "_acc_f32" : "_f32"))
        .str();
  }
  if (plan->executionKind ==
      aarch64::gemm::GemmExecutionKind::kVectorMatrix) {
    if (!isRowMajor || *m != 1 || *n > 16 || abi.kVectorUnroll != 2) {
      call.emitOpError("has an invalid vector-matrix microtile");
      return failure();
    }
    return (llvm::Twine("annc_aarch64_neon_vecmat_n") + llvm::Twine(*n) +
            "_" + kVariant +
            (kcMode.getValue() == "accumulate" ? "_acc_f32" : "_f32"))
        .str();
  }
  prefix = isRowMajor ? "annc_aarch64_neon_kernel_rm_mr"
                      : "annc_aarch64_neon_kernel_mr";
  return (llvm::Twine(prefix) + llvm::Twine(*m) + "_n" + llvm::Twine(*n) +
          "_" + kVariant +
          (kcMode.getValue() == "accumulate" ? "_acc_f32" : "_f32"))
      .str();
}

FailureOr<std::string> getGeneratedMicrokernelSymbol(func::CallOp call,
                                                     bool isRowMajor,
                                                     bool isFused) {
  auto generated = call->getAttrOfType<DictionaryAttr>(
      aarch64::gemm::kGeneratedMicrokernelAttrName);
  if (!generated) return failure();
  auto symbol = generated.getAs<StringAttr>("symbol");
  if (!symbol || symbol.getValue().empty()) {
    call.emitOpError("has an invalid generated microkernel metadata attr");
    return failure();
  }
  if (!isFused || !symbol.getValue().contains("_generated_")) {
    call.emitOpError(
        "generated metadata is attached to an ordinary or static symbol");
    return failure();
  }
  FailureOr<std::string> staticSymbol =
      selectMicrokernelSymbol(call, isRowMajor);
  if (failed(staticSymbol)) return failure();
  auto kcMode =
      call->getAttrOfType<mlir::StringAttr>(aarch64::gemm::kKcModeAttrName);
  if (!kcMode) return failure();
  llvm::StringRef modeSuffix =
      kcMode.getValue() == "accumulate" ? "_acc_f32" : "_f32";
  llvm::StringRef expectedBase(*staticSymbol);
  if (!expectedBase.ends_with(modeSuffix)) return failure();
  expectedBase = expectedBase.drop_back(modeSuffix.size());
  std::string expectedPrefix = (expectedBase + "_generated_").str();
  if (!symbol.getValue().starts_with(expectedPrefix) ||
      !symbol.getValue().ends_with(modeSuffix)) {
    call.emitOpError(
        "generated microkernel symbol does not match the selected kernel ABI");
    return failure();
  }
  return symbol.getValue().str();
}

class AArch64GemmMicrokernelLowering
    : public AArch64GemmMicrokernelLoweringBase<
          AArch64GemmMicrokernelLowering> {
 public:
  using Base::Base;

  void runOnOperation() override {
    Builder builder(&getContext());
    getOperation().walk([&](func::CallOp call) {
      const bool isRowMajor =
          aarch64::gemm::isRowMajorMicrokernelLeaf(call.getCallee());
      const bool isFused =
          aarch64::gemm::isFusedMicrokernelLeaf(call.getCallee());
      if (call.getCallee() != aarch64::gemm::kMicrokernelLeafName &&
          call.getCallee() != aarch64::gemm::kMicrokernelRmLeafName &&
          !isFused)
        return;
      if (isFused &&
          !call->hasAttr(aarch64::gemm::kGeneratedMicrokernelAttrName)) {
        call.emitOpError("fused leaf has no generated microkernel metadata");
        signalPassFailure();
        return;
      }
      if (!aarch64::gemm::hasStage(call, aarch64::gemm::kPackedStage)) {
        call.emitOpError("requires the packed GEMM leaf stage");
        signalPassFailure();
        return;
      }
      FailureOr<std::string> symbol =
          call->hasAttr(aarch64::gemm::kGeneratedMicrokernelAttrName)
              ? getGeneratedMicrokernelSymbol(call, isRowMajor, isFused)
              : selectMicrokernelSymbol(call, isRowMajor);
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
