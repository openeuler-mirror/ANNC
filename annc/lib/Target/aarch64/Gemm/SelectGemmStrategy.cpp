#include "GemmPlan.h"

#include <cstdlib>
#include <optional>

#include "Dialect/Atir/Passes/GemmEpilogueCandidate.h"
#include "Target/aarch64/Passes.h"
#include "llvm/Support/Error.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace annc {
namespace {

bool epilogueMatvecEnabled() {
  const char *value = std::getenv("ANNC_ENABLE_EPILOGUE_MATVEC");
  return !value || StringRef(value) != "0";
}

LogicalResult selectGemmStrategy(
    Operation *op, int64_t intraThreadCount,
    const aarch64::gemm::GemmTuningConfig &config, bool enablePrepack) {
  if (!op->hasAttr(aarch64::gemm::kProblemAttrName)) return success();
  if (op->hasAttr(aarch64::gemm::kCandidateAttrName)) {
    return op->emitOpError("already has a GEMM candidate");
  }
  FailureOr<aarch64::gemm::GemmProblem> problem =
      aarch64::gemm::readProblem(op);
  if (failed(problem)) return failure();

  const auto selectPath = [&](bool allowSpecialized) {
    return aarch64::gemm::selectGemmPath(
        config.isa, problem->m, problem->n, problem->k, enablePrepack,
        op->hasAttr(aarch64::gemm::kRhsNameAttrName), allowSpecialized);
  };
  const auto &kernelPolicy =
      aarch64::gemm::getGemmKernelTilingStrategy(config.kernel.kind);
  const auto configureKernel = [&](aarch64::gemm::GemmExecutionKind kind)
      -> std::optional<aarch64::gemm::GemmKernelStrategy> {
    const auto &abi = aarch64::gemm::getGemmKernelABI(
        config.target, config.isa, config.dataType, kind);
    auto kernel = kernelPolicy.configureForExecution(config.kernel, kind,
                                                     config.isa, abi);
    if (!kernel || !kernelPolicy.isValidForExecution(
                       *kernel, kind, config.isa, config.dataType, abi))
      return std::nullopt;
    return kernel;
  };
  auto selection = selectPath(true);
  if (op->hasAttr(atir::kGemmEpilogueAttr) &&
      selection.executionKind != aarch64::gemm::GemmExecutionKind::kGemm &&
      (selection.executionKind !=
           aarch64::gemm::GemmExecutionKind::kMatrixVector ||
       !epilogueMatvecEnabled()))
    selection.executionKind = aarch64::gemm::GemmExecutionKind::kGemm;
  auto kernel = configureKernel(selection.executionKind);
  if (!kernel &&
      selection.executionKind != aarch64::gemm::GemmExecutionKind::kGemm) {
    selection = selectPath(false);
    kernel = configureKernel(selection.executionKind);
  }
  if (!kernel)
    return op->emitOpError("has a tuning strategy unsupported by the kernel ABI");
  if (selection.rhsPacking == aarch64::gemm::RhsPacking::kPrepacked &&
      !aarch64::gemm::isRhsPrepackSupported(config.cache.kind,
                                           kernel->kind)) {
    return op->emitOpError(
        "prepacked RHS currently requires fixed cache and kernel strategies");
  }
  aarch64::gemm::writeThreadPlanningInput(op,
                                          {{config.target,
                                            config.isa,
                                            config.dataType,
                                            selection.executionKind,
                                            config.cache,
                                            *kernel,
                                            selection.rhsPacking},
                                           config.costModel,
                                           intraThreadCount});
  return success();
}

class AArch64SelectGemmStrategy
    : public AArch64SelectGemmStrategyBase<AArch64SelectGemmStrategy> {
 public:
  using Base::Base;

  explicit AArch64SelectGemmStrategy(StringRef path, bool allowPrepack,
                                     int64_t threads = 1) {
    configPath = path.str();
    enablePrepack = allowPrepack;
    intraThreadCount = threads;
  }

  void runOnOperation() override {
    // The annc-asm caller supplies the runtime intra-op thread count through
    // the pipeline / pass option; 1 keeps GEMM execution serial.
    if (intraThreadCount <= 0) {
      getOperation()->emitError(
          "aarch64-select-gemm-strategy intra-thread-count must be positive");
      signalPassFailure();
      return;
    }
    int64_t threadCount = intraThreadCount;
    std::string path = configPath;
    if (path.empty()) {
      if (const char *environment = std::getenv("ANNC_GEMM_CONFIG"))
        path = environment;
    }
    std::optional<aarch64::gemm::GemmTuningConfig> config;
    WalkResult result = getOperation().walk([&](Operation *op) -> WalkResult {
      if (!aarch64::gemm::isGemmAnchor(op)) return WalkResult::advance();
      if (!op->hasAttr(aarch64::gemm::kProblemAttrName))
        return WalkResult::advance();
      if (!config) {
        if (path.empty()) {
          op->emitOpError(
              "requires --config-path or ANNC_GEMM_CONFIG for AArch64 GEMM");
          return WalkResult::interrupt();
        }
        auto loaded = aarch64::gemm::loadGemmTuningConfig(path);
        if (!loaded) {
          op->emitOpError() << llvm::toString(loaded.takeError());
          return WalkResult::interrupt();
        }
        config = *loaded;
      }
      return failed(selectGemmStrategy(op, threadCount, *config, enablePrepack))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    if (result.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAArch64SelectGemmStrategy() {
  return std::make_unique<AArch64SelectGemmStrategy>();
}

std::unique_ptr<mlir::Pass> createAArch64SelectGemmStrategy(
    llvm::StringRef configPath, bool enablePrepack, int64_t intraThreadCount) {
  return std::make_unique<AArch64SelectGemmStrategy>(configPath, enablePrepack,
                                                     intraThreadCount);
}

}  // namespace annc
