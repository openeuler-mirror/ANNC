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
  if (op->hasAttr(aarch64::gemm::kCandidateAttrName) ||
      op->hasAttr(aarch64::gemm::kPlanAttrName)) {
    return op->emitOpError("already has a GEMM candidate or finalized plan");
  }
  FailureOr<aarch64::gemm::GemmProblem> problem =
      aarch64::gemm::readProblem(op);
  if (failed(problem)) return failure();

  auto selection = aarch64::gemm::selectGemmPath(
      config.isa, problem->m, problem->n, problem->k, enablePrepack,
      op->hasAttr(aarch64::gemm::kRhsNameAttrName));
  if (op->hasAttr(atir::kGemmEpilogueAttr) &&
      selection.executionKind != aarch64::gemm::GemmExecutionKind::kGemm &&
      (selection.executionKind !=
           aarch64::gemm::GemmExecutionKind::kMatrixVector ||
       !epilogueMatvecEnabled())) {
    selection.executionKind = aarch64::gemm::GemmExecutionKind::kGemm;
  }
  aarch64::gemm::GemmKernelTile kernelTile = config.kernel.tile;
  if (selection.executionKind != aarch64::gemm::GemmExecutionKind::kGemm) {
    const aarch64::gemm::GemmKernelABI &abi = aarch64::gemm::getGemmKernelABI(
        config.target, config.isa, config.dataType, selection.executionKind);
    kernelTile = {abi.maxMr, abi.maxPanelLanes};
  }
  aarch64::gemm::writeThreadPlanningInput(op,
                                          {{config.target,
                                            config.isa,
                                            config.dataType,
                                            selection.executionKind,
                                            config.cache,
                                            {config.kernel.kind, kernelTile},
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
