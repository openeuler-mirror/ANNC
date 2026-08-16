#include "GemmPlan.h"

#include <cstdlib>
#include <optional>

#include "Target/aarch64/Passes.h"
#include "llvm/Support/Error.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace annc {
namespace {

FailureOr<int64_t> getIntraThreadCount(ModuleOp module) {
  auto attr = module->getAttrOfType<IntegerAttr>(
      aarch64::gemm::kIntraThreadCountAttrName);
  // The annc-asm caller sets this module attribute from the available intra-op
  // thread count. Omitted attributes intentionally keep GEMM execution serial.
  if (!attr) return int64_t{1};
  if (attr.getInt() <= 0) {
    module.emitError() << aarch64::gemm::kIntraThreadCountAttrName
                       << " must be a positive i64 module attribute";
    return failure();
  }
  return attr.getInt();
}

LogicalResult selectGemmStrategy(
    Operation *op, int64_t intraThreadCount,
    const aarch64::gemm::GemmTuningConfig &config) {
  if (!op->hasAttr(aarch64::gemm::kProblemAttrName)) return success();
  if (op->hasAttr(aarch64::gemm::kCandidateAttrName) ||
      op->hasAttr(aarch64::gemm::kPlanAttrName)) {
    return op->emitOpError("already has a GEMM candidate or finalized plan");
  }
  FailureOr<aarch64::gemm::GemmProblem> problem =
      aarch64::gemm::readProblem(op);
  if (failed(problem)) return failure();

  Builder builder(op->getContext());
  NamedAttrList candidate;
  candidate.append("version",
                   builder.getI64IntegerAttr(aarch64::gemm::kPlanVersion));
  const aarch64::gemm::GemmKernelABI &abi = aarch64::gemm::getGemmKernelABI(
      config.target, config.isa, config.dataType);
  candidate.append(
      "target_arch",
      builder.getStringAttr(aarch64::gemm::getGemmTargetName(config.target)));
  candidate.append(
      "isa", builder.getStringAttr(aarch64::gemm::getGemmIsaName(config.isa)));
  candidate.append("data_type",
                   builder.getStringAttr(
                       aarch64::gemm::getGemmDataTypeName(config.dataType)));
  candidate.append("kernel_family", builder.getStringAttr(abi.family));
  candidate.append("mc", builder.getI64IntegerAttr(config.cacheTile.mc));
  candidate.append("nc", builder.getI64IntegerAttr(config.cacheTile.nc));
  candidate.append("kc", builder.getI64IntegerAttr(config.cacheTile.kc));
  candidate.append("mr", builder.getI64IntegerAttr(config.kernelTile.mr));
  candidate.append("panel_lanes",
                   builder.getI64IntegerAttr(config.kernelTile.panelLanes));
  candidate.append("thread_count", builder.getI64IntegerAttr(intraThreadCount));
  candidate.append("thread_partition", builder.getStringAttr("static-2d"));
  op->setDiscardableAttr(aarch64::gemm::kCandidateAttrName,
                         candidate.getDictionary(op->getContext()));
  return success();
}

class AArch64SelectGemmStrategy
    : public AArch64SelectGemmStrategyBase<AArch64SelectGemmStrategy> {
 public:
  using Base::Base;

  explicit AArch64SelectGemmStrategy(StringRef path) {
    configPath = path.str();
  }

  void runOnOperation() override {
    FailureOr<int64_t> intraThreadCount = getIntraThreadCount(getOperation());
    if (failed(intraThreadCount)) {
      signalPassFailure();
      return;
    }
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
      return failed(selectGemmStrategy(op, *intraThreadCount, *config))
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
    llvm::StringRef configPath) {
  return std::make_unique<AArch64SelectGemmStrategy>(configPath);
}

}  // namespace annc
