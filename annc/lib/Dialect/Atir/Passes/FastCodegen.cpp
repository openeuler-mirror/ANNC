#include <cstdlib>

#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistry.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;

namespace atir {
namespace {

constexpr char kEnableCustomOpsEnv[] = "ANNC_FAST_CODEGEN_ENABLE_CUSTOM_OPS";
constexpr char kDisableCustomOpsEnv[] = "ANNC_FAST_CODEGEN_DISABLE_CUSTOM_OPS";
constexpr StringLiteral kDefaultDisabledCustomOps[] = {"MatMul", "MatMulAdd",
                                                       "MatMulAddRelu"};

void appendCustomOpTypesFromEnv(const char *name,
                                SmallVectorImpl<std::string> &types) {
  const char *value = std::getenv(name);
  if (!value || !*value) return;

  SmallVector<StringRef> entries;
  StringRef(value).split(entries, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef entry : entries) {
    entry = entry.trim();
    if (!entry.empty()) types.push_back(entry.str());
  }
}

}  // namespace

class AtirFastCodegenPass : public AtirFastCodegenBase<AtirFastCodegenPass> {
 public:
  AtirFastCodegenPass() = default;
  AtirFastCodegenPass(const AtirFastCodegenPass &) = default;

  void runOnOperation() override {
    auto m = getOperation();
    auto ctx = m.getContext();

    // Set annc.enable_kdnn module attribute from the --enable-kdnn option.
    // This flag controls whether KDNN optimized kernels are preferred
    // over aarch64 defaults during kernel resolution.
    if (enableKdnn) {
      m->setAttr("annc.enable_kdnn", BoolAttr::get(ctx, true));
    }

    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);

    RewritePatternSet patterns(ctx);
    SmallVector<std::string> environmentEnabledCustomOps;
    appendCustomOpTypesFromEnv(kEnableCustomOpsEnv,
                               environmentEnabledCustomOps);
    SmallVector<std::string> effectiveEnabledCustomOps;
    if (enabledCustomOps.empty()) {
      effectiveEnabledCustomOps = std::move(environmentEnabledCustomOps);
    } else {
      effectiveEnabledCustomOps.append((*enabledCustomOps).begin(),
                                       (*enabledCustomOps).end());
    }

    SmallVector<std::string> effectiveDisabledCustomOps;
    appendCustomOpTypesFromEnv(kDisableCustomOpsEnv,
                               effectiveDisabledCustomOps);
    effectiveDisabledCustomOps.append((*disabledCustomOps).begin(),
                                      (*disabledCustomOps).end());
    for (StringRef type : kDefaultDisabledCustomOps) {
      bool explicitlyEnabled = llvm::any_of(
          effectiveEnabledCustomOps,
          [&](const std::string &entry) { return StringRef(entry) == type; });
      if (!explicitlyEnabled) effectiveDisabledCustomOps.push_back(type.str());
    }

    CustomOpTypeFilter customOpFilter(effectiveEnabledCustomOps,
                                      effectiveDisabledCustomOps);
    PatternRegistry::instance().populatePatterns(patterns, customOpFilter);
    (void)applyPatternsGreedily(m, std::move(patterns), config);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAtirFastCodegenPass() {
  return std::make_unique<AtirFastCodegenPass>();
}
}  // namespace atir
