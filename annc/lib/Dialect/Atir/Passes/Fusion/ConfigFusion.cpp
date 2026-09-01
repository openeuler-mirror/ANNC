#include "Dialect/Atir/Passes/Passes.h"
#include "ConfigFusionSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace atir;

namespace atir {

// Entry point for atir-config-fusion:
// 1. Load the external JSON configuration.
// 2. Try configured rules by priority until no new fusion is produced.
// 3. Optionally continue with builtin op fusion for pipeline compatibility.
class AtirConfigFusionPass : public AtirConfigFusionBase<AtirConfigFusionPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    // Config materialization creates !llvm.ptr for annc_execution_v2 entry
    // points. Registering the dialect here initializes its type storage in the
    // pass context before the materializer constructs that type.
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto mainFunc = module.lookupSymbol<func::FuncOp>("main");
    if (!mainFunc) return;

    if (config.empty()) {
      mainFunc.emitError() << "missing --config for atir-config-fusion";
      signalPassFailure();
      return;
    }
    if (warnLimit < 0) {
      mainFunc.emitError() << "--warn-limit must be greater than or equal to 0";
      signalPassFailure();
      return;
    }

    auto parsed = config_fusion::parseConfigFile(config);
    if (!parsed) {
      mainFunc.emitError() << llvm::toString(std::move(parsed.takeError()));
      signalPassFailure();
      return;
    }

    std::vector<config_fusion::PatternReport> reports;
    reports.reserve(parsed->patterns.size());
    for (const config_fusion::PatternRule &rule : parsed->patterns) {
      reports.emplace_back(rule.name);
    }

    bool changed = false;
    do {
      changed = false;
      // Materialize one pattern per round and rescan main so later rules do not
      // inspect stale IR after earlier replacements or erasures.
      for (size_t i = 0; i < parsed->patterns.size(); ++i) {
        const config_fusion::PatternRule &rule = parsed->patterns[i];
        if (config_fusion::tryMatchRule(module, mainFunc, rule, &reports[i],
                                        warnLimit)) {
          changed = true;
          break;
        }
      }
    } while (changed);

    config_fusion::emitPatternWarnings(mainFunc, reports);
    if (!report.empty()) {
      if (llvm::Error error =
              config_fusion::writeFusionReport(report, config, reports)) {
        mainFunc.emitError() << llvm::toString(std::move(error));
        signalPassFailure();
        return;
      }
    }

    if (runBuiltinAfter) {
      RewritePatternSet patterns(&getContext());
      populateAtirOpFusionPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(mainFunc, std::move(patterns)))) {
        signalPassFailure();
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAtirConfigFusionPass() {
  return std::make_unique<AtirConfigFusionPass>();
}

}  // namespace atir
