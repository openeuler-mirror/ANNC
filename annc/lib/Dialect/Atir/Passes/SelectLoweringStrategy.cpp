#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/AtirTypes.h.inc"
#include "Dialect/Atir/Passes/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "Helper.h"

using namespace llvm;
using namespace mlir;

namespace atir {

constexpr StringLiteral kConfigAttrName = "lowering_config";

struct SelectLoweringStrategyRewrite : public OpRewritePattern<MatMulOp> {
  SelectLoweringStrategyRewrite(MLIRContext* context, PatternBenefit benefit = 9)
      : OpRewritePattern<MatMulOp>(context, benefit) {}

 public:
  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter& rewriter) const override {
    //todo autotune
    llvm::dbgs() << "this is SelectLoweringStrategyRewrite\n";

    auto lowering_config = op->getAttr(kConfigAttrName);

    if (lowering_config) {
      return failure();
    }

    SmallVector<int64_t> distribution;
    distribution.push_back(128);
    distribution.push_back(128);
    distribution.push_back(0);

    SmallVector<int64_t> cache_parallel;
    cache_parallel.push_back(64);
    cache_parallel.push_back(64);
    cache_parallel.push_back(0);

    SmallVector<int64_t> cache_reduction;
    cache_reduction.push_back(0);
    cache_reduction.push_back(0);
    cache_reduction.push_back(16);

    SmallVector<int64_t> vector_common_parallel;
    vector_common_parallel.push_back(4);
    vector_common_parallel.push_back(4);
    vector_common_parallel.push_back(0);

    SmallVector<int64_t> vector_reduction;
    vector_reduction.push_back(0);
    vector_reduction.push_back(0);
    vector_reduction.push_back(4);

    SmallVector<int64_t> vector_inner_parallel;
    vector_inner_parallel.push_back(0);
    vector_inner_parallel.push_back(0);
    vector_inner_parallel.push_back(0);

    auto loweringConfigAttr = LoweringConfigAttr::get(getContext(), distribution, cache_parallel, cache_reduction, vector_common_parallel, vector_reduction, vector_inner_parallel);
    op->setAttr(kConfigAttrName, loweringConfigAttr);
    return success();
  };

};

class AtirSelectLoweringStrategyPass : public AtirSelectLoweringStrategyBase<AtirSelectLoweringStrategyPass> {
 public:
  AtirSelectLoweringStrategyPass() = default;

  void runOnOperation() override {
    llvm::dbgs() << "this is AtirSelectLoweringStrategyPass\n";
    auto m = getOperation();
    auto ctx = m.getContext();
    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);

    RewritePatternSet patterns(ctx);
    patterns.add<SelectLoweringStrategyRewrite>(ctx);
    (void)applyPatternsGreedily(m, std::move(patterns), config);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAtirSelectLoweringStrategyPass() {
  llvm::dbgs() << "this is createAtirSelectLoweringStrategyPass\n";
  return std::make_unique<AtirSelectLoweringStrategyPass>();
}
}  // namespace pimp