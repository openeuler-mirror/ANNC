#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/AtirTypes.h.inc"
#include "Dialect/Atir/Passes/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "Helper.h"
#include "Support/Log.h"

using namespace llvm;
using namespace mlir;

namespace atir {

    //todo
    //patternOp CanonicalizeOp
    struct FuseReluRewrite : public OpRewritePattern<ReluOp> {
        FuseReluRewrite(MLIRContext* context, PatternBenefit benefit = 9)
                : OpRewritePattern<ReluOp>(context, benefit) {}

    public:
        LogicalResult matchAndRewrite(ReluOp op,
                                      PatternRewriter& rewriter) const override {
            ANNC_LOG_DEBUG("block-fusion") << "this is FuseReluRewrite\n";
            auto reluSourceOp = op.getInput().getDefiningOp();
            if (!reluSourceOp->getResult(0).hasOneUse()) {
                return failure();
            }
            auto relu_limit = op.getReluLimit().convertToFloat();

            if (!reluSourceOp->hasAttr("do_relu")) {
                return failure();
            }

            auto do_relu = llvm::dyn_cast<mlir::BoolAttr>(reluSourceOp->getAttr("do_relu")).getValue();

            if (do_relu && reluSourceOp->hasAttr("relu_limit")) {
                auto old_limit = llvm::dyn_cast<mlir::FloatAttr>(reluSourceOp->getAttr("relu_limit")).getValue().convertToFloat();
                if (old_limit > relu_limit) {
                    relu_limit = old_limit;
                }
            }

            reluSourceOp->setAttr("do_relu", rewriter.getBoolAttr(true));
            reluSourceOp->setAttr("relu_limit", rewriter.getF32FloatAttr(relu_limit));

            reluSourceOp->setLoc(op.getLoc());

            if (llvm::isa<atir::AddOp>(reluSourceOp)) {
                auto addOp = llvm::dyn_cast<atir::AddOp>(reluSourceOp);

                std::vector<NamedAttribute> attrs;
                attrs.push_back(NamedAttribute(rewriter.getStringAttr("do_relu"),
                                               rewriter.getBoolAttr(true)));

                attrs.push_back(NamedAttribute(rewriter.getStringAttr("relu_limit"),
                                               rewriter.getF32FloatAttr(relu_limit)));

                if (addOp.getScalar()) {
                    attrs.push_back(NamedAttribute(rewriter.getStringAttr("scalar"),
                                                   rewriter.getF32FloatAttr(addOp.getScalar()->convertToFloat())));
                }

                std::vector<Type> outs;
                outs.push_back(op.getResult().getType());

                auto newAddOp = rewriter.create<atir::AddOp>(op.getLoc(), outs,
                                                            reluSourceOp->getOperands(),attrs);

                op.getResult().replaceAllUsesWith(newAddOp);
                op.erase();
                reluSourceOp->erase();
                return success();
            }

            return failure();
        };

    };

    class AtirBlockFusionPass : public AtirBlockFusionBase<AtirBlockFusionPass> {
    public:
        AtirBlockFusionPass() = default;

        void runOnOperation() override {
            ANNC_LOG_DEBUG("block-fusion") << "this is AtirBlockFusionPass\n";

            auto m = getOperation();
            auto ctx = m.getContext();
            GreedyRewriteConfig config;
            config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);

            RewritePatternSet patterns(ctx);
            patterns.add<FuseReluRewrite>(ctx);
            (void)applyPatternsGreedily(m, std::move(patterns), config);
            ANNC_LOG_DEBUG("block-fusion") << "Block fusion analysis completed\n";

        }
    };

    std::unique_ptr<OperationPass<ModuleOp>> createAtirBlockFusionPass() {
        ANNC_LOG_DEBUG("block-fusion") << "this is createAtirBlockFusionPass\n";
        return std::make_unique<AtirBlockFusionPass>();
    }
}  // namespace atir
