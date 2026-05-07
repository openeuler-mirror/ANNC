#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/AtirTypes.h.inc"
#include "Dialect/Atir/Passes/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "Helper.h"
#include "iostream"

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
            std::cout << "this is FuseReluRewrite" << std::endl;
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

    //matmul + add => matmul
    struct MatmulWithBiasRewrite : public OpRewritePattern<MatMulOp> {
        MatmulWithBiasRewrite(MLIRContext* context, PatternBenefit benefit = 9)
                : OpRewritePattern<MatMulOp>(context, benefit) {}

    public:
        LogicalResult matchAndRewrite(MatMulOp op,
                                      PatternRewriter& rewriter) const override {

            std::cout << "this is MatmulWithBiasRewrite" << std::endl;
            if (op.getBias() != nullptr) {
                return failure();
            }
            CHECK_LOGICAL_SUCCESS(op.getResult().getNumUses() == 1)
            for (auto* user: op.getResult().getUsers()) {
                if (auto addOp = llvm::dyn_cast<atir::AddOp>(user)) {
                    //biasmatmulOp
                    CHECK_LOGICAL_SUCCESS(addOp->getOperands().size() == 2);
                    //
                    std::vector<NamedAttribute> attrs;
                    attrs.push_back(NamedAttribute(rewriter.getStringAttr("withBias"),
                                                   rewriter.getBoolAttr(true)));
                    attrs.push_back(NamedAttribute(rewriter.getStringAttr("right_transpose"),
                                                   rewriter.getBoolAttr(op.getRightTranspose())));

                    attrs.push_back(NamedAttribute(rewriter.getStringAttr("left_transpose"),
                                                   rewriter.getBoolAttr(op.getLeftTranspose())));

                    attrs.push_back(NamedAttribute(rewriter.getStringAttr("output_transpose"),
                                                   rewriter.getBoolAttr(op.getOutputTranspose())));

                    auto matmul_do_relu = op.getDoRelu();
                    auto matmul_relu_limit = op.getReluLimit().convertToFloat();
                    auto add_do_relu = addOp.getDoRelu();
                    auto add_relu_limit = addOp.getReluLimit().convertToFloat();

                    float relu_limit = matmul_relu_limit;
                    if (matmul_do_relu && matmul_relu_limit > relu_limit) {
                        relu_limit = matmul_relu_limit;
                    }
                    if (add_do_relu && add_relu_limit > relu_limit) {
                        relu_limit = add_relu_limit;
                    }
                    if (matmul_do_relu || add_do_relu) {
                        attrs.push_back(NamedAttribute(rewriter.getStringAttr("do_relu"),
                                                       rewriter.getBoolAttr(true)));
                        attrs.push_back(NamedAttribute(rewriter.getStringAttr("relu_limit"),
                                                       rewriter.getF32FloatAttr(relu_limit)));
                    }
                    //ins
                    std::vector<Value> ins;
                    ins.push_back(op.getLhs());
                    ins.push_back(op.getRhs());
                    ins.push_back(op.getC());
                    ins.push_back(addOp.getOperand(1));
                    //outs
                    std::vector<Type> outs;
                    outs.push_back(addOp.getResult().getType());
                    if (addOp.getOperand(1).getDefiningOp() != nullptr
                        && llvm::isa<atir::ConstantOp>(addOp.getOperand(1).getDefiningOp())) {
                        addOp.getOperand(1).getDefiningOp()->moveBefore(op);
                    }

                    auto newMatmulOp = rewriter.create<atir::MatMulOp>(addOp.getLoc(), outs,
                                                    ins,attrs);
                    addOp.getResult().replaceAllUsesWith(newMatmulOp);
                    addOp.erase();
                    op.erase();
                }
            }
            return failure();
        };

    };

    class AtirBlockFusionPass : public AtirBlockFusionBase<AtirBlockFusionPass> {
    public:
        AtirBlockFusionPass() = default;

        void runOnOperation() override {
            std::cout << "this is AtirBlockFusionPass" << std::endl;

            auto m = getOperation();
            auto ctx = m.getContext();
            GreedyRewriteConfig config;
            config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);

            RewritePatternSet patterns(ctx);
            patterns.add<FuseReluRewrite>(ctx);
            patterns.add<MatmulWithBiasRewrite>(ctx);
            (void)applyPatternsGreedily(m, std::move(patterns), config);
            std::cout << "Block fusion analysis completed" << std::endl;

        }
    };

    std::unique_ptr<OperationPass<ModuleOp>> createAtirBlockFusionPass() {
        std::cout << "this is createAtirBlockFusionPass" << std::endl;
        return std::make_unique<AtirBlockFusionPass>();
    }
}  // namespace atir