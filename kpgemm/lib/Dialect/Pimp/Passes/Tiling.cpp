#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Pimp/PimpOps.h"
#include "Dialect/Pimp/Passes/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "Helper.h"
#include "iostream"

using namespace llvm;
using namespace mlir;

namespace pimp {
struct TilingMatMulRewrite : public OpRewritePattern<MatMulOp> {
    TilingMatMulRewrite(MLIRContext* context, PatternBenefit benefit = 9)
            : OpRewritePattern<MatMulOp>(context, benefit) {}

public:
    LogicalResult matchAndRewrite(MatMulOp op,
                                  PatternRewriter& rewriter) const override {

        //
        std::cout << "this is PimpTilingPass MatMulRewrite" << std::endl;

        //todo tilingmatch
        auto lhsTilingAttr = op.getLhs().getType().getOnchipParallel();
        auto rhsTilingAttr = op.getRhs().getType().getOnchipParallel();

        if (lhsTilingAttr != nullptr && rhsTilingAttr != nullptr) {
            return failure();
        }

        //shape m,n,k
        TensorType leftType = op.getLhs().getType();
        TensorType rightType = op.getRhs().getType();

        std::vector<int64_t> lhsShape = leftType.getValueOfShape();
        std::vector<int64_t> rhsShape = rightType.getValueOfShape();

        int M = lhsShape[0];
        int N = rhsShape[1];
        int K = lhsShape[1];

        //todo interThread intraThread

        //todo tiling
        //jit
        int BLOCK_M = 384;
        int BLOCK_K = 64 * 3;
        int BLOCK_N = 96 * 3;

//        int BLOCK_M = 5;
//        int BLOCK_K = 4;
//        int BLOCK_N = 3;

        //todo tilingTensor

        SmallVector<Attribute> lmStart;
        SmallVector<Attribute> lmSize;

        for (int start = 0; start < M; ) {
            lmStart.push_back(rewriter.getI64IntegerAttr(start));
            lmSize.push_back(rewriter.getI64IntegerAttr((BLOCK_M < (M - start)) ? BLOCK_M : (M - start)));
            start += BLOCK_M;
        }

        SmallVector<Attribute> lkStart;
        SmallVector<Attribute> lkSize;

        for (int start = 0; start < K; ) {
            lkStart.push_back(rewriter.getI64IntegerAttr(start));
            lkSize.push_back(rewriter.getI64IntegerAttr((BLOCK_K < (K - start)) ? BLOCK_K : (K - start)));
            start += BLOCK_K;
        }

        SmallVector<Attribute> rnStart;
        SmallVector<Attribute> rnSize;

        for (int start = 0; start < N; ) {
            rnStart.push_back(rewriter.getI64IntegerAttr(start));
            rnSize.push_back(rewriter.getI64IntegerAttr((BLOCK_N < (N - start)) ? BLOCK_N : (N - start)));
            start += BLOCK_N;
        }

        SmallVector<Attribute> axesVector;

        axesVector.push_back(rewriter.getI64IntegerAttr(0));
        axesVector.push_back(rewriter.getI64IntegerAttr(1));
        auto lAxes = ArrayAttr::get(getContext(),axesVector);

        auto lStart = ArrayAttr::get(getContext(), {
                ArrayAttr::get(getContext(), lmStart),
                ArrayAttr::get(getContext(), lkStart)
        });
        auto lSize = ArrayAttr::get(getContext(), {
                ArrayAttr::get(getContext(), lmSize),
                ArrayAttr::get(getContext(), lkSize)
        });

        auto rAxes = ArrayAttr::get(getContext(),axesVector);

        auto rStart = ArrayAttr::get(getContext(), {
                ArrayAttr::get(getContext(), lkStart),
                ArrayAttr::get(getContext(), rnStart)
        });

        auto rSize = ArrayAttr::get(getContext(), {
                ArrayAttr::get(getContext(), lkSize),
                ArrayAttr::get(getContext(), rnSize)
        });

        auto lTilingAttr = TilingAttr::get(getContext(), lAxes, lStart, lSize);
        auto rTilingAttr = TilingAttr::get(getContext(), rAxes, rStart, rSize);

//        leftType.cloneWithOnchipTiling(tilingAttr);
//        rightType.cloneWithOnchipTiling(tilingAttr);

//        op.getLhsMutable().

        leftType.setOnchipParallel(lTilingAttr);
        rightType.setOnchipParallel(rTilingAttr);

        return success();
    };

};

class PimpTilingPass : public PimpTilingBase<PimpTilingPass> {
public:
  PimpTilingPass() = default;

  void runOnOperation() override {
      std::cout << "this is PimpTilingPass" << std::endl;
      auto m = getOperation();
      auto ctx = m.getContext();
//      m.dump();
      GreedyRewriteConfig config;
      config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);

      RewritePatternSet patterns(ctx);
      patterns.add<TilingMatMulRewrite>(ctx);
      (void)applyPatternsGreedily(m, std::move(patterns), config);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createPimpTilingPass() {
    std::cout << "this is createPimpTilingPass" << std::endl;
  return std::make_unique<PimpTilingPass>();
}
}  // namespace pimp