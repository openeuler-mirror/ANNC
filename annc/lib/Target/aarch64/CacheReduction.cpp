#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Dialect/Atir/AtirTypes.h.inc"
#include "Dialect/Atir/AtirAttr.h.inc"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"

#include "Target/aarch64/Passes.h"

namespace annc
{
struct CacheReductionMatmulTilingPattern : public OpRewritePattern<linalg::MatmulOp> {
  CacheReductionMatmulTilingPattern(MLIRContext* context, PatternBenefit benefit = 9)
      : OpRewritePattern<linalg::MatmulOp>(context, benefit) {}
 public:
  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter& rewriter) const override {
    llvm::dbgs() << "this is CacheReductionMatmulTilingPattern \n";
    constexpr StringLiteral kConfigAttrName = "lowering_config";
    auto attr = op->getAttr(kConfigAttrName);
    attr.dump();
    return failure();
  }
};

class CacheReduction : public CacheReductionBase<CacheReduction>
{
  using Base::Base;

  void runOnOperation() override
  {
    llvm::dbgs() << "this is CacheReduction\n";

    ModuleOp module = getOperation();
    auto ctx = module->getContext();
    module.walk([&](linalg::MatmulOp matmulOp) {
      constexpr StringLiteral kConfigAttrName = "lowering_config";
      auto attr = matmulOp->getAttr(kConfigAttrName);
      auto lowering_config_attr = llvm::dyn_cast<atir::LoweringConfigAttr>(attr);

      OpBuilder builder(matmulOp);
      Location loc = matmulOp.getLoc();

      Value A = matmulOp.getInputs()[0];
      Value B = matmulOp.getInputs()[1];
      Value C = matmulOp.getOutputs()[0];

      auto aType = llvm::dyn_cast<RankedTensorType>(A.getType());
      auto bType = llvm::dyn_cast<RankedTensorType>(B.getType());
      auto cType = llvm::dyn_cast<RankedTensorType>(C.getType());

      int64_t M = aType.getShape()[0];
      int64_t K = aType.getShape()[1];
      int64_t N = bType.getShape()[1];

      std::vector<int64_t> cache_reduction_config  = lowering_config_attr.getLoweringConfigCacheReduction();

      int64_t tileK = cache_reduction_config[2];

      Value accInit = C;

      Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value K_value = builder.create<arith::ConstantIndexOp>(loc, K);
      Value tileK_value = builder.create<arith::ConstantIndexOp>(loc, tileK);

      auto a_pack = builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{M, tileK}, aType.getElementType());

      auto b_pack = builder.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{tileK, N}, bType.getElementType());

      auto forOp = builder.create<scf::ForOp>(loc, zero, K_value, tileK_value, ValueRange{accInit});

      builder.setInsertionPointToStart(forOp.getBody());


      Value ivK = forOp.getInductionVar();
      Value accIter = forOp.getRegionIterArgs()[0];

      SmallVector<OpFoldResult> apack_offsets;
      apack_offsets.push_back(builder.getIndexAttr(0));
      apack_offsets.push_back(ivK);
      SmallVector<OpFoldResult> apack_sizes;
      apack_sizes.push_back(builder.getIndexAttr(M));
      apack_sizes.push_back(builder.getIndexAttr(tileK));
      SmallVector<OpFoldResult> apack_stridess;
      apack_stridess.push_back(builder.getIndexAttr(1));
      apack_stridess.push_back(builder.getIndexAttr(1));

      // A_k = A[:, k:k+tileK]
      Value Ak = builder.create<tensor::ExtractSliceOp>(
          loc, A,
          apack_offsets,
          apack_sizes,
          apack_stridess);

      SmallVector<OpFoldResult> bpack_offsets;
      bpack_offsets.push_back(ivK);
      bpack_offsets.push_back(builder.getIndexAttr(0));
      SmallVector<OpFoldResult> bpack_sizes;
      bpack_sizes.push_back(builder.getIndexAttr(tileK));
      bpack_sizes.push_back(builder.getIndexAttr(N));
      SmallVector<OpFoldResult> bpack_stridess;
      bpack_stridess.push_back(builder.getIndexAttr(1));
      bpack_stridess.push_back(builder.getIndexAttr(1));

      Value Bk = builder.create<tensor::ExtractSliceOp>(
          loc, B,
          bpack_offsets,
          bpack_sizes,
          bpack_stridess);

      builder.create<linalg::CopyOp>(
          loc,
          ValueRange{Ak},
          ValueRange{a_pack}
      );

      builder.create<linalg::CopyOp>(
          loc,
          ValueRange{Bk},
          ValueRange{b_pack}
      );
      auto cacheReduceMatmul = builder.create<linalg::MatmulOp>(
          loc, ValueRange{a_pack, b_pack}, ValueRange{accIter});
      Value newAcc = cacheReduceMatmul.getResult(0);

      cacheReduceMatmul->setAttr(kConfigAttrName,
                                 matmulOp->getAttr(kConfigAttrName));

      builder.create<scf::YieldOp>(loc, newAcc);

      matmulOp->getParentOp()->dump();

      RewritePatternSet patterns(ctx);


      builder.setInsertionPointAfter(forOp);
      matmulOp.replaceAllUsesWith(forOp);
      matmulOp.erase();
    });
  }


};

std::unique_ptr<mlir::Pass> createCacheReduction()
{
  llvm::dbgs() << "this is createCacheReduction\n";
  return std::make_unique<CacheReduction>();
}
}