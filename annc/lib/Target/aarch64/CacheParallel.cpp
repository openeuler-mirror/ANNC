#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Dialect/Atir/AtirTypes.h.inc"
#include "Dialect/Atir/AtirAttr.h.inc"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"

#include "Target/aarch64/Passes.h"

namespace annc
{
struct CacheParallelMatmulTilingPattern : public OpRewritePattern<linalg::MatmulOp> {
  CacheParallelMatmulTilingPattern(MLIRContext* context, PatternBenefit benefit = 9)
      : OpRewritePattern<linalg::MatmulOp>(context, benefit) {}
 public:
  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter& rewriter) const override {
    llvm::dbgs() << "this is CacheParallelMatmulTilingPattern \n";
    constexpr StringLiteral kConfigAttrName = "lowering_config";
    auto attr = matmulOp->getAttr(kConfigAttrName);
    attr.dump();
    return failure();
  }
};

class CacheParallel : public CacheParallelBase<CacheParallel>
{
  using Base::Base;

  void runOnOperation() override
  {
    llvm::dbgs() << "this is CacheParallel\n";
    ModuleOp module = getOperation();
    module.walk([&](linalg::MatmulOp op) {
      //                if (failed(matmulCacheParallel(op))) {
      //                    signalPassFailure();
      //                };

      if (failed(matmulCacheParallelAffine(op))) {
        signalPassFailure();
      }

    });
  }

  LogicalResult matmulCacheParallel(linalg::MatmulOp matmul) {
    constexpr StringLiteral kConfigAttrName = "lowering_config";
    auto loc = matmul.getLoc();
    OpBuilder builder(matmul);

    auto attr = matmul->getAttr(kConfigAttrName);
    if (!attr) {
      return success();
    }
    auto lowering_config_attr = llvm::dyn_cast<atir::LoweringConfigAttr>(attr);

    std::vector<int64_t> cache_parallel_config = lowering_config_attr.getLoweringConfigCacheParallel();

    int64_t tileM = cache_parallel_config[0];
    int64_t tileN = cache_parallel_config[1];

    if (tileM == 0 || tileN == 0) {
      return success();
    }

    Value A = matmul.getInputs()[0];
    Value B = matmul.getInputs()[1];
    Value C = matmul.getOutputs()[0];

    auto aType = llvm::dyn_cast<RankedTensorType>(A.getType());
    auto bType = llvm::dyn_cast<RankedTensorType>(B.getType());
    auto cType = llvm::dyn_cast<RankedTensorType>(C.getType());

    int64_t M = aType.getShape()[0];
    int64_t K = aType.getShape()[1];
    int64_t N = bType.getShape()[1];

    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value M_value = builder.create<arith::ConstantIndexOp>(loc, M);
    Value N_value = builder.create<arith::ConstantIndexOp>(loc, N);
    Value K_value = builder.create<arith::ConstantIndexOp>(loc, K);
    Value tileM_value  = builder.create<arith::ConstantIndexOp>(loc, tileM);
    Value tileN_value  = builder.create<arith::ConstantIndexOp>(loc, tileN);


    auto outer = builder.create<scf::ParallelOp>(
        loc,
        ValueRange{zero},
        ValueRange{M_value},
        ValueRange{tileM_value});

    builder.setInsertionPointToStart(outer.getBody());

    Value ivM = outer.getInductionVars()[0];

    auto inner = builder.create<scf::ParallelOp>(
        loc,
        ValueRange{zero},
        ValueRange{N_value},
        ValueRange{tileN_value});

    builder.setInsertionPointToStart(inner.getBody());
    Value ivN = inner.getInductionVars()[0];

    SmallVector<OpFoldResult> a_offsets;
    a_offsets.push_back(ivM);
    a_offsets.push_back(builder.getIndexAttr(0));

    SmallVector<OpFoldResult> a_sizes;
    a_sizes.push_back(builder.getIndexAttr(tileM));
    a_sizes.push_back(builder.getIndexAttr(K));

    SmallVector<OpFoldResult> a_strides;
    a_strides.push_back(builder.getIndexAttr(1));
    a_strides.push_back(builder.getIndexAttr(1));

    auto sliceA = builder.create<tensor::ExtractSliceOp>(
        loc, A,
        a_offsets,
        a_sizes,
        a_strides);

    SmallVector<OpFoldResult> b_offsets;
    b_offsets.push_back(builder.getIndexAttr(0));
    b_offsets.push_back(ivN);

    SmallVector<OpFoldResult> b_sizes;
    b_sizes.push_back(builder.getIndexAttr(K));
    b_sizes.push_back(builder.getIndexAttr(tileN));

    SmallVector<OpFoldResult> b_strides;
    b_strides.push_back(builder.getIndexAttr(1));
    b_strides.push_back(builder.getIndexAttr(1));


    auto sliceB = builder.create<tensor::ExtractSliceOp>(
        loc, B,
        b_offsets,
        b_sizes,
        b_strides);

    SmallVector<OpFoldResult> c_offsets;
    c_offsets.push_back(ivM);
    c_offsets.push_back(ivN);

    SmallVector<OpFoldResult> c_sizes;
    c_sizes.push_back(builder.getIndexAttr(tileM));
    c_sizes.push_back(builder.getIndexAttr(tileN));

    SmallVector<OpFoldResult> c_strides;
    c_strides.push_back(builder.getIndexAttr(1));
    c_strides.push_back(builder.getIndexAttr(1));

    auto sliceC = builder.create<tensor::ExtractSliceOp>(
        loc, C,
        c_offsets,
        c_sizes,
        c_strides);

    auto tiledMatmul = builder.create<linalg::MatmulOp>(
        loc,
        TypeRange{sliceC.getType()},
        ValueRange{sliceA, sliceB},
        ValueRange{sliceC});

    //  lowering_config cache_reduction / vector pass 
    tiledMatmul->setAttr("lowering_config",
                         matmul->getAttr("lowering_config"));
    matmul.erase();
    return success();
  }

  LogicalResult matmulCacheParallelAffine(linalg::MatmulOp matmulOp) {
    Location loc = matmulOp->getLoc();
    OpBuilder builder(matmulOp);

    constexpr StringLiteral kConfigAttrName = "lowering_config";
    auto attr = matmulOp->getAttr(kConfigAttrName);

    if (!attr) {
      return success();
    }

    auto lowering_config_attr = llvm::dyn_cast<atir::LoweringConfigAttr>(attr);

    std::vector<int64_t> cache_parallel_config = lowering_config_attr.getLoweringConfigCacheParallel();

    int64_t blockM = cache_parallel_config[0];
    int64_t blockN = cache_parallel_config[1];

    if (blockM == 0 || blockN == 0) {
      return failure();
    }

    Value A = matmulOp.getInputs()[0];
    Value B = matmulOp.getInputs()[1];
    Value C = matmulOp.getOutputs()[0];

    auto aType = llvm::dyn_cast<MemRefType>(A.getType());
    auto bType = llvm::dyn_cast<MemRefType>(B.getType());
    auto cType = llvm::dyn_cast<MemRefType>(C.getType());

    int64_t M = aType.getShape()[0];
    int64_t K = aType.getShape()[1];
    int64_t N = bType.getShape()[1];

    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value M_Value = builder.create<arith::ConstantIndexOp>(loc, M);
    Value K_Value = builder.create<arith::ConstantIndexOp>(loc, K);
    Value N_value = builder.create<arith::ConstantIndexOp>(loc, N);
    Value blockM_value = builder.create<arith::ConstantIndexOp>(loc, blockM);
    Value blockN_value = builder.create<arith::ConstantIndexOp>(loc, blockN);

    auto lbMap = AffineMap::get(
        /*dims=*/0,
        /*symbols=*/0,
        builder.getAffineConstantExpr(0),
        builder.getContext());

    SmallVector<AffineMap, 2> lbMaps{lbMap, lbMap};
    SmallVector<Value, 0> lbVals;

    SmallVector<AffineMap, 2> ubMaps = {
        AffineMap::get(0, 2, builder.getAffineSymbolExpr(0)),
        AffineMap::get(0, 2, builder.getAffineSymbolExpr(1))
    };
    SmallVector<Value, 2> ubVals{M_Value, N_value};

    SmallVector<int64_t, 2> steps = {blockM, blockN};

    auto parallelOp = builder.create<affine::AffineParallelOp>(
        loc,
        TypeRange{},
        ArrayRef<arith::AtomicRMWKind>{},
        lbMaps, lbVals,
        ubMaps, ubVals,
        steps);

    Block *body = parallelOp.getBody();
    builder.setInsertionPointToStart(body);

    Value i = body->getArgument(0); // %i
    Value j = body->getArgument(1); // %j

    SmallVector<OpFoldResult> a_offsets = {
        i,
        builder.getIndexAttr(0)
    };
    SmallVector<OpFoldResult> a_sizes = {
        builder.getIndexAttr(blockM),
        builder.getIndexAttr(K)
    };
    SmallVector<OpFoldResult> a_strides = {
        builder.getIndexAttr(1),
        builder.getIndexAttr(1)
    };

    Value Asub = builder.create<memref::SubViewOp>(
        loc,
        A,
        a_offsets,
        a_sizes,
        a_strides);

    SmallVector<OpFoldResult> b_offsets = {
        builder.getIndexAttr(0),
        j
    };
    SmallVector<OpFoldResult> b_sizes = {
        builder.getIndexAttr(K),
        builder.getIndexAttr(blockN)
    };
    SmallVector<OpFoldResult> b_strides = {
        builder.getIndexAttr(1),
        builder.getIndexAttr(1)
    };

    Value Bsub = builder.create<memref::SubViewOp>(
        loc,
        B,
        b_offsets,
        b_sizes,
        b_strides);

    SmallVector<OpFoldResult> c_offsets = {
        i,
        j
    };
    SmallVector<OpFoldResult> c_sizes = {
        builder.getIndexAttr(blockM),
        builder.getIndexAttr(blockN)
    };
    SmallVector<OpFoldResult> c_strides = {
        builder.getIndexAttr(1),
        builder.getIndexAttr(1)
    };
    Value Csub = builder.create<memref::SubViewOp>(
        loc,
        C,
        c_offsets,
        c_sizes,
        c_strides);

    auto new_matmul = builder.create<linalg::MatmulOp>(
        loc,
        ValueRange{Asub, Bsub},
        ValueRange{Csub});
    new_matmul->setAttr(kConfigAttrName, attr);

    matmulOp.erase();
    llvm::dbgs() << "start dump parallelOp\n";
    parallelOp->getParentOp()->dump();

    return success();
  }

};

std::unique_ptr<mlir::Pass> createCacheParallel()
{
  llvm::dbgs() << "this is createCacheParallel\n";
  return std::make_unique<CacheParallel>();
}
}