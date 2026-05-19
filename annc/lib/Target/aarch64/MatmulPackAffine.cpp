#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Dialect/Atir/AtirTypes.h.inc"
#include "Dialect/Atir/AtirAttr.h.inc"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"

#include "Target/aarch64/Passes.h"

namespace annc {
class MatmulPackAffine : public MatmulPackAffineBase<MatmulPackAffine> {
  using Base::Base;

  void runOnOperation() override {
    llvm::dbgs() << "this is MatmulPackAffine\n";
    ModuleOp module = getOperation();
    module.walk([&](linalg::MatmulOp op) {
      if (failed(matmulPackAffine(op))) {
        signalPassFailure();
      }
    });
  }

  LogicalResult matmulPackAffine(linalg::MatmulOp matmulOp) {

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

    std::vector<int64_t> cache_reduction_config = lowering_config_attr.getLoweringConfigCacheReduction();
    int64_t blockK = cache_reduction_config[2];

    std::vector<int64_t > vector_common_parallel_config = lowering_config_attr.getLoweringConfigVectorCommonParallel();
    int64_t kernelM = vector_common_parallel_config[0];
    int64_t kernelN = vector_common_parallel_config[1];

    std::vector<int64_t> vector_reduction_config = lowering_config_attr.getLoweringConfigVectorReduction();
    int64_t kernelK = vector_reduction_config[2];

    if (blockM == 0 || blockN == 0) {
      return success();
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

    Value M_Value = builder.create<arith::ConstantIndexOp>(loc, M);
    Value N_value = builder.create<arith::ConstantIndexOp>(loc, N);

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

    //mn for
    auto blockMNParallelOp = builder.create<affine::AffineParallelOp>(
        loc,
        TypeRange{},
        ArrayRef<arith::AtomicRMWKind>{},
        lbMaps, lbVals,
        ubMaps, ubVals,
        steps);

    Block *body = blockMNParallelOp.getBody();
    builder.setInsertionPointToStart(body);

    Value i = body->getArgument(0); // %i
    Value j = body->getArgument(1); // %j

    auto C_tile = builder.create<memref::SubViewOp>(
        loc, C,
        SmallVector<OpFoldResult>{i, j},
        SmallVector<OpFoldResult>{
            builder.getIndexAttr(blockM),
            builder.getIndexAttr(blockN)},
        SmallVector<OpFoldResult>{
            builder.getIndexAttr(1),
            builder.getIndexAttr(1)});

    //pack alloca
    Value A_pack = builder.create<memref::AllocOp>(
        loc,MemRefType::get(blockM * blockK, aType.getElementType()));
    Value B_pack = builder.create<memref::AllocOp>(
        loc, MemRefType::get(blockK * blockN, bType.getElementType()));

    //k for
    auto blockKForOp = builder.create<affine::AffineForOp>(loc, 0, K, blockK);

    builder.setInsertionPointToStart(blockKForOp.getBody());
    Value k = blockKForOp.getInductionVar();

    auto A_tile = builder.create<memref::SubViewOp>(
        loc, A,
        SmallVector<OpFoldResult>{i, k},
        SmallVector<OpFoldResult>{
            builder.getIndexAttr(blockM),
            builder.getIndexAttr(blockK)},
        SmallVector<OpFoldResult>{
            builder.getIndexAttr(1),
            builder.getIndexAttr(1)});

    auto B_tile = builder.create<memref::SubViewOp>(
        loc, B,
        SmallVector<OpFoldResult>{k, j},
        SmallVector<OpFoldResult>{
            builder.getIndexAttr(blockK),
            builder.getIndexAttr(blockN)},
        SmallVector<OpFoldResult>{
            builder.getIndexAttr(1),
            builder.getIndexAttr(1)});

    //Pack A
    auto mb = builder.create<affine::AffineForOp>(loc, 0, blockM/kernelM);
    builder.setInsertionPointToStart(mb.getBody());
    auto kb = builder.create<affine::AffineForOp>(loc, 0, blockK);
    builder.setInsertionPointToStart(kb.getBody());
    auto mi = builder.create<affine::AffineForOp>(loc, 0, kernelM);
    builder.setInsertionPointToStart(mi.getBody());

    Value mIdx = builder.create<affine::AffineApplyOp>(
        loc,
        AffineMap::get(2, 0,
                       builder.getAffineDimExpr(0) * kernelM +
                           builder.getAffineDimExpr(1)),
        ValueRange{mb.getInductionVar(), mi.getInductionVar()});

    Value loadA = builder.create<memref::LoadOp>(
        loc, A_tile, ValueRange{mIdx, kb.getInductionVar()});

    Value aPackIdx = builder.create<affine::AffineApplyOp>(
        loc,
        AffineMap::get(3, 0,
                       (builder.getAffineDimExpr(0) * blockK +
                        builder.getAffineDimExpr(1)) * kernelM +
                           builder.getAffineDimExpr(2)),
        ValueRange{
            mb.getInductionVar(),
            kb.getInductionVar(),
            mi.getInductionVar()});

    builder.create<memref::StoreOp>(loc, loadA, A_pack, ValueRange{aPackIdx});

    //Pack B
    builder.setInsertionPointAfter(mb);
    auto nb = builder.create<affine::AffineForOp>(loc, 0, blockN / kernelN);
    builder.setInsertionPointToStart(nb.getBody());
    auto kb2 = builder.create<affine::AffineForOp>(loc, 0, blockK);
    builder.setInsertionPointToStart(kb2.getBody());
    auto ni = builder.create<affine::AffineForOp>(loc, 0, kernelN);
    builder.setInsertionPointToStart(ni.getBody());

    Value nIdx = builder.create<affine::AffineApplyOp>(
        loc,
        AffineMap::get(2, 0,
                       builder.getAffineDimExpr(0) * kernelN +
                           builder.getAffineDimExpr(1)),
        ValueRange{nb.getInductionVar(), ni.getInductionVar()});

    Value loadB = builder.create<memref::LoadOp>(
        loc, B_tile, ValueRange{kb2.getInductionVar(), nIdx});

    Value bPackIdx = builder.create<affine::AffineApplyOp>(
        loc,
        AffineMap::get(3, 0,
                       (builder.getAffineDimExpr(0) * blockK +
                        builder.getAffineDimExpr(1)) * kernelN +
                           builder.getAffineDimExpr(2)),
        ValueRange{
            nb.getInductionVar(),
            kb2.getInductionVar(),
            ni.getInductionVar()});

    builder.create<memref::StoreOp>(loc, loadB, B_pack, ValueRange{bPackIdx});

    //reinterpret cast pack
    builder.setInsertionPointAfter(nb);
    SmallVector<int64_t> A3_sizes = {blockM / kernelM, blockK, kernelM};
    SmallVector<int64_t> A3_strides = {kernelM * blockK, kernelM, 1};
    Value A3 = builder.create<memref::ReinterpretCastOp>(
        loc, MemRefType::get({blockM / kernelM, blockK, kernelM}, aType.getElementType()),
        A_pack,
        0,
        A3_sizes,
        A3_strides);

    SmallVector<int64_t> B3_sizes = {blockN / kernelN, blockK, kernelN};
    SmallVector<int64_t> B3_strides = {kernelN * blockK, kernelN, 1};
    Value B3 = builder.create<memref::ReinterpretCastOp>(
        loc, MemRefType::get({blockN / kernelN, blockK, kernelN}, bType.getElementType()),
        B_pack,
        0,
        B3_sizes,
        B3_strides);

    //kernel
    auto mker = builder.create<affine::AffineForOp>(loc, 0, blockM, kernelM);
    builder.setInsertionPointToStart(mker.getBody());
    auto nker = builder.create<affine::AffineForOp>(loc, 0, blockN, kernelN);
    builder.setInsertionPointToStart(nker.getBody());
    auto kker = builder.create<affine::AffineForOp>(loc, 0, blockK, kernelK);
    builder.setInsertionPointToStart(kker.getBody());

    //a subview
    Value mbIdx = builder.create<affine::AffineApplyOp>(
        loc, AffineMap::get(
                 /*dimCount=*/1, /*symbolCount=*/0,
                 builder.getAffineDimExpr(0).floorDiv(kernelM)),
        ValueRange{mker.getInductionVar()});
    SmallVector<OpFoldResult> a_kernel_offsets = {
        mbIdx,
        kker.getInductionVar(),
        builder.getIndexAttr(0)
    };
    SmallVector<OpFoldResult> a_kernel_sizes = {
        builder.getIndexAttr(1),
        builder.getIndexAttr(kernelK),
        builder.getIndexAttr(kernelM)
    };
    SmallVector<OpFoldResult> a_kernel_strides = {
        builder.getIndexAttr(1),
        builder.getIndexAttr(1),
        builder.getIndexAttr(1)
    };

    Value A_kernel = builder.create<memref::SubViewOp>(
        loc,
        A3,
        a_kernel_offsets,
        a_kernel_sizes,
        a_kernel_strides);

    //b subview
    Value nbIdx = builder.create<affine::AffineApplyOp>(
        loc, AffineMap::get(
                 /*dimCount=*/1, /*symbolCount=*/0,
                 builder.getAffineDimExpr(0).floorDiv(kernelN)),
        ValueRange{nker.getInductionVar()});
    SmallVector<OpFoldResult> b_kernel_offsets = {
        nbIdx,
        kker.getInductionVar(),
        builder.getIndexAttr(0)
    };
    SmallVector<OpFoldResult> b_kernel_sizes = {
        builder.getIndexAttr(1),
        builder.getIndexAttr(kernelK),
        builder.getIndexAttr(kernelN)
    };
    SmallVector<OpFoldResult> b_kernel_strides = {
        builder.getIndexAttr(1),
        builder.getIndexAttr(1),
        builder.getIndexAttr(1)
    };
    Value B_kernel = builder.create<memref::SubViewOp>(
        loc, B3,
        b_kernel_offsets,
        b_kernel_sizes,
        b_kernel_strides);

    // C subview
    auto C_kernel = builder.create<memref::SubViewOp>(
        loc, C_tile,
        SmallVector<OpFoldResult>{mker.getInductionVar(), nker.getInductionVar()},
        SmallVector<OpFoldResult>{
            builder.getIndexAttr(kernelM),
            builder.getIndexAttr(kernelN)},
        SmallVector<OpFoldResult>{
            builder.getIndexAttr(1),
            builder.getIndexAttr(1)});

    // linalg.matmul with custom indexing_maps
    SmallVector<AffineMap> maps = {
        AffineMap::get(3, 0,
                       {builder.getAffineDimExpr(2),
                        builder.getAffineDimExpr(0)}, builder.getContext()), // A(k,m)
        AffineMap::get(3, 0,
                       {builder.getAffineDimExpr(2),
                        builder.getAffineDimExpr(1)}, builder.getContext()), // B(k,n)
        AffineMap::get(3, 0,
                       {builder.getAffineDimExpr(0),
                        builder.getAffineDimExpr(1)}, builder.getContext())  // C(m,n)
    };



    auto A_kernel_2D = builder.create<memref::CollapseShapeOp>(loc, A_kernel, ArrayRef<ReassociationIndices>{{0, 1}, {2}});
    auto B_kernel_2D = builder.create<memref::CollapseShapeOp>(loc, B_kernel, ArrayRef<ReassociationIndices>{{0, 1}, {2}});

    auto new_matmul = builder.create<linalg::MatmulOp>(
        loc,
        ValueRange{A_kernel_2D, B_kernel_2D},
        ValueRange{C_kernel});

    new_matmul->setAttr("indexing_maps", builder.getAffineMapArrayAttr(maps));
    matmulOp.erase();

    return success();
  }
};

std::unique_ptr<mlir::Pass> createMatmulPackAffine() {
  llvm::dbgs() << "this is createMatmulPackAffine\n";
  return std::make_unique<MatmulPackAffine>();
}
} // namespace annc
