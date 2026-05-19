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
class VectorReduction : public VectorReductionBase<VectorReduction>
{
  using Base::Base;

  void runOnOperation() override
  {
    llvm::dbgs() << "this is VectorReduction\n";

    ModuleOp module = getOperation();
    auto ctx = module->getContext();
    module.walk([&](linalg::MatmulOp matmulOp) {
      //                if (failed(matmulVectorReduction(matmulOp))) {
      //                    signalPassFailure();
      //                }
      auto res = matmulVectorReductionAffine(matmulOp);
      if (failed(res)) {
        signalPassFailure();
      }

    });
  }

  LogicalResult matmulVectorReduction(linalg::MatmulOp matmulOp) {
    llvm::dbgs() << "VectorReduction 1\n";
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

    std::vector<int64_t> vector_reduction  = lowering_config_attr.getLoweringConfigVectorReduction();

    int64_t vector_tile = vector_reduction[2];

    Value accInit = C;

    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value K_value = builder.create<arith::ConstantIndexOp>(loc, K);
    Value vector_tile_value = builder.create<arith::ConstantIndexOp>(loc, vector_tile);

    auto forOp = builder.create<scf::ForOp>(loc, zero, K_value, vector_tile_value, ValueRange{accInit});

    builder.setInsertionPointToStart(forOp.getBody());

    Value ivK = forOp.getInductionVar();
    Value accIter = forOp.getRegionIterArgs()[0];

    SmallVector<OpFoldResult> akernel_offsets;
    akernel_offsets.push_back(builder.getIndexAttr(0));
    akernel_offsets.push_back(ivK);
    SmallVector<OpFoldResult> akernel_sizes;
    akernel_sizes.push_back(builder.getIndexAttr(M));
    akernel_sizes.push_back(builder.getIndexAttr(vector_tile));
    SmallVector<OpFoldResult> akernel_stridess;
    akernel_stridess.push_back(builder.getIndexAttr(1));
    akernel_stridess.push_back(builder.getIndexAttr(1));

    // A_k = A[:, k:k+tileK]
    Value A_kernel = builder.create<tensor::ExtractSliceOp>(
        loc, A,
        akernel_offsets,
        akernel_sizes,
        akernel_stridess);

    SmallVector<OpFoldResult> bkernel_offsets;
    bkernel_offsets.push_back(ivK);
    bkernel_offsets.push_back(builder.getIndexAttr(0));
    SmallVector<OpFoldResult> bkernel_sizes;
    bkernel_sizes.push_back(builder.getIndexAttr(vector_tile));
    bkernel_sizes.push_back(builder.getIndexAttr(N));
    SmallVector<OpFoldResult> bkernel_stridess;
    bkernel_stridess.push_back(builder.getIndexAttr(1));
    bkernel_stridess.push_back(builder.getIndexAttr(1));

    Value B_kernel = builder.create<tensor::ExtractSliceOp>(
        loc, B,
        bkernel_offsets,
        bkernel_sizes,
        bkernel_stridess);

    auto vectorReductionMatmul = builder.create<linalg::MatmulOp>(
        loc, ValueRange{A_kernel, B_kernel}, ValueRange{accIter});
    Value newAcc = vectorReductionMatmul.getResult(0);
    builder.create<scf::YieldOp>(loc, newAcc);
    matmulOp.replaceAllUsesWith(forOp);
    matmulOp.erase();
    return success();
  }

  LogicalResult matmulVectorReductionAffine(linalg::MatmulOp matmulOp) {

    constexpr StringLiteral configAttrName = "lowering_config";
    auto attr = matmulOp->getAttr(configAttrName);
    auto lowering_config_attr = llvm::dyn_cast<atir::LoweringConfigAttr>(attr);

    OpBuilder builder(matmulOp);
    Location loc = matmulOp.getLoc();

    Value A = matmulOp.getInputs()[0];
    Value B = matmulOp.getInputs()[1];
    Value C = matmulOp.getOutputs()[0];

    auto aType = llvm::dyn_cast<MemRefType>(A.getType());
    auto bType = llvm::dyn_cast<MemRefType>(B.getType());
    auto cType = llvm::dyn_cast<MemRefType>(C.getType());

    int64_t M = aType.getShape()[0];
    int64_t K = aType.getShape()[1];
    int64_t N = bType.getShape()[1];

    std::vector<int64_t> vector_reduction  = lowering_config_attr.getLoweringConfigVectorReduction();

    int64_t vector_tile = vector_reduction[2];

    auto kLoop = builder.create<affine::AffineForOp>(loc, 0, K, vector_tile);

    builder.setInsertionPointToStart(kLoop.getBody());

    Value k = kLoop.getInductionVar();


    SmallVector<OpFoldResult> a_offsets = {
        builder.getIndexAttr(0),
        k
    };
    SmallVector<OpFoldResult> a_sizes = {
        builder.getIndexAttr(M),
        builder.getIndexAttr(vector_tile)
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
        k,
        builder.getIndexAttr(0)
    };
    SmallVector<OpFoldResult> b_sizes = {
        builder.getIndexAttr(vector_tile),
        builder.getIndexAttr(N)
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

    auto new_matmul = builder.create<linalg::MatmulOp>(loc, ValueRange{Asub, Bsub}, ValueRange{C});

    matmulOp.erase();
    return success();
  }

};

std::unique_ptr<mlir::Pass> createVectorReduction()
{
  llvm::dbgs() << "this is createVectorReduction\n";
  return std::make_unique<VectorReduction>();
}
}