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
class VectorCommonParallel : public VectorCommonParallelBase<VectorCommonParallel>
{
  using Base::Base;

  void runOnOperation() override
  {
    llvm::dbgs() << "this is VectorCommonParallel\n";

    ModuleOp module = getOperation();
    module.walk([&](linalg::MatmulOp matmulOp) {
      //                if (failed(matmulVectorCommonParallel(matmulOp))) {
      //                    signalPassFailure();
      //                }

      if (failed(matmulVectorCommonParallelAffine(matmulOp))) {
        signalPassFailure();
      }
    });
  }

  LogicalResult matmulVectorCommonParallel(linalg::MatmulOp matmulOp) {
    constexpr StringLiteral kConfigAttrName = "lowering_config";
    auto attr = matmulOp->getAttr(kConfigAttrName);

    auto lowering_config_attr = llvm::dyn_cast<atir::LoweringConfigAttr>(attr);

    std::vector<int64_t> vector_common_parallel_config = lowering_config_attr.getLoweringConfigVectorCommonParallel();

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

    int64_t vm = vector_common_parallel_config[0];
    int64_t vn = vector_common_parallel_config[1];

    Value c0  = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value cM  = builder.create<arith::ConstantIndexOp>(loc, M);
    Value cVm = builder.create<arith::ConstantIndexOp>(loc, vm);

    Value cN  = builder.create<arith::ConstantIndexOp>(loc, N);
    Value cVn = builder.create<arith::ConstantIndexOp>(loc, vn);

    auto forM = builder.create<scf::ForOp>(loc, c0, cM, cVm, ValueRange{C});
    builder.setInsertionPointToStart(forM.getBody());
    Value ivM = forM.getInductionVar();
    Value mIter = forM.getRegionIterArgs()[0];

    auto forN = builder.create<scf::ForOp>(loc, c0, cN, cVn, mIter);
    builder.setInsertionPointToStart(forN.getBody());
    Value ivN = forN.getInductionVar();
    Value nIter = forN.getRegionIterArgs()[0];

    SmallVector<OpFoldResult> aslice_offsets;
    aslice_offsets.push_back(ivM);
    aslice_offsets.push_back(builder.getIndexAttr(0));
    SmallVector<OpFoldResult> aslice_sizes;
    aslice_sizes.push_back(builder.getIndexAttr(vm));
    aslice_sizes.push_back(builder.getIndexAttr(K));
    SmallVector<OpFoldResult> aslice_stridess;
    aslice_stridess.push_back(builder.getIndexAttr(1));
    aslice_stridess.push_back(builder.getIndexAttr(1));

    Value A_slice = builder.create<tensor::ExtractSliceOp>(
        loc, A,
        aslice_offsets,
        aslice_sizes,
        aslice_stridess);

    SmallVector<OpFoldResult> bslice_offsets;
    bslice_offsets.push_back(builder.getIndexAttr(0));
    bslice_offsets.push_back(ivN);
    SmallVector<OpFoldResult> bslice_sizes;
    bslice_sizes.push_back(builder.getIndexAttr(K));
    bslice_sizes.push_back(builder.getIndexAttr(vn));
    SmallVector<OpFoldResult> bslice_stridess;
    bslice_stridess.push_back(builder.getIndexAttr(1));
    bslice_stridess.push_back(builder.getIndexAttr(1));

    // A_k = A[:, k:k+tileK]
    Value B_slice = builder.create<tensor::ExtractSliceOp>(
        loc, B,
        bslice_offsets,
        bslice_sizes,
        bslice_stridess);

    SmallVector<OpFoldResult> cslice_offsets;
    cslice_offsets.push_back(ivM);
    cslice_offsets.push_back(ivN);
    SmallVector<OpFoldResult> cslice_sizes;
    cslice_sizes.push_back(builder.getIndexAttr(vm));
    cslice_sizes.push_back(builder.getIndexAttr(vn));
    SmallVector<OpFoldResult> cslice_stridess;
    cslice_stridess.push_back(builder.getIndexAttr(1));
    cslice_stridess.push_back(builder.getIndexAttr(1));

    // A_k = A[:, k:k+tileK]
    Value C_slice = builder.create<tensor::ExtractSliceOp>(
        loc, nIter,
        cslice_offsets,
        cslice_sizes,
        cslice_stridess);

    auto vectorCommonParallelMatmul = builder.create<linalg::MatmulOp>(loc, ValueRange{A_slice, B_slice}, ValueRange{C_slice});
    vectorCommonParallelMatmul->setAttr(kConfigAttrName, matmulOp->getAttr(kConfigAttrName));

    Value matmul_value = vectorCommonParallelMatmul.getResult(0);

    SmallVector<OpFoldResult> cinsert_offsets;
    cinsert_offsets.push_back(ivM);
    cinsert_offsets.push_back(ivN);
    SmallVector<OpFoldResult> cinsert_sizes;
    cinsert_sizes.push_back(builder.getIndexAttr(vm));
    cinsert_sizes.push_back(builder.getIndexAttr(vn));
    SmallVector<OpFoldResult> cinsert_stridess;
    cinsert_stridess.push_back(builder.getIndexAttr(1));
    cinsert_stridess.push_back(builder.getIndexAttr(1));

    Value C_insert = builder.create<tensor::InsertSliceOp>(
        loc, matmul_value,
        nIter,cinsert_offsets, cinsert_sizes, cinsert_stridess);

    builder.create<scf::YieldOp>(loc, C_insert);
    builder.setInsertionPointToEnd(forM.getBody());

    auto forN_value = forN.getResult(0);
    builder.create<scf::YieldOp>(loc, forN_value);

    matmulOp.replaceAllUsesWith(forM);
    matmulOp.erase();
    return success();
  }

  LogicalResult matmulVectorCommonParallelAffine(linalg::MatmulOp matmulOp) {
    constexpr StringLiteral configAttrName = "lowering_config";
    auto attr = matmulOp->getAttr(configAttrName);

    auto lowering_config_attr = llvm::dyn_cast<atir::LoweringConfigAttr>(attr);

    std::vector<int64_t> vector_common_parallel_config = lowering_config_attr.getLoweringConfigVectorCommonParallel();

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

    int64_t vm = vector_common_parallel_config[0];
    int64_t vn = vector_common_parallel_config[1];

    auto mLoop = builder.create<affine::AffineForOp>(loc, 0 , M, vm);
    builder.setInsertionPointToStart(mLoop.getBody());

    auto nLoop = builder.create<affine::AffineForOp>(loc, 0, N, vn);
    builder.setInsertionPointToStart(nLoop.getBody());

    Value mi = mLoop.getInductionVar();
    Value nj = nLoop.getInductionVar();

    SmallVector<OpFoldResult> a_offsets = {
        mi,
        builder.getIndexAttr(0)
    };
    SmallVector<OpFoldResult> a_sizes = {
        builder.getIndexAttr(vm),
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
        nj
    };
    SmallVector<OpFoldResult> b_sizes = {
        builder.getIndexAttr(K),
        builder.getIndexAttr(vn)
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
        mi,
        nj
    };
    SmallVector<OpFoldResult> c_sizes = {
        builder.getIndexAttr(vm),
        builder.getIndexAttr(vn)
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

    auto new_matmul = builder.create<linalg::MatmulOp>(loc, ValueRange{Asub, Bsub}, ValueRange{Csub});

    new_matmul->setAttr(configAttrName, attr);

    matmulOp->erase();
    return success();
  }

};

std::unique_ptr<mlir::Pass> createVectorCommonParallel()
{
  llvm::dbgs() << "this is createVectorCommonParallel\n";
  return std::make_unique<VectorCommonParallel>();
}
}