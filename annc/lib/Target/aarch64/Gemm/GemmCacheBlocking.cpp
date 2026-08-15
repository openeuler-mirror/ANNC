#include <algorithm>

#include "GemmPlan.h"
#include "GemmTilingUtils.h"
#include "Target/aarch64/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace annc {
namespace {

using aarch64::gemm::GemmTilingPlan;

LogicalResult validateProblem(Operation *op, const GemmTilingPlan &plan) {
  auto lhsType =
      llvm::dyn_cast<MemRefType>(aarch64::gemm::getGemmInput(op, 0).getType());
  auto rhsType =
      llvm::dyn_cast<MemRefType>(aarch64::gemm::getGemmInput(op, 1).getType());
  auto outType =
      llvm::dyn_cast<MemRefType>(aarch64::gemm::getGemmOutput(op, 0).getType());
  if (!lhsType || !rhsType || !outType || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape() || !outType.hasStaticShape()) {
    return op->emitOpError(
        "requires statically shaped bufferized memref operands");
  }
  if (lhsType.getShape() != ArrayRef<int64_t>{plan.m, plan.k} ||
      rhsType.getShape() != ArrayRef<int64_t>{plan.k, plan.n} ||
      outType.getShape() != ArrayRef<int64_t>{plan.m, plan.n}) {
    return op->emitOpError("has plan dimensions inconsistent with operands");
  }
  return success();
}

Value createIndexConstant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

LogicalResult createCacheTile(OpBuilder &builder, Operation *original, Value ic,
                              Value pc, Value jc, int64_t mSize, int64_t kSize,
                              int64_t nSize, llvm::StringRef kcMode) {
  Location loc = original->getLoc();
  if (auto generic = llvm::dyn_cast<linalg::GenericOp>(original)) {
    SmallVector<OpFoldResult> offsets = {ic, jc, pc};
    SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(mSize),
                                       builder.getIndexAttr(nSize),
                                       builder.getIndexAttr(kSize)};
    FailureOr<linalg::GenericOp> tiled =
        aarch64::gemm::cloneTiledGemmGeneric(builder, generic, offsets, sizes);
    if (failed(tiled)) return failure();
    aarch64::gemm::setStage(*tiled, builder, aarch64::gemm::kCacheBlockedStage);
    (*tiled)->setDiscardableAttr(aarch64::gemm::kKcModeAttrName,
                                 builder.getStringAttr(kcMode));
    return success();
  }

  auto matmul = llvm::cast<linalg::MatmulOp>(original);
  SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1),
                                       builder.getIndexAttr(1)};

  Value lhsTile = builder.create<memref::SubViewOp>(
      loc, matmul.getInputs()[0], SmallVector<OpFoldResult>{ic, pc},
      SmallVector<OpFoldResult>{builder.getIndexAttr(mSize),
                                builder.getIndexAttr(kSize)},
      strides);
  Value rhsTile = builder.create<memref::SubViewOp>(
      loc, matmul.getInputs()[1], SmallVector<OpFoldResult>{pc, jc},
      SmallVector<OpFoldResult>{builder.getIndexAttr(kSize),
                                builder.getIndexAttr(nSize)},
      strides);
  Value outTile = builder.create<memref::SubViewOp>(
      loc, matmul.getOutputs()[0], SmallVector<OpFoldResult>{ic, jc},
      SmallVector<OpFoldResult>{builder.getIndexAttr(mSize),
                                builder.getIndexAttr(nSize)},
      strides);

  auto tiled = builder.create<linalg::MatmulOp>(
      loc, ValueRange{lhsTile, rhsTile}, ValueRange{outTile});
  tiled->setAttrs(original->getAttrs());
  aarch64::gemm::setStage(tiled, builder, aarch64::gemm::kCacheBlockedStage);
  tiled->setDiscardableAttr(aarch64::gemm::kKcModeAttrName,
                            builder.getStringAttr(kcMode));
  return success();
}

LogicalResult createJcBlocks(OpBuilder &builder, Operation *original, Value ic,
                             Value pc, int64_t mSize, int64_t kSize, int64_t n,
                             int64_t nc, llvm::StringRef kcMode) {
  Location loc = original->getLoc();
  const int64_t fullN = n / nc * nc;
  if (fullN > 0) {
    if (fullN == nc) {
      if (failed(createCacheTile(builder, original, ic, pc,
                                 createIndexConstant(builder, loc, 0), mSize,
                                 kSize, nc, kcMode)))
        return failure();
    } else {
      Value zero = createIndexConstant(builder, loc, 0);
      Value upper = createIndexConstant(builder, loc, fullN);
      Value step = createIndexConstant(builder, loc, nc);
      auto jcLoop = builder.create<scf::ForOp>(loc, zero, upper, step);
      aarch64::gemm::disableLoopUnrolling(builder, jcLoop);
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(jcLoop.getBody());
      if (failed(createCacheTile(builder, original, ic, pc,
                                 jcLoop.getInductionVar(), mSize, kSize, nc,
                                 kcMode)))
        return failure();
    }
  }
  if (fullN != n) {
    if (failed(createCacheTile(builder, original, ic, pc,
                               createIndexConstant(builder, loc, fullN), mSize,
                               kSize, n - fullN, kcMode)))
      return failure();
  }
  return success();
}

LogicalResult createKcBlocks(OpBuilder &builder, Operation *original, Value ic,
                             int64_t mSize, const GemmTilingPlan &plan) {
  Location loc = original->getLoc();
  const int64_t firstK = std::min(plan.k, plan.cacheTile.kc);
  if (failed(createJcBlocks(builder, original, ic,
                            createIndexConstant(builder, loc, 0), mSize, firstK,
                            plan.n, plan.cacheTile.nc,
                            aarch64::gemm::getKcModeName(plan.firstKcMode))))
    return failure();

  const int64_t remainingK = plan.k - firstK;
  const int64_t fullCount = remainingK / plan.cacheTile.kc;
  if (fullCount == 1) {
    if (failed(createJcBlocks(builder, original, ic,
                              createIndexConstant(builder, loc, firstK), mSize,
                              plan.cacheTile.kc, plan.n, plan.cacheTile.nc,
                              aarch64::gemm::getKcModeName(plan.nextKcMode))))
      return failure();
  } else if (fullCount > 1) {
    Value lower = createIndexConstant(builder, loc, firstK);
    Value upper =
        createIndexConstant(builder, loc,
                            firstK + fullCount * plan.cacheTile.kc);
    Value step = createIndexConstant(builder, loc, plan.cacheTile.kc);
    auto pcLoop = builder.create<scf::ForOp>(loc, lower, upper, step);
    aarch64::gemm::disableLoopUnrolling(builder, pcLoop);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(pcLoop.getBody());
    if (failed(createJcBlocks(builder, original, ic, pcLoop.getInductionVar(),
                              mSize, plan.cacheTile.kc, plan.n,
                              plan.cacheTile.nc,
                              aarch64::gemm::getKcModeName(plan.nextKcMode))))
      return failure();
  }
  const int64_t tailK = remainingK % plan.cacheTile.kc;
  if (tailK != 0) {
    if (failed(createJcBlocks(
            builder, original, ic,
            createIndexConstant(
                builder, loc, firstK + fullCount * plan.cacheTile.kc),
            mSize, tailK, plan.n, plan.cacheTile.nc,
            aarch64::gemm::getKcModeName(plan.nextKcMode))))
      return failure();
  }
  return success();
}

LogicalResult materializeCacheBlocking(Operation *op) {
  if (op->hasAttr(aarch64::gemm::kStageAttrName) &&
      !aarch64::gemm::hasStage(op, aarch64::gemm::kThreadTiledStage))
    return success();
  FailureOr<GemmTilingPlan> plan = aarch64::gemm::readTilingPlan(op);
  if (failed(plan) || failed(validateProblem(op, *plan))) return failure();
  if (op->getNumResults() != 0)
    return op->emitOpError("requires buffer-semantics GEMM anchors");

  OpBuilder builder(op);
  Location loc = op->getLoc();
  const int64_t fullM =
      plan->m / plan->cacheTile.mc * plan->cacheTile.mc;
  if (fullM == plan->cacheTile.mc) {
    if (failed(createKcBlocks(builder, op, createIndexConstant(builder, loc, 0),
                              plan->cacheTile.mc, *plan)))
      return failure();
  } else if (fullM > plan->cacheTile.mc) {
    Value zero = createIndexConstant(builder, loc, 0);
    Value upper = createIndexConstant(builder, loc, fullM);
    Value step = createIndexConstant(builder, loc, plan->cacheTile.mc);
    auto icLoop = builder.create<scf::ForOp>(loc, zero, upper, step);
    aarch64::gemm::disableLoopUnrolling(builder, icLoop);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(icLoop.getBody());
    if (failed(createKcBlocks(builder, op, icLoop.getInductionVar(),
                              plan->cacheTile.mc, *plan)))
      return failure();
  }
  if (fullM != plan->m) {
    if (failed(createKcBlocks(builder, op,
                              createIndexConstant(builder, loc, fullM),
                              plan->m - fullM, *plan)))
      return failure();
  }

  op->erase();
  return success();
}

class AArch64GemmCacheBlocking
    : public AArch64GemmCacheBlockingBase<AArch64GemmCacheBlocking> {
 public:
  using Base::Base;

  void runOnOperation() override {
    SmallVector<Operation *> candidates;
    getOperation().walk([&](Operation *op) {
      if (aarch64::gemm::isGemmAnchor(op) &&
          op->hasAttr(aarch64::gemm::kPlanAttrName) &&
          (!op->hasAttr(aarch64::gemm::kStageAttrName) ||
           aarch64::gemm::hasStage(op, aarch64::gemm::kThreadTiledStage))) {
        candidates.push_back(op);
      }
    });

    for (Operation *op : candidates) {
      if (failed(materializeCacheBlocking(op))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAArch64GemmCacheBlocking() {
  return std::make_unique<AArch64GemmCacheBlocking>();
}

}  // namespace annc
