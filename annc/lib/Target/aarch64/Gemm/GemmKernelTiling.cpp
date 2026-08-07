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

Value createIndexConstant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

LogicalResult getStaticTileShape(Operation *op, int64_t &m, int64_t &n,
                                 int64_t &k) {
  auto lhsType =
      llvm::dyn_cast<MemRefType>(aarch64::gemm::getGemmInput(op, 0).getType());
  auto rhsType =
      llvm::dyn_cast<MemRefType>(aarch64::gemm::getGemmInput(op, 1).getType());
  auto outType =
      llvm::dyn_cast<MemRefType>(aarch64::gemm::getGemmOutput(op, 0).getType());
  if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 ||
      rhsType.getRank() != 2 || outType.getRank() != 2 ||
      !lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      !outType.hasStaticShape()) {
    return op->emitOpError(
        "requires statically shaped cache-blocked memref operands");
  }
  m = lhsType.getShape()[0];
  k = lhsType.getShape()[1];
  n = rhsType.getShape()[1];
  if (m <= 0 || n <= 0 || k <= 0 || rhsType.getShape()[0] != k ||
      outType.getShape()[0] != m || outType.getShape()[1] != n) {
    return op->emitOpError("has inconsistent static cache tile dimensions");
  }
  return success();
}

LogicalResult createKernelTile(OpBuilder &builder, Operation *original,
                               Value ir, Value jr, int64_t mSize, int64_t nSize,
                               int64_t kSize) {
  Location loc = original->getLoc();
  Value zero = createIndexConstant(builder, loc, 0);
  if (auto generic = llvm::dyn_cast<linalg::GenericOp>(original)) {
    SmallVector<OpFoldResult> offsets = {ir, jr, zero};
    SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(mSize),
                                       builder.getIndexAttr(nSize),
                                       builder.getIndexAttr(kSize)};
    FailureOr<linalg::GenericOp> tiled =
        aarch64::gemm::cloneTiledGemmGeneric(builder, generic, offsets, sizes);
    if (failed(tiled)) return failure();
    aarch64::gemm::setStage(*tiled, builder, aarch64::gemm::kKernelTiledStage);
    return success();
  }

  auto matmul = llvm::cast<linalg::MatmulOp>(original);
  SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1),
                                       builder.getIndexAttr(1)};
  Value lhsTile = builder.create<memref::SubViewOp>(
      loc, matmul.getInputs()[0], SmallVector<OpFoldResult>{ir, zero},
      SmallVector<OpFoldResult>{builder.getIndexAttr(mSize),
                                builder.getIndexAttr(kSize)},
      strides);
  Value rhsTile = builder.create<memref::SubViewOp>(
      loc, matmul.getInputs()[1], SmallVector<OpFoldResult>{zero, jr},
      SmallVector<OpFoldResult>{builder.getIndexAttr(kSize),
                                builder.getIndexAttr(nSize)},
      strides);
  Value outTile = builder.create<memref::SubViewOp>(
      loc, matmul.getOutputs()[0], SmallVector<OpFoldResult>{ir, jr},
      SmallVector<OpFoldResult>{builder.getIndexAttr(mSize),
                                builder.getIndexAttr(nSize)},
      strides);

  auto tiled = builder.create<linalg::MatmulOp>(
      loc, ValueRange{lhsTile, rhsTile}, ValueRange{outTile});
  tiled->setAttrs(original->getAttrs());
  aarch64::gemm::setStage(tiled, builder, aarch64::gemm::kKernelTiledStage);
  return success();
}

LogicalResult createNBlocks(OpBuilder &builder, Operation *original, Value ir,
                            int64_t mSize, int64_t n, int64_t k, int64_t nr) {
  Location loc = original->getLoc();
  const int64_t fullN = n / nr * nr;
  if (fullN == nr) {
    if (failed(createKernelTile(builder, original, ir,
                                createIndexConstant(builder, loc, 0), mSize, nr,
                                k)))
      return failure();
  } else if (fullN > nr) {
    Value zero = createIndexConstant(builder, loc, 0);
    Value upper = createIndexConstant(builder, loc, fullN);
    Value step = createIndexConstant(builder, loc, nr);
    auto jrLoop = builder.create<scf::ForOp>(loc, zero, upper, step);
    aarch64::gemm::disableLoopUnrolling(builder, jrLoop);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(jrLoop.getBody());
    if (failed(createKernelTile(builder, original, ir, jrLoop.getInductionVar(),
                                mSize, nr, k)))
      return failure();
  }
  if (fullN != n) {
    if (failed(createKernelTile(builder, original, ir,
                                createIndexConstant(builder, loc, fullN), mSize,
                                n - fullN, k)))
      return failure();
  }
  return success();
}

LogicalResult materializeKernelTiling(Operation *op) {
  FailureOr<GemmTilingPlan> plan = aarch64::gemm::readTilingPlan(op);
  if (failed(plan)) return failure();
  FailureOr<int64_t> nr = aarch64::gemm::getGemmNr(
      plan->kernelTile, plan->vectorLengthBytes, plan->dataType);
  if (failed(nr)) return op->emitOpError("has an invalid kernel N tile");
  const bool isCacheBlocked =
      aarch64::gemm::hasStage(op, aarch64::gemm::kCacheBlockedStage);
  if (!isCacheBlocked) return op->emitOpError("requires a cache-blocked GEMM");
  if (op->getNumResults() != 0)
    return op->emitOpError("requires buffer-semantics GEMM anchors");

  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  if (failed(getStaticTileShape(op, m, n, k))) return failure();

  OpBuilder builder(op);
  Location loc = op->getLoc();
  const int64_t fullM =
      m / plan->kernelTile.mr * plan->kernelTile.mr;
  if (fullM == plan->kernelTile.mr) {
    if (failed(createNBlocks(builder, op, createIndexConstant(builder, loc, 0),
                             plan->kernelTile.mr, n, k, *nr)))
      return failure();
  } else if (fullM > plan->kernelTile.mr) {
    Value zero = createIndexConstant(builder, loc, 0);
    Value upper = createIndexConstant(builder, loc, fullM);
    Value step = createIndexConstant(builder, loc, plan->kernelTile.mr);
    auto irLoop = builder.create<scf::ForOp>(loc, zero, upper, step);
    aarch64::gemm::disableLoopUnrolling(builder, irLoop);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(irLoop.getBody());
    if (failed(createNBlocks(builder, op, irLoop.getInductionVar(),
                             plan->kernelTile.mr, n, k, *nr)))
      return failure();
  }
  if (fullM != m) {
    if (failed(createNBlocks(builder, op,
                             createIndexConstant(builder, loc, fullM),
                             m - fullM, n, k, *nr)))
      return failure();
  }

  op->erase();
  return success();
}

class AArch64GemmKernelTiling
    : public AArch64GemmKernelTilingBase<AArch64GemmKernelTiling> {
 public:
  using Base::Base;

  void runOnOperation() override {
    SmallVector<Operation *> candidates;
    getOperation().walk([&](Operation *op) {
      if (aarch64::gemm::isGemmAnchor(op) &&
          aarch64::gemm::hasStage(op, aarch64::gemm::kCacheBlockedStage)) {
        candidates.push_back(op);
      }
    });
    for (Operation *op : candidates) {
      if (failed(materializeKernelTiling(op))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAArch64GemmKernelTiling() {
  return std::make_unique<AArch64GemmKernelTiling>();
}

}  // namespace annc
