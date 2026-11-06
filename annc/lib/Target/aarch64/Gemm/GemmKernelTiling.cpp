#include "GemmPlan.h"
#include "GemmTilingUtils.h"
#include "Target/aarch64/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace annc {
namespace {

Value createIndexConstant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

LogicalResult createKernelTile(OpBuilder &builder, Operation *original,
                               Value ir, Value jr, int64_t mSize, int64_t nSize,
                               int64_t kSize) {
  Location loc = original->getLoc();
  Value zero = createIndexConstant(builder, loc, 0);
  if (auto generic = llvm::dyn_cast<linalg::GenericOp>(original)) {
    FailureOr<linalg::GenericOp> tiled = aarch64::gemm::createTiledGemmGeneric(
        builder, generic, ir, jr, zero, mSize, nSize, kSize,
        aarch64::gemm::kKernelTiledStage);
    return failed(tiled) ? failure() : success();
  }

  auto matmul = llvm::cast<linalg::MatmulOp>(original);
  auto tiled = aarch64::gemm::createTiledMatmul(
      builder, loc, matmul.getInputs()[0], matmul.getInputs()[1],
      matmul.getOutputs()[0], ir, jr, zero, mSize, nSize, kSize,
      original->getAttrs());
  aarch64::gemm::setStage(tiled, builder, aarch64::gemm::kKernelTiledStage);
  return success();
}

LogicalResult materializeKernelTiling(Operation *op) {
  FailureOr<aarch64::gemm::GemmTuningStrategy> strategy =
      aarch64::gemm::readStrategy(op);
  if (failed(strategy)) return failure();
  if (!aarch64::gemm::hasStage(op, aarch64::gemm::kCacheBlockedStage))
    return op->emitOpError("requires a cache-blocked GEMM");
  if (op->getNumResults() != 0)
    return op->emitOpError("requires buffer-semantics GEMM anchors");

  FailureOr<aarch64::gemm::GemmBlockShape> shape =
      aarch64::gemm::readGemmBlockShape(op);
  if (failed(shape)) return failure();

  OpBuilder builder(op);
  Location loc = op->getLoc();
  const auto &kernelStrategy = strategy->kernel;
  const auto tilingPlan =
      aarch64::gemm::getGemmKernelTilingStrategy(kernelStrategy.kind)
          .buildPlan({kernelStrategy, *shape,
                      {aarch64::gemm::getGemmKernelABI(*strategy).vectorLengthBytes,
                       strategy->dataType, strategy->executionKind}});
  for (const auto &group : tilingPlan.groups) {
    const bool rowsFirst =
        group.order == aarch64::gemm::GemmKernelOrder::kRowsFirst;
    auto emitOuter = [&](Value outerOffset, int64_t outerSize) {
      return aarch64::gemm::materializeGemmPartition(
          builder, loc, group.inner,
          [&](Value innerOffset, int64_t innerSize) {
            return createKernelTile(
                builder, op, rowsFirst ? outerOffset : innerOffset,
                rowsFirst ? innerOffset : outerOffset,
                rowsFirst ? outerSize : innerSize,
                rowsFirst ? innerSize : outerSize, shape->k);
          });
    };
    if (failed(aarch64::gemm::materializeGemmTileRun(
            builder, loc, group.outer, emitOuter)))
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
