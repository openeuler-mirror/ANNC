#include "GemmPlan.h"
#include "GemmTilingUtils.h"
#include "Target/aarch64/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace annc {
namespace {

using aarch64::gemm::GemmTilingPlan;

LogicalResult validateProblem(Operation *op, const GemmTilingPlan &plan,
                              const aarch64::gemm::GemmTileShape &shape) {
  const bool local =
      aarch64::gemm::hasStage(op, aarch64::gemm::kThreadTiledStage);
  if (shape.k != plan.k || shape.m > plan.m || shape.n > plan.n ||
      (!local && (shape.m != plan.m || shape.n != plan.n)))
    return op->emitOpError("has plan dimensions inconsistent with operands");
  return success();
}

LogicalResult createCacheTile(OpBuilder &builder, Operation *original, Value ic,
                              Value pc, Value jc, int64_t mSize, int64_t kSize,
                              int64_t nSize, llvm::StringRef kcMode,
                              llvm::StringRef kcRole) {
  Location loc = original->getLoc();
  auto tagKRun = [&](Operation *tiled) {
    tiled->setDiscardableAttr(aarch64::gemm::kKcModeAttrName,
                              builder.getStringAttr(kcMode));
    tiled->setDiscardableAttr(aarch64::gemm::kKcRoleAttrName,
                              builder.getStringAttr(kcRole));
  };
  if (auto generic = llvm::dyn_cast<linalg::GenericOp>(original)) {
    FailureOr<linalg::GenericOp> tiled = aarch64::gemm::createTiledGemmGeneric(
        builder, generic, ic, jc, pc, mSize, nSize, kSize,
        aarch64::gemm::kCacheBlockedStage);
    if (failed(tiled)) return failure();
    tagKRun(*tiled);
    return success();
  }

  auto matmul = llvm::cast<linalg::MatmulOp>(original);
  auto tiled = aarch64::gemm::createTiledMatmul(
      builder, loc, matmul.getInputs()[0], matmul.getInputs()[1],
      matmul.getOutputs()[0], ic, jc, pc, mSize, nSize, kSize,
      original->getAttrs());
  aarch64::gemm::setStage(tiled, builder, aarch64::gemm::kCacheBlockedStage);
  tagKRun(tiled);
  return success();
}

llvm::StringRef getKRoleName(aarch64::gemm::GemmKRole role) {
  switch (role) {
    case aarch64::gemm::GemmKRole::kSingle:
      return "single";
    case aarch64::gemm::GemmKRole::kFirst:
      return "first";
    case aarch64::gemm::GemmKRole::kMiddle:
      return "middle";
    case aarch64::gemm::GemmKRole::kFinal:
      return "final";
  }
  llvm_unreachable("unknown GEMM K role");
}

LogicalResult materializeCacheBlocking(Operation *op) {
  if (op->hasAttr(aarch64::gemm::kStageAttrName) &&
      !aarch64::gemm::hasStage(op, aarch64::gemm::kThreadTiledStage))
    return success();
  FailureOr<GemmTilingPlan> plan = aarch64::gemm::readTilingPlan(op);
  FailureOr<aarch64::gemm::GemmTileShape> shape =
      aarch64::gemm::readGemmTileShape(op);
  if (failed(plan) || failed(shape) ||
      failed(validateProblem(op, *plan, *shape)))
    return failure();
  if (op->getNumResults() != 0)
    return op->emitOpError("requires buffer-semantics GEMM anchors");

  OpBuilder builder(op);
  Location loc = op->getLoc();
  const aarch64::gemm::GemmCacheStrategy strategy{plan->cacheStrategyKind,
                                                  plan->cacheTile};
  const auto description = aarch64::gemm::describeGemmCache(
      strategy, shape->m, shape->n, shape->k);
  auto emitRows = [&](Value ic, int64_t mSize) -> LogicalResult {
    for (const auto &reduction : description.reduction) {
      const bool firstRun =
          reduction.role == aarch64::gemm::GemmKRole::kSingle ||
          reduction.role == aarch64::gemm::GemmKRole::kFirst;
      const auto mode = firstRun ? plan->firstKcMode : plan->nextKcMode;
      const llvm::StringRef role = getKRoleName(reduction.role);
      auto emitReduction = [&](Value pc, int64_t kSize) -> LogicalResult {
        auto emitColumns = [&](Value jc, int64_t nSize) {
          return createCacheTile(builder, op, ic, pc, jc, mSize, kSize, nSize,
                                 aarch64::gemm::getKcModeName(mode), role);
        };
        return aarch64::gemm::materializeGemmPartition(
            builder, loc, description.columns, emitColumns);
      };
      if (failed(aarch64::gemm::materializeGemmTileRun(
              builder, loc, reduction.tiles, emitReduction)))
        return failure();
    }
    return success();
  };
  if (failed(aarch64::gemm::materializeGemmPartition(
          builder, loc, description.rows, emitRows)))
    return failure();
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
