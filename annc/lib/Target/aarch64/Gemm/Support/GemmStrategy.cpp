#include "../GemmStrategy.h"

#include <cassert>

#include "llvm/Support/ErrorHandling.h"

#include "../GemmPlan.h"

namespace annc::aarch64::gemm {

llvm::StringRef getGemmCacheStrategyName(GemmCacheStrategyKind kind) {
  switch (kind) {
    case GemmCacheStrategyKind::kFixed:
      return "fixed";
  }
  llvm_unreachable("unknown GEMM cache strategy");
}

llvm::StringRef getGemmKernelStrategyName(GemmKernelStrategyKind kind) {
  switch (kind) {
    case GemmKernelStrategyKind::kFixed:
      return "fixed";
  }
  llvm_unreachable("unknown GEMM kernel strategy");
}

std::optional<GemmCacheStrategyKind> parseGemmCacheStrategyName(
    llvm::StringRef name) {
  if (name == "fixed") return GemmCacheStrategyKind::kFixed;
  return std::nullopt;
}

std::optional<GemmKernelStrategyKind> parseGemmKernelStrategyName(
    llvm::StringRef name) {
  if (name == "fixed") return GemmKernelStrategyKind::kFixed;
  return std::nullopt;
}

bool isRhsPrepackSupported(GemmCacheStrategyKind cacheStrategy,
                           GemmKernelStrategyKind kernelStrategy) {
  return cacheStrategy == GemmCacheStrategyKind::kFixed &&
         kernelStrategy == GemmKernelStrategyKind::kFixed;
}

bool isThreadPlanningSupported(GemmCacheStrategyKind cacheStrategy,
                               GemmKernelStrategyKind kernelStrategy) {
  return cacheStrategy == GemmCacheStrategyKind::kFixed &&
         kernelStrategy == GemmKernelStrategyKind::kFixed;
}

GemmPartition describeUniformPartition(int64_t extent, int64_t blockSize) {
  assert(extent > 0 && blockSize > 0);
  GemmPartition partition;
  const int64_t fullCount = extent / blockSize;
  const int64_t tail = extent % blockSize;
  if (fullCount) partition.push_back({0, blockSize, fullCount});
  if (tail) partition.push_back({fullCount * blockSize, tail, 1});
  return partition;
}

namespace {

class FixedCacheTilingStrategy final : public GemmCacheTilingStrategy {
 public:
  GemmCacheTilingPlan buildPlan(
      const GemmCacheTilingRequest &request) const override {
    assert(request.strategy.kind == GemmCacheStrategyKind::kFixed);
    const auto &tile = getFixedCacheParameters(request.strategy);
    GemmCacheTilingPlan description{
        describeUniformPartition(request.inputBlock.m, tile.mc), {},
        describeUniformPartition(request.inputBlock.n, tile.nc)};
    const auto reduction =
        describeUniformPartition(request.inputBlock.k, tile.kc);
    auto &runs = description.reduction;
    if (request.inputBlock.k <= tile.kc) {
      runs.push_back({{0, request.inputBlock.k, 1}, GemmKRole::kSingle});
      return description;
    }

    runs.push_back({{0, tile.kc, 1}, GemmKRole::kFirst});
    const auto &last = reduction.back();
    const int64_t finalOffset = last.offset + (last.count - 1) * last.extent;
    const int64_t middleCount = (finalOffset - tile.kc) / tile.kc;
    if (middleCount)
      runs.push_back({{tile.kc, tile.kc, middleCount}, GemmKRole::kMiddle});
    runs.push_back({{finalOffset, last.extent, 1}, GemmKRole::kFinal});
    return description;
  }

};

class FixedKernelTilingStrategy final : public GemmKernelTilingStrategy {
 public:
  std::optional<GemmKernelStrategy> configureForExecution(
      const GemmKernelStrategy &strategy, GemmExecutionKind executionKind,
      GemmIsa isa, const GemmKernelABI &abi) const override {
    if (executionKind == GemmExecutionKind::kGemm) return strategy;
    if (isa != GemmIsa::kNeon) return std::nullopt;
    return GemmKernelStrategy{strategy.kind,
                              FixedKernelParameters{abi.maxMr,
                                                    abi.maxPanelLanes}};
  }

  bool isValidForExecution(const GemmKernelStrategy &strategy,
                           GemmExecutionKind executionKind, GemmIsa isa,
                           GemmDataType dataType,
                           const GemmKernelABI &abi) const override {
    const auto &tile = getFixedKernelParameters(strategy);
    if (tile.mr <= 0 || tile.panelLanes <= 0 || tile.mr > abi.maxMr ||
        tile.panelLanes > abi.maxPanelLanes ||
        mlir::failed(getGemmNr(tile.panelLanes, abi.vectorLengthBytes,
                               dataType, executionKind)))
      return false;
    return executionKind == GemmExecutionKind::kGemm ||
           (isa == GemmIsa::kNeon && tile.mr == abi.maxMr &&
            tile.panelLanes == abi.maxPanelLanes);
  }

  GemmKernelTilingPlan buildPlan(
      const GemmKernelTilingRequest &request) const override {
    assert(request.strategy.kind == GemmKernelStrategyKind::kFixed);
    const auto &tile = getFixedKernelParameters(request.strategy);
    const auto rows = describeUniformPartition(request.inputBlock.m, tile.mr);
    const auto nr = getGemmNr(tile.panelLanes,
                              request.targetInfo.vectorLengthBytes,
                              request.targetInfo.dataType,
                              request.targetInfo.executionKind);
    assert(succeeded(nr));
    const auto columns = describeUniformPartition(request.inputBlock.n, *nr);
    GemmKernelTilingPlan plan;
    for (const GemmTileRun &row : rows)
      plan.groups.push_back({GemmKernelOrder::kRowsFirst, row, columns});
    return plan;
  }

  int64_t getTaskRowAlignment(
      const GemmKernelStrategy &strategy) const override {
    const auto &tile = getFixedKernelParameters(strategy);
    assert(strategy.kind == GemmKernelStrategyKind::kFixed && tile.mr > 0);
    return tile.mr;
  }
};

}  // namespace

const GemmCacheTilingStrategy &
getGemmCacheTilingStrategy(GemmCacheStrategyKind kind) {
  static const FixedCacheTilingStrategy fixed;
  switch (kind) {
    case GemmCacheStrategyKind::kFixed:
      return fixed;
  }
  llvm_unreachable("unknown GEMM cache tiling strategy");
}

const GemmKernelTilingStrategy &
getGemmKernelTilingStrategy(GemmKernelStrategyKind kind) {
  static const FixedKernelTilingStrategy fixed;
  switch (kind) {
    case GemmKernelStrategyKind::kFixed:
      return fixed;
  }
  llvm_unreachable("unknown GEMM kernel tiling strategy");
}

}  // namespace annc::aarch64::gemm
