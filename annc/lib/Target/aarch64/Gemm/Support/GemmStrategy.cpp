#include "../GemmStrategy.h"

#include <cassert>

namespace annc::aarch64::gemm {

GemmPartition describeFixedPartition(int64_t extent, int64_t blockSize) {
  assert(extent > 0 && blockSize > 0);
  GemmPartition partition;
  const int64_t fullCount = extent / blockSize;
  const int64_t tail = extent % blockSize;
  if (fullCount) partition.push_back({0, blockSize, fullCount});
  if (tail) partition.push_back({fullCount * blockSize, tail, 1});
  return partition;
}

GemmCacheDescription describeGemmCache(const GemmCacheStrategy &strategy,
                                       int64_t m, int64_t n, int64_t k) {
  assert(strategy.kind == GemmCacheStrategyKind::kFixed);
  const auto &tile = strategy.tile;
  GemmCacheDescription description{describeFixedPartition(m, tile.mc),
                                   {},
                                   describeFixedPartition(n, tile.nc)};
  const auto reduction = describeFixedPartition(k, tile.kc);
  auto &runs = description.reduction;
  if (k <= tile.kc) {
    runs.push_back({{0, k, 1}, GemmKRole::kSingle});
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

std::vector<GemmKernelRegion> describeGemmKernel(GemmKernelStrategyKind kind,
                                                 int64_t mr, int64_t nr,
                                                 int64_t m, int64_t n) {
  assert(kind == GemmKernelStrategyKind::kFixed);
  return {{GemmKernelOrder::kRowsFirst, describeFixedPartition(m, mr),
           describeFixedPartition(n, nr)}};
}

int64_t getGemmTaskRowAlignment(GemmKernelStrategyKind kind, int64_t mr) {
  assert(kind == GemmKernelStrategyKind::kFixed && mr > 0);
  return mr;
}

}  // namespace annc::aarch64::gemm
