#ifndef ANNC_LIB_TARGET_AARCH64_GEMM_GEMMSTRATEGY_H
#define ANNC_LIB_TARGET_AARCH64_GEMM_GEMMSTRATEGY_H

#include <cstdint>
#include <optional>
#include <vector>

#include "llvm/ADT/StringRef.h"

namespace annc::aarch64::gemm {

enum class GemmCacheStrategyKind { kFixed };
enum class GemmKernelStrategyKind { kFixed };

llvm::StringRef getGemmCacheStrategyName(GemmCacheStrategyKind kind);
llvm::StringRef getGemmKernelStrategyName(GemmKernelStrategyKind kind);
std::optional<GemmCacheStrategyKind> parseGemmCacheStrategyName(
    llvm::StringRef name);
std::optional<GemmKernelStrategyKind> parseGemmKernelStrategyName(
    llvm::StringRef name);

struct GemmCacheTile {
  int64_t mc;
  int64_t nc;
  int64_t kc;
};

struct GemmKernelTile {
  int64_t mr;
  int64_t panelLanes;
};

struct GemmCacheStrategy {
  GemmCacheStrategyKind kind;
  GemmCacheTile tile;
};

struct GemmKernelStrategy {
  GemmKernelStrategyKind kind;
  GemmKernelTile tile;
};

// A run describes adjacent equal-size tiles without expanding them.
struct GemmTileRun {
  int64_t offset;
  int64_t extent;
  int64_t count;
};

using GemmPartition = std::vector<GemmTileRun>;
GemmPartition describeFixedPartition(int64_t extent, int64_t blockSize);

enum class GemmKRole { kSingle, kFirst, kMiddle, kFinal };

struct GemmKRun {
  GemmTileRun tiles;
  GemmKRole role;
  bool accumulates() const {
    return role == GemmKRole::kMiddle || role == GemmKRole::kFinal;
  }
};

struct GemmCacheDescription {
  GemmPartition rows;
  std::vector<GemmKRun> reduction;
  GemmPartition columns;
};

GemmCacheDescription describeGemmCache(const GemmCacheStrategy &strategy,
                                       int64_t m, int64_t n, int64_t k);

enum class GemmKernelOrder { kRowsFirst, kColumnsFirst };

// Each region has its own row decomposition. Columns-first regions can use
// different M partitions for different N panels without changing the pass.
struct GemmKernelRegion {
  GemmKernelOrder order;
  GemmPartition rows;
  GemmPartition columns;
};

std::vector<GemmKernelRegion> describeGemmKernel(GemmKernelStrategyKind kind,
                                                 int64_t mr, int64_t nr,
                                                 int64_t m, int64_t n);
int64_t getGemmTaskRowAlignment(GemmKernelStrategyKind kind, int64_t mr);

}  // namespace annc::aarch64::gemm

#endif  // ANNC_LIB_TARGET_AARCH64_GEMM_GEMMSTRATEGY_H
