#ifndef ANNC_LIB_TARGET_AARCH64_GEMM_GEMMSTRATEGY_H
#define ANNC_LIB_TARGET_AARCH64_GEMM_GEMMSTRATEGY_H

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "llvm/ADT/StringRef.h"

namespace annc::aarch64::gemm {

enum class GemmDataType;
enum class GemmExecutionKind;
enum class GemmIsa;
struct GemmKernelABI;

enum class GemmCacheStrategyKind { kFixed };
enum class GemmKernelStrategyKind { kFixed };

llvm::StringRef getGemmCacheStrategyName(GemmCacheStrategyKind kind);
llvm::StringRef getGemmKernelStrategyName(GemmKernelStrategyKind kind);
std::optional<GemmCacheStrategyKind> parseGemmCacheStrategyName(
    llvm::StringRef name);
std::optional<GemmKernelStrategyKind> parseGemmKernelStrategyName(
    llvm::StringRef name);
bool isRhsPrepackSupported(GemmCacheStrategyKind cacheStrategy,
                           GemmKernelStrategyKind kernelStrategy);
bool isThreadPlanningSupported(GemmCacheStrategyKind cacheStrategy,
                               GemmKernelStrategyKind kernelStrategy);

struct GemmBlockShape {
  int64_t m;
  int64_t n;
  int64_t k;
};

struct FixedCacheParameters {
  int64_t mc;
  int64_t nc;
  int64_t kc;
};

struct FixedKernelParameters {
  int64_t mr;
  int64_t panelLanes;
};

struct GemmCacheStrategy {
  GemmCacheStrategyKind kind;
  std::variant<FixedCacheParameters> parameters;
};

struct GemmKernelStrategy {
  GemmKernelStrategyKind kind;
  std::variant<FixedKernelParameters> parameters;
};

inline const FixedCacheParameters &getFixedCacheParameters(
    const GemmCacheStrategy &strategy) {
  return std::get<FixedCacheParameters>(strategy.parameters);
}

inline const FixedKernelParameters &getFixedKernelParameters(
    const GemmKernelStrategy &strategy) {
  return std::get<FixedKernelParameters>(strategy.parameters);
}

// A run describes adjacent equal-size tiles without expanding them.
struct GemmTileRun {
  int64_t offset;
  int64_t extent;
  int64_t count;
};

using GemmPartition = std::vector<GemmTileRun>;
GemmPartition describeUniformPartition(int64_t extent, int64_t blockSize);

enum class GemmKRole { kSingle, kFirst, kMiddle, kFinal };

struct GemmKRun {
  GemmTileRun tiles;
  GemmKRole role;
  bool accumulates() const {
    return role == GemmKRole::kMiddle || role == GemmKRole::kFinal;
  }
};

struct GemmCacheTilingPlan {
  GemmPartition rows;
  std::vector<GemmKRun> reduction;
  GemmPartition columns;
};

struct GemmCacheTilingRequest {
  GemmCacheStrategy strategy;
  GemmBlockShape inputBlock;
};

enum class GemmKernelOrder { kRowsFirst, kColumnsFirst };

struct GemmKernelTargetInfo {
  int64_t vectorLengthBytes;
  GemmDataType dataType;
  GemmExecutionKind executionKind;
};

// One outer tile run owns an ordered sequence of inner tile runs.
struct GemmKernelTileGroup {
  GemmKernelOrder order;
  GemmTileRun outer;
  GemmPartition inner;
};

struct GemmKernelTilingPlan {
  std::vector<GemmKernelTileGroup> groups;
};

struct GemmKernelTilingRequest {
  GemmKernelStrategy strategy;
  GemmBlockShape inputBlock;
  GemmKernelTargetInfo targetInfo;
};

// Cache and kernel tiling are independent policy dimensions with different
// request/plan types, so each exposes its own stage-specific interface.
class GemmCacheTilingStrategy {
 public:
  virtual ~GemmCacheTilingStrategy() = default;
  virtual GemmCacheTilingPlan buildPlan(
      const GemmCacheTilingRequest &request) const = 0;
};

class GemmKernelTilingStrategy {
 public:
  virtual ~GemmKernelTilingStrategy() = default;
  virtual std::optional<GemmKernelStrategy> configureForExecution(
      const GemmKernelStrategy &strategy, GemmExecutionKind executionKind,
      GemmIsa isa, const GemmKernelABI &abi) const = 0;
  virtual bool isValidForExecution(const GemmKernelStrategy &strategy,
                                   GemmExecutionKind executionKind,
                                   GemmIsa isa, GemmDataType dataType,
                                   const GemmKernelABI &abi) const = 0;
  virtual GemmKernelTilingPlan buildPlan(
      const GemmKernelTilingRequest &request) const = 0;
  virtual int64_t getTaskRowAlignment(
      const GemmKernelStrategy &strategy) const = 0;
};

const GemmCacheTilingStrategy &
getGemmCacheTilingStrategy(GemmCacheStrategyKind kind);
const GemmKernelTilingStrategy &
getGemmKernelTilingStrategy(GemmKernelStrategyKind kind);

}  // namespace annc::aarch64::gemm

#endif  // ANNC_LIB_TARGET_AARCH64_GEMM_GEMMSTRATEGY_H
