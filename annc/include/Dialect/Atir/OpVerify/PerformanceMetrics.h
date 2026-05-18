#ifndef DIALECT_ATIR_OPVERIFY_PERFORMANCEMETRICS_H
#define DIALECT_ATIR_OPVERIFY_PERFORMANCEMETRICS_H

#include <string>
#include <vector>

namespace atir {

/// @brief Performance metrics for kernel execution
struct PerformanceMetrics {
  std::string kernelPath;              // Kernel library path
  double execTimeMs = 0.0;             // Execution time in milliseconds
  
  PerformanceMetrics() = default;
  PerformanceMetrics(const std::string &path, double time)
      : kernelPath(path), execTimeMs(time) {}
};

/// @brief Warmup configuration for performance profiling
struct WarmupConfig {
  int maxWarmupRuns = 50000;           // Maximum warmup iterations
  int minWarmupRuns = 30;            // Minimum runs before checking stability
  double stabilityThreshold = 0.05; // 5% max-min difference threshold (Scheme B)
  int statRuns = 100000;               // Number of runs for final statistics
};

/// @brief Sort kernels based on performance metrics
std::vector<std::string> sortKernelsByPerformance(
    const std::vector<PerformanceMetrics> &metrics,
    double timeWeight = 1.0);

} // namespace atir

#endif // DIALECT_ATIR_OPVERIFY_PERFORMANCEMETRICS_H
