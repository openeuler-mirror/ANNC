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
      : kernelPath(path), execTimeMs(time){}
};

/// @brief Sort kernels based on performance metrics
std::vector<std::string> sortKernelsByPerformance(
    const std::vector<PerformanceMetrics> &metrics,
    double timeWeight = 1);
} // namespace atir

#endif // DIALECT_ATIR_OPVERIFY_PERFORMANCEMETRICS_H
