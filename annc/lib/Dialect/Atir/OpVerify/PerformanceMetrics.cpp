#include "Dialect/Atir/OpVerify/PerformanceMetrics.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace atir {

std::vector<std::string> sortKernelsByPerformance(
    const std::vector<PerformanceMetrics> &metrics,
    double /*timeWeight*/) {
  
  if (metrics.empty()) {
    llvm::errs() << "Warning: No metrics to sort\n";
    return {};
  }

  // Sort by execution time (ascending)
  struct ScoredKernel {
    std::string path;
    double time;
  };

  std::vector<ScoredKernel> scoredKernels;
  scoredKernels.reserve(metrics.size());

  for (const auto &m : metrics) {
    scoredKernels.push_back({m.kernelPath, m.execTimeMs});
  }

  std::sort(scoredKernels.begin(), scoredKernels.end(),
            [](const ScoredKernel &a, const ScoredKernel &b) {
              return a.time < b.time;
            });

  std::vector<std::string> sortedPaths;
  sortedPaths.reserve(scoredKernels.size());
  for (const auto &sk : scoredKernels) {
    sortedPaths.push_back(sk.path);
  }

  llvm::outs() << "\n=== Kernel Performance Ranking ===\n";
  for (size_t i = 0; i < scoredKernels.size(); ++i) {
    llvm::outs() << "Rank " << (i + 1) << ": " << scoredKernels[i].path 
                 << " (" << scoredKernels[i].time << " ms)\n";
  }
  llvm::outs() << "================================\n";

  return sortedPaths;
}

} // namespace atir
