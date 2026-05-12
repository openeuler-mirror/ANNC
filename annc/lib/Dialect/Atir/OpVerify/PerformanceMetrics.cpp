#include "Dialect/Atir/OpVerify/PerformanceMetrics.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace atir {

std::vector<std::string> sortKernelsByPerformance(
    const std::vector<PerformanceMetrics> &metrics,
    double timeWeight) {
  
  if (metrics.empty()) {
    llvm::errs() << "Warning: No metrics to sort\n";
    return {};
  }

  double minTime = std::numeric_limits<double>::max();
  double maxTime = 0.0;

  for (const auto &m : metrics) {
    minTime = std::min(minTime, m.execTimeMs);
    maxTime = std::max(maxTime, m.execTimeMs);
  }

  double timeRange = (maxTime - minTime) > 1e-9 ? (maxTime - minTime) : 1.0;

  struct ScoredKernel {
    std::string path;
    double score;
  };

  std::vector<ScoredKernel> scoredKernels;
  scoredKernels.reserve(metrics.size());

  for (const auto &m : metrics) {
    double normalizedTime = (m.execTimeMs - minTime) / timeRange;

    double score = timeWeight * normalizedTime;

    scoredKernels.push_back({m.kernelPath, score});
  }

  std::sort(scoredKernels.begin(), scoredKernels.end(),
            [](const ScoredKernel &a, const ScoredKernel &b) {
              return a.score < b.score;
            });

  std::vector<std::string> sortedPaths;
  sortedPaths.reserve(scoredKernels.size());
  for (const auto &sk : scoredKernels) {
    sortedPaths.push_back(sk.path);
  }

  llvm::outs() << "\n=== Kernel Performance Ranking ===\n";
  llvm::outs() << "Weights: time=" << timeWeight << "\n";
  for (size_t i = 0; i < scoredKernels.size(); ++i) {
    llvm::outs() << "Rank " << (i + 1) << ": " << scoredKernels[i].path 
                 << " (score: " << scoredKernels[i].score << ")\n";
  }
  llvm::outs() << "================================\n";

  return sortedPaths;
}

} // namespace atir
