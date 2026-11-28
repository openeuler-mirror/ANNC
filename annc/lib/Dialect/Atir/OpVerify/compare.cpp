#include "Dialect/Atir/OpVerify/compare.h"
#include "Dialect/Atir/OpVerify/IoDef.h"
#include "Dialect/Atir/OpVerify/PerformanceMetrics.h"
#include "Dialect/Atir/AtirOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <set>

namespace atir {

std::string getCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
  return ss.str();
}

double calculateMaxDiff(mlir::DenseElementsAttr lhs, 
                        mlir::DenseElementsAttr rhs, 
                        double tolerance) {
  double maxDiff = 0.0;

  if (lhs.getElementType().isF32()) {
    auto lhsVals = lhs.getValues<float>();
    auto rhsVals = rhs.getValues<float>();
    for (size_t i = 0; i < lhsVals.size() && i < rhsVals.size(); ++i) {
      double diff = std::abs(lhsVals[i] - rhsVals[i]);
      maxDiff = std::max(maxDiff, diff);
    }
  } else if (lhs.getElementType().isF64()) {
    auto lhsVals = lhs.getValues<double>();
    auto rhsVals = rhs.getValues<double>();
    for (size_t i = 0; i < lhsVals.size() && i < rhsVals.size(); ++i) {
      double diff = std::abs(lhsVals[i] - rhsVals[i]);
      maxDiff = std::max(maxDiff, diff);
    }
  } else if (lhs.getElementType().isInteger(32)) {
    auto lhsVals = lhs.getValues<int32_t>();
    auto rhsVals = rhs.getValues<int32_t>();

    for (size_t i = 0; i < lhsVals.size() && i < rhsVals.size(); ++i) {
      double diff = std::abs(static_cast<double>(lhsVals[i]) -
                             static_cast<double>(rhsVals[i]));
      maxDiff = std::max(maxDiff, diff);
    }
  }

  return maxDiff;
}

void saveVerificationResult(
    const std::string &filename, size_t tensorCount, bool passed,
    const std::vector<std::pair<std::string, std::string>> &mismatchedTensors) {
  std::ofstream outFile(filename);

  if (!outFile.is_open()) {
    llvm::errs() << "Error: Could not open file " << filename << " for writing\n";
    return;
  }

  outFile << "Timestamp: " << getCurrentTimestamp() << "\n";
  outFile << "Status: " << (passed ? "PASSED" : "FAILED") << "\n";
  outFile << "Mismatched Tensors: " << mismatchedTensors.size() << "\n";

  if (!mismatchedTensors.empty()) {
    for (const auto &[name, reason] : mismatchedTensors) {
      outFile << "- " << name << ": " << reason << "\n";
    }
  }

  outFile.close();
  llvm::dbgs() << "Verification result saved to " << filename << "\n";
}

bool verifyTensor(IoTensorDef *lhs, IoTensorDef *rhs, double tolerance) {
  if (!lhs || !rhs) {
    llvm::errs() << "Error: verifyTensor called with null pointers\n";
    return false;
  }

  size_t lhsCount = lhs->getOutputCount();
  size_t rhsCount = rhs->getOutputCount();

  if (lhsCount != rhsCount) {
    llvm::errs() << "Error: tensor count mismatch - lhs: " << lhsCount
                 << ", rhs: " << rhsCount << "\n";
    return false;
  }

  if (lhsCount == 0 || rhsCount == 0) {
    llvm::dbgs() << "Info: no output tensors to verify\n";
    return true;
  }

  bool allMatch = true;
  std::vector<std::pair<std::string, std::string>> mismatchedTensors;

  for (size_t i = 0; i < lhsCount; ++i) {
    auto &lhsTensor = lhs->getOutputs()[i];
    auto &rhsTensor = rhs->getOutputs()[i];

    const std::string &name = lhsTensor.first;

    llvm::dbgs() << "Verifying tensor: " << name << "\n";

    try {
      auto lhsOpResult = lhsTensor.second;
      auto rhsOpResult = rhsTensor.second;
      auto lhsTensorType = llvm::dyn_cast<atir::TensorType>(lhsOpResult.getType());
      auto rhsTensorType = llvm::dyn_cast<atir::TensorType>(rhsOpResult.getType());
      if (!lhsTensorType || !rhsTensorType) {
        llvm::errs() << "Error: Cannot cast to TensorType for tensor " << name
                     << "\n";
        allMatch = false;
        mismatchedTensors.emplace_back(name, "Cannot cast to TensorType");
        continue;
      }

      auto lhsAttr = lhsTensorType.getCacheData();
      auto rhsAttr = rhsTensorType.getCacheData();
      if (!lhsAttr || !rhsAttr) {
        llvm::errs() << "Warning: No cache data for tensor " << name << "\n";
        continue;
      }

      // Check shape match before comparing values
      auto lhsRankedType = llvm::dyn_cast<mlir::RankedTensorType>(lhsAttr.getType());
      auto rhsRankedType = llvm::dyn_cast<mlir::RankedTensorType>(rhsAttr.getType());
      if (lhsRankedType && rhsRankedType &&
          lhsRankedType.getShape() != rhsRankedType.getShape()) {
        std::string lhsShapeStr, rhsShapeStr;
        auto lhsShape = lhsRankedType.getShape();
        auto rhsShape = rhsRankedType.getShape();
        for (auto d : lhsShape) { lhsShapeStr += std::to_string(d) + "x"; }
        for (auto d : rhsShape) { rhsShapeStr += std::to_string(d) + "x"; }
        if (!lhsShapeStr.empty()) lhsShapeStr.pop_back();
        if (!rhsShapeStr.empty()) rhsShapeStr.pop_back();
        llvm::errs() << "Error: Shape mismatch for tensor " << name
                     << " (lhs: " << lhsShapeStr << ", rhs: " << rhsShapeStr << ")\n";
        allMatch = false;
        mismatchedTensors.emplace_back(name, "Shape mismatch: " + lhsShapeStr + " vs " + rhsShapeStr);
        continue;
      }

      if (lhsAttr == rhsAttr) {
        llvm::dbgs() << "Info: tensor " << name << " matches\n";
        continue;
      } else {
        llvm::dbgs() << "Warning: tensor " << name << " does not match\n";
        llvm::dbgs() << "  LHS data for " << name << ": ";
        lhsAttr.dump();
        llvm::dbgs() << "  RHS data for " << name << ": ";
        rhsAttr.dump();

        if (auto lhsDense = llvm::dyn_cast<mlir::DenseElementsAttr>(lhsAttr)) {
          if (auto rhsDense = llvm::dyn_cast<mlir::DenseElementsAttr>(rhsAttr)) {
            double maxDiff = calculateMaxDiff(lhsDense, rhsDense, tolerance);
            llvm::dbgs() << "  Max difference: " << maxDiff << "\n";

            if (maxDiff > tolerance) {
              llvm::errs()
                  << "Error: Significant difference detected in tensor " << name
                  << " (threshold: " << tolerance << ", actual: " << maxDiff
                  << ")\n";
              allMatch = false;
              mismatchedTensors.emplace_back(
                  name, "Value difference > " + std::to_string(tolerance));
            } else {
              llvm::dbgs() << "Info: Differences within tolerance for tensor "
                           << name << " (max diff: " << maxDiff
                           << " <= " << tolerance << ")\n";
            }
          }
        }
      }
    } catch (const std::exception &e) {
      llvm::errs() << "Error during verification of tensor " << name << ": "
                   << e.what() << "\n";
      allMatch = false;
      mismatchedTensors.emplace_back(name,
                                     std::string("Exception: ") + e.what());
    }
  }

  saveVerificationResult("tensor_verification_result.txt", lhsCount, allMatch,
                         mismatchedTensors);
  if (allMatch) {
    llvm::outs() << "✓ Tensor verification PASSED - All tensors match\n";
  } else {
    llvm::outs() << "✗ Tensor verification FAILED - " << mismatchedTensors.size()
                 << " tensors do not match\n";
    for (const auto &[name, reason] : mismatchedTensors) {
      llvm::outs() << "  - " << name << ": " << reason << "\n";
    }
  }

  return allMatch;
}

std::string sortAndSelectBestKernel(
    const std::vector<PerformanceMetrics> &allMetrics,
    const std::vector<std::string> &verifiedKernels,
    const std::string &outputFilename) {
  
  // Filter kernels that passed verification
  std::set<std::string> verifiedSet(verifiedKernels.begin(), verifiedKernels.end());
  std::vector<PerformanceMetrics> verifiedMetrics;
  
  for (const auto &m : allMetrics) {
    if (verifiedSet.count(m.kernelPath)) {
      verifiedMetrics.push_back(m);
    }
  }
  
  if (verifiedMetrics.empty()) {
    llvm::errs() << "Warning: No kernels passed verification\n";
    return "";
  }
  
  // Sort kernels by performance
  auto sortedPaths = sortKernelsByPerformance(verifiedMetrics);
  
  if (sortedPaths.empty()) {
    return "";
  }
  
  // Save ranking results to file
  std::ofstream outFile(outputFilename);
  if (outFile.is_open()) {
    outFile << "Kernel Ranking Result\n";
    outFile << "====================\n";
    outFile << "Timestamp: " << getCurrentTimestamp() << "\n";
    outFile << "Total verified kernels: " << verifiedMetrics.size() << "\n\n";
    
    for (size_t i = 0; i < sortedPaths.size(); ++i) {
      outFile << "Rank " << (i + 1) << ": " << sortedPaths[i] << "\n";
      
      // Find corresponding metrics
      for (const auto &m : verifiedMetrics) {
        if (m.kernelPath == sortedPaths[i]) {
          outFile << "  - Exec time: " << m.execTimeMs << " ms\n";
          break;
        }
      }
    }
    
    outFile << "\nBest kernel: " << sortedPaths[0] << "\n";
    outFile.close();
    llvm::dbgs() << "Kernel ranking saved to " << outputFilename << "\n";
  }
  
  llvm::outs() << "\n✓ Best kernel selected: " << sortedPaths[0] << "\n";
  return sortedPaths[0];
}

} // namespace atir
