#ifndef DIALECT_ATIR_OPVERIFY_COMPARE_H
#define DIALECT_ATIR_OPVERIFY_COMPARE_H

#include "Dialect/Atir/OpVerify/IoDef.h"
#include "Dialect/Atir/OpVerify/PerformanceMetrics.h"
#include "mlir/IR/Attributes.h"
#include <string>
#include <vector>
#include <utility>

namespace atir {

/// Calculate the maximum difference between two DenseElementsAttr
double calculateMaxDiff(mlir::DenseElementsAttr lhs, 
                        mlir::DenseElementsAttr rhs, 
                        double tolerance);

/// Verify tensors between two IoTensorDef instances
bool verifyTensor(IoTensorDef *lhs, IoTensorDef *rhs, double tolerance = 1e-6);

/// Save verification results to a text file
void saveVerificationResult(
    const std::string &filename, 
    size_t tensorCount, 
    bool passed,
    const std::vector<std::pair<std::string, std::string>> &mismatchedTensors);

/// Sort kernels by performance and select best one
std::string sortAndSelectBestKernel(
    const std::vector<PerformanceMetrics> &allMetrics,
    const std::vector<std::string> &verifiedKernels,
    const std::string &outputFilename = "kernel_ranking_result.txt");

} // namespace atir

#endif // DIALECT_ATIR_OPVERIFY_COMPARE_H
