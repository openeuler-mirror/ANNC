#include "Dialect/Atir/OpVerify/OpVerify.h"

#include <filesystem>
#include <map>
#include <vector>

#include "Dialect/Atir/OpVerify/IoDef.h"
#include "Dialect/Atir/OpVerify/PerformanceMetrics.h"
#include "Dialect/Atir/OpVerify/compare.h"
#include "Dialect/Atir/OpVerify/runKpGenOp.h"
#include "Dialect/Atir/OpVerify/runOriGraph.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/PassManager.h"

using namespace llvm;
using namespace mlir;

namespace atir {

// Global warmup configuration (can be moved to Pass options later)
static const WarmupConfig kDefaultWarmupConfig;

class AtirOpVerifyPass : public AtirOpVerifyBase<AtirOpVerifyPass> {
 public:
  AtirOpVerifyPass() = default;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    IoTensorDef inputDef;
    IoTensorDef oriOutput;

    // Parse the generated-kernel paths.
    std::string kpRawValue = kpGenLibPath.getValue();
    std::string kpPathValue = kpRawValue;

    namespace fs = std::filesystem;

    // Parse library paths (both support comma-separated multiple paths)
    auto parseLibPaths =
        [](const std::string& pathStr) -> std::vector<std::string> {
      std::vector<std::string> paths;
      if (pathStr.empty()) return paths;

      size_t start = 0;
      size_t end = pathStr.find(',');
      while (end != std::string::npos) {
        std::string path = pathStr.substr(start, end - start);
        // Trim leading and trailing whitespace
        size_t first = path.find_first_not_of(" \t");
        size_t last = path.find_last_not_of(" \t");
        if (first != std::string::npos && last != std::string::npos) {
          paths.push_back(path.substr(first, last - first + 1));
        }
        start = end + 1;
        end = pathStr.find(',', start);
      }
      // Add the last path
      std::string path = pathStr.substr(start);
      size_t first = path.find_first_not_of(" \t");
      size_t last = path.find_last_not_of(" \t");
      if (first != std::string::npos && last != std::string::npos) {
        paths.push_back(path.substr(first, last - first + 1));
      }
      return paths;
    };

    std::vector<std::string> kpLibPaths = parseLibPaths(kpPathValue);
    if (kpLibPaths.empty()) {
      emitError(getOperation().getLoc(),
                "Missing or Wrong option: kpGenLibPath must be specified.\n"
                "Usage:\n"
                "  "
                "--atir-op-verify=\"kpGenLibPath=/path/to/lib1.so,/path/to/"
                "lib2.so\"");
      signalPassFailure();
      return;
    }

    // Validate all paths exist
    for (const auto& libPath : kpLibPaths) {
      if (!fs::exists(libPath)) {
        emitError(getOperation().getLoc(),
                  "Library path does not exist: " + libPath);
        signalPassFailure();
        return;
      }
    }
    // Get function name from the first KP library.
    std::string firstPath = kpLibPaths[0];
    std::string funcName = getFuncNameFromPath(firstPath);
    if (funcName.empty()) {
      emitError(
          getOperation().getLoc(),
          "Failed to extract function name from library path: " + firstPath);
      signalPassFailure();
      return;
    }

    // Data layer - create input/output definitions
    IoTensorDef outputDef;
    createIoDef(moduleOp, &inputDef, &outputDef, funcName);

    // Execute original subgraph
    runOriGraph(moduleOp, funcName, &inputDef, &oriOutput);

    // Collect all kernels for comparison
    struct KernelResult {
      std::string path;
      IoTensorDef output;
      PerformanceMetrics metrics;
      bool execSuccess;
      bool verificationPassed;
      bool strictVerificationPassed;
    };

    std::vector<KernelResult> allResults;

    // Run KP libraries
    for (size_t i = 0; i < kpLibPaths.size(); ++i) {
      const auto& kpPath = kpLibPaths[i];
      llvm::outs() << "\n[" << (i + 1) << "/" << kpLibPaths.size()
                   << "] Running KP Gen OP: " << kpPath << "\n";

      KernelResult result;
      result.path = kpPath;
      result.execSuccess = false;
      result.verificationPassed = false;
      result.strictVerificationPassed = false;

      IoTensorDef kpGenOutput;
      bool execSuccess = runKpGenOp(kpPath, &inputDef, &outputDef, &kpGenOutput,
                                    result.metrics, kDefaultWarmupConfig);

      if (!execSuccess) {
        llvm::errs() << "✗ KP backend execution failed for: " << kpPath << "\n";
        allResults.push_back(result);
        continue;
      }

      result.output = std::move(kpGenOutput);
      result.execSuccess = true;

      // Verify against original
      result.verificationPassed =
          verifyTensor(&oriOutput, &result.output, 1e-4);
      result.strictVerificationPassed =
          verifyTensor(&oriOutput, &result.output, 1e-6);

      llvm::outs() << "  Exec time: " << result.metrics.execTimeMs << " ms\n";
      if (result.verificationPassed) {
        llvm::outs() << "  ✓ Verification PASSED (tolerance: 1e-4)\n";
        if (result.strictVerificationPassed) {
          llvm::outs() << "    Additionally passed strict verification "
                          "(tolerance: 1e-6)\n";
        }
      } else {
        llvm::outs() << "  ✗ Verification FAILED\n";
      }

      allResults.push_back(std::move(result));
    }

    // Restore return operand types
    moduleOp.walk([&](mlir::func::ReturnOp returnOp) {
      auto funcOp = returnOp->getParentOfType<mlir::func::FuncOp>();
      if (!funcOp) return;
      auto expectedTypes = funcOp.getFunctionType().getResults();
      for (auto [operand, expectedType] :
           llvm::zip(returnOp.getOperands(), expectedTypes)) {
        auto expType = llvm::dyn_cast<atir::TensorType>(expectedType);
        if (!expType) continue;
        operand.setType(expType);
      }
    });

    // Collect verified kernels for ranking
    std::vector<PerformanceMetrics> verifiedMetrics;
    std::vector<std::string> verifiedPaths;

    for (const auto& result : allResults) {
      if (result.execSuccess && result.verificationPassed) {
        verifiedMetrics.push_back(result.metrics);
        verifiedPaths.push_back(result.path);
      }
    }

    // Performance ranking
    std::string bestKernel;
    if (!verifiedMetrics.empty()) {
      bestKernel = sortAndSelectBestKernel(verifiedMetrics, verifiedPaths);
    }

    // Output overall summary
    llvm::outs() << "\n========================================\n";
    llvm::outs() << "Overall Comparison Summary\n";
    llvm::outs() << "========================================\n";
    llvm::outs() << "Total libraries executed: " << allResults.size() << "\n";
    llvm::outs() << "  - KP libraries: " << kpLibPaths.size() << "\n";

    size_t passedCount = 0;
    for (const auto& result : allResults) {
      if (result.execSuccess && result.verificationPassed) {
        passedCount++;
      }
    }

    llvm::outs() << "Passed verification: " << passedCount << "/"
                 << allResults.size() << "\n";

    if (passedCount == allResults.size()) {
      llvm::outs() << "✓ ALL verifications PASSED\n";
    } else if (passedCount > 0) {
      llvm::outs() << "⚠ SOME verifications PASSED, some FAILED\n";
    } else {
      llvm::outs() << "✗ ALL verifications FAILED\n";
    }

    if (!bestKernel.empty()) {
      llvm::outs() << "\n★ Best kernel (fastest): " << bestKernel << "\n";
    }

    llvm::outs() << "========================================\n";
    llvm::outs() << "OpVerifyPass End\n";
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAtirOpVerifyPass() {
  return std::make_unique<AtirOpVerifyPass>();
}
}  // namespace atir
