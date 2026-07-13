#ifndef ANNC_FUSION_METADATA_FUSION_METADATA_H
#define ANNC_FUSION_METADATA_FUSION_METADATA_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "llvm/Support/Error.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace annc::fusion {

struct FusionArg {
  std::string role;
  std::string tfName;
  std::vector<int64_t> shape;
  int64_t rank = -1;
  std::string dtype;
};

struct FusionInfo {
  int schemaVersion = 1;
  std::string name;
  std::string pattern;
  std::string kernelName;
  std::string outputTensor;
  std::vector<std::string> originalNodes;
  std::vector<FusionArg> args;
  std::vector<FusionArg> outputs;
  std::map<std::string, std::string> patternAttrs;

  // Legacy runtime metadata kept during migration. These fields mirror the
  // current fusion.metadata dictionary so existing GraphDef rewrite logic can
  // move to the shared library before the producer is redesigned.
  std::vector<std::string> inputs;
  std::vector<std::vector<int64_t>> inputShapes;
  std::vector<int64_t> outputShape;
  std::string abi = "mlir_ciface";
  int64_t nConstants = 0;
  int64_t nFixed = 0;
  int64_t nDynamic = 0;
  int64_t numOutputs = 1;
  std::vector<int64_t> outputRanks;
  std::vector<int64_t> inputRanks;
  std::vector<int64_t> dynamicDims;
  std::vector<int64_t> kernelArgOrder;
  std::string symbolicSignature;
  std::string fallbackFunction;
};

llvm::Error validateFusionInfo(const FusionInfo &info);

void normalizeFusionInfo(FusionInfo &info);

llvm::Expected<FusionInfo> readFusionInfo(mlir::func::FuncOp func);

llvm::Expected<std::vector<FusionInfo>> extractFusionInfos(
    mlir::ModuleOp module);

llvm::Expected<std::vector<FusionInfo>> extractFusionInfosFromMlirText(
    const std::string &path);

}  // namespace annc::fusion

#endif  // ANNC_FUSION_METADATA_FUSION_METADATA_H
