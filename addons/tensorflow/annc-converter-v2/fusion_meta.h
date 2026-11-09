#ifndef ANNC_TOOLS_ANNC_CONVERTER_V2_FUSION_META_H
#define ANNC_TOOLS_ANNC_CONVERTER_V2_FUSION_META_H

// Standalone Fusion Metadata reader: a non-LLVM mirror of the JSON-read
// surface of annc/include/FusionMetadata/ (same fields, grammar and
// validation rules). The converter binary must never link LLVM/MLIR (TF and
// LLVM cannot share a process), hence this copy; keep it in sync with the
// canonical headers (unit tests guard against drift). JSON-only: produce
// fusion_metadata.json with annc-fusion-metadata and pass --metadata_json.

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace annc::converter_v2 {

// Minimal replacement for llvm::Error / llvm::Expected without LLVM.
struct Status {
  std::string error;
  bool ok() const { return error.empty(); }
};

template <typename T>
struct Result {
  T value{};
  std::string error;
  bool ok() const { return error.empty(); }
};

// One tensor at a fusion boundary (mirrors annc::fusion::FusionArg).
struct FusionArg {
  std::string role;
  std::string tfName;
  std::vector<int64_t> shape;
  int64_t rank = -1;
  std::string dtype;
};

// Metadata contract for one generated fusion kernel.
struct FusionInfo {
  std::string name;
  std::string pattern;
  std::string executionMode = "aot";
  std::string kernelName;
  std::string templateFingerprint;
  std::vector<FusionArg> args;
  std::vector<FusionArg> outputs;
  std::vector<std::string> memberNodeNames;
  std::map<std::string, std::string> patternAttrs;
  std::string abi = "mlir_ciface";
  std::vector<int64_t> dynamicDims;
  std::vector<int64_t> kernelArgOrder;
  // Scalar attributes in the shared context schema order (not tensor inputs);
  // annc_context_v3 only.
  std::vector<int64_t> contextI64Attrs;
  std::string symbolicSignature;
  std::string fallbackFunction;
};

// Tensor endpoint: "node", "node:port" or control edge "^node".
struct TensorEndpoint {
  std::string node;
  int64_t port = 0;
  bool control = false;

  std::string canonicalDataName() const;
};

Result<TensorEndpoint> parseTensorEndpoint(std::string_view input);

Status validateFusionInfo(const FusionInfo &info);

// Reads annc-fusion-metadata JSON: a {"fusions": [...]} object or one fusion.
Result<std::vector<FusionInfo>> readFusionMetadataJson(const std::string &path);

}  // namespace annc::converter_v2

#endif  // ANNC_TOOLS_ANNC_CONVERTER_V2_FUSION_META_H
