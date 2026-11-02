#ifndef ANNC_TARGET_AARCH64_GEMM_CODEGEN_ADAPTER_H
#define ANNC_TARGET_AARCH64_GEMM_CODEGEN_ADAPTER_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace annc::aarch64::gemm::codegen {

// Public semantic inputs only. No register, template, or K-unroll knowledge.
enum class Family { NeonPacked, NeonDirect, SvePacked, SveDirect, NeonMatvec };
enum class Opcode { Add, Mul, Relu, Sigmoid };
enum class Broadcast { None, Scalar, N, M, Matrix };
struct KernelDesc {
  Family family = Family::NeonPacked;
  uint32_t m = 0, n = 0, k = 0, vectorLengthBytes = 16;
  bool accumulate = false;
};
struct Step {
  Opcode opcode = Opcode::Relu;
  Broadcast broadcast = Broadcast::None;
  std::optional<float> reluLimit;
};
struct PreparedKernel {
  unsigned supportedPrefixLength = 0;
  std::string normalizedRequest, entrySymbol, artifactKey, buildId;
};
struct CompiledObject {
  std::vector<uint8_t> bytes;
  std::vector<std::string> definedSymbols, dependencySymbols;
  std::string artifactKey;
};

bool isAvailable();
bool prepare(const KernelDesc &kernel, const std::vector<Step> &steps,
             PreparedKernel &prepared, std::string &error);
bool compileSerializedRequest(const std::string &requestHex,
                              const std::string &expectedBuildId,
                              CompiledObject &object, std::string &error);

}  // namespace annc::aarch64::gemm::codegen
#endif
