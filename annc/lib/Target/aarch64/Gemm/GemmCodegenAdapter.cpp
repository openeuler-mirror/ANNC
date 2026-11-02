#include "Target/aarch64/Gemm/GemmCodegenAdapter.h"

#include <cstring>
#include <limits>
#include <memory>

#ifndef ANNC_GEMM_PACKAGE_AVAILABLE
#define ANNC_GEMM_PACKAGE_AVAILABLE 0
#endif

#if ANNC_GEMM_PACKAGE_AVAILABLE
#include "annc_gemm_codegen.h"
static_assert(ANNC_GEMM_CODEGEN_API_VERSION == 1, "unsupported codegen API");
static_assert(ANNC_GEMM_CODEGEN_MAX_INPUT_SLOTS == 8,
              "codegen input slots must match the existing fused call ABI");
#endif

namespace annc::aarch64::gemm::codegen {
namespace {
#if ANNC_GEMM_PACKAGE_AVAILABLE
using Context = std::unique_ptr<AnncGemmCodegenContext,
                                decltype(&annc_gemm_codegen_destroy_context)>;
Context context(std::string &error) {
  AnncGemmCodegenContext *raw = nullptr;
  if (annc_gemm_codegen_create_context(&raw) != ANNC_GEMM_CODEGEN_OK)
    error = "cannot create GEMM codegen context";
  return Context(raw, annc_gemm_codegen_destroy_context);
}
bool check(AnncGemmCodegenStatus status, AnncGemmCodegenContext *ctx,
           std::string &error) {
  if (status == ANNC_GEMM_CODEGEN_OK) return true;
  error = annc_gemm_codegen_get_diagnostic(ctx);
  if (error.empty())
    error = "GEMM codegen failed: status " + std::to_string(status);
  return false;
}
uint32_t family(Family value) {
  switch (value) {
    case Family::NeonPacked:
      return ANNC_GEMM_FAMILY_NEON_PACKED_F32;
    case Family::NeonDirect:
      return ANNC_GEMM_FAMILY_NEON_DIRECT_F32;
    case Family::SvePacked:
      return ANNC_GEMM_FAMILY_SVE_PACKED_F32;
    case Family::SveDirect:
      return ANNC_GEMM_FAMILY_SVE_DIRECT_F32;
    case Family::NeonMatvec:
      return ANNC_GEMM_FAMILY_NEON_MATVEC_F32;
  }
  return 0;
}
uint32_t opcode(Opcode value) {
  switch (value) {
    case Opcode::Add:
      return ANNC_GEMM_EPILOGUE_ADD;
    case Opcode::Mul:
      return ANNC_GEMM_EPILOGUE_MUL;
    case Opcode::Relu:
      return ANNC_GEMM_EPILOGUE_RELU;
    case Opcode::Sigmoid:
      return ANNC_GEMM_EPILOGUE_SIGMOID;
  }
  return 0;
}
uint32_t broadcast(Broadcast value) {
  switch (value) {
    case Broadcast::None:
      return ANNC_GEMM_BROADCAST_NONE;
    case Broadcast::Scalar:
      return ANNC_GEMM_BROADCAST_SCALAR;
    case Broadcast::N:
      return ANNC_GEMM_BROADCAST_N;
    case Broadcast::M:
      return ANNC_GEMM_BROADCAST_M;
    case Broadcast::Matrix:
      return ANNC_GEMM_BROADCAST_MATRIX;
  }
  return UINT32_MAX;
}
AnncGemmKernelDesc convert(const KernelDesc &input) {
  return {sizeof(AnncGemmKernelDesc),
          ANNC_GEMM_CODEGEN_API_VERSION,
          ANNC_GEMM_TARGET_AARCH64_LINUX,
          family(input.family),
          ANNC_GEMM_DTYPE_F32,
          ANNC_GEMM_CODEGEN_KERNEL_ABI_VERSION,
          ANNC_GEMM_CODEGEN_PACKING_ABI_VERSION,
          input.m,
          input.n,
          input.k,
          input.accumulate ? ANNC_GEMM_ACCUMULATE : ANNC_GEMM_OVERWRITE,
          input.vectorLengthBytes};
}
bool convert(const std::vector<Step> &input,
             std::vector<AnncGemmEpilogueStep> &storage,
             AnncGemmEpilogueDesc &result, std::string &error) {
  if (input.size() > std::numeric_limits<uint32_t>::max()) {
    error = "epilogue candidate has too many steps";
    return false;
  }
  uint32_t slot = 0;
  for (const Step &step : input) {
    const bool binary =
        step.opcode == Opcode::Add || step.opcode == Opcode::Mul;
    uint32_t bits = 0;
    if (step.reluLimit) std::memcpy(&bits, &*step.reluLimit, sizeof(bits));
    storage.push_back(
        {sizeof(AnncGemmEpilogueStep), opcode(step.opcode),
         broadcast(step.broadcast),
         binary ? slot++ : ANNC_GEMM_CODEGEN_NO_INPUT,
         step.reluLimit ? uint32_t(ANNC_GEMM_EPILOGUE_HAS_LIMIT) : 0u, bits});
  }
  result = {
      sizeof(AnncGemmEpilogueDesc),          ANNC_GEMM_CODEGEN_API_VERSION,
      ANNC_GEMM_CODEGEN_NUMERICAL_VERSION,   slot,
      static_cast<uint32_t>(storage.size()), storage.data()};
  return true;
}
std::string hex(const uint8_t *bytes, size_t size) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result(size * 2, '0');
  for (size_t i = 0; i < size; ++i) {
    result[i * 2] = digits[bytes[i] >> 4];
    result[i * 2 + 1] = digits[bytes[i] & 15];
  }
  return result;
}
#else
bool unavailable(std::string &error) {
  error =
      "GEMM epilogue codegen component is unavailable; install a compatible "
      "binary package in third_party/annc-gemm-kernels and rebuild ANNC";
  return false;
}
#endif
}  // namespace

bool isAvailable() { return ANNC_GEMM_PACKAGE_AVAILABLE; }

bool prepare(const KernelDesc &kernel, const std::vector<Step> &steps,
             PreparedKernel &result, std::string &error) {
  result = {};
  error.clear();
#if ANNC_GEMM_PACKAGE_AVAILABLE
  auto ctx = context(error);
  if (!ctx) return false;
  auto desc = convert(kernel);
  std::vector<AnncGemmEpilogueStep> storage;
  AnncGemmEpilogueDesc candidate;
  if (!convert(steps, storage, candidate, error)) return false;
  AnncGemmCodegenAnalysis analysis{};
  analysis.struct_size = sizeof(analysis);
  AnncGemmCodegenPrepared *raw = nullptr;
  if (!check(annc_gemm_codegen_prepare(ctx.get(), &desc, &candidate, &analysis,
                                       &raw),
             ctx.get(), error))
    return false;
  std::unique_ptr<AnncGemmCodegenPrepared,
                  decltype(&annc_gemm_codegen_destroy_prepared)>
      prepared(raw, annc_gemm_codegen_destroy_prepared);
  if (analysis.supported_prefix_length > steps.size() ||
      (analysis.supported_prefix_length == 0) != (raw == nullptr)) {
    error = "GEMM codegen returned an invalid prefix result";
    return false;
  }
  result.supportedPrefixLength = analysis.supported_prefix_length;
  if (!raw) return true;
  const auto *info = annc_gemm_codegen_get_prepared_info(raw);
  if (!info || !info->normalized_request_size || !info->normalized_request ||
      !info->entry_symbol || !info->artifact_key || !info->build_id ||
      info->kernel_abi_version != ANNC_GEMM_CODEGEN_KERNEL_ABI_VERSION) {
    error = "GEMM codegen returned invalid prepared metadata";
    return false;
  }
  result.normalizedRequest =
      hex(info->normalized_request, info->normalized_request_size);
  result.entrySymbol = info->entry_symbol;
  result.artifactKey = info->artifact_key;
  result.buildId = info->build_id;
  return true;
#else
  return unavailable(error);
#endif
}

bool compileSerializedRequest(const std::string &requestHex,
                              const std::string &expectedBuildId,
                              CompiledObject &result, std::string &error) {
  result = {};
  error.clear();
#if ANNC_GEMM_PACKAGE_AVAILABLE
  if (requestHex.empty() || requestHex.size() % 2 || expectedBuildId.empty()) {
    error = "invalid serialized GEMM codegen request or build identity";
    return false;
  }
  std::vector<uint8_t> bytes(requestHex.size() / 2);
  auto digit = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
  };
  for (size_t i = 0; i < bytes.size(); ++i) {
    int high = digit(requestHex[2 * i]), low = digit(requestHex[2 * i + 1]);
    if (high < 0 || low < 0) {
      error = "normalized GEMM codegen request must be lowercase hexadecimal";
      return false;
    }
    bytes[i] = static_cast<uint8_t>((high << 4) | low);
  }
  auto ctx = context(error);
  if (!ctx) return false;
  AnncGemmCodegenObject *raw = nullptr;
  if (!check(annc_gemm_codegen_compile(ctx.get(), bytes.data(), bytes.size(),
                                       expectedBuildId.c_str(), &raw),
             ctx.get(), error))
    return false;
  std::unique_ptr<AnncGemmCodegenObject,
                  decltype(&annc_gemm_codegen_destroy_object)>
      object(raw, annc_gemm_codegen_destroy_object);
  const auto *info = annc_gemm_codegen_get_object_info(raw);
  if (!info || !info->object_bytes || !info->object_size || !info->build_id ||
      expectedBuildId != info->build_id || !info->artifact_key ||
      info->kernel_abi_version != ANNC_GEMM_CODEGEN_KERNEL_ABI_VERSION ||
      !info->defined_symbol_count || !info->defined_symbols ||
      (info->dependency_symbol_count && !info->dependency_symbols)) {
    error = "GEMM codegen returned invalid object metadata";
    return false;
  }
  result.bytes.assign(info->object_bytes,
                      info->object_bytes + info->object_size);
  result.artifactKey = info->artifact_key;
  for (size_t i = 0; i < info->defined_symbol_count; ++i)
    result.definedSymbols.emplace_back(info->defined_symbols[i]);
  for (size_t i = 0; i < info->dependency_symbol_count; ++i)
    result.dependencySymbols.emplace_back(info->dependency_symbols[i]);
  return true;
#else
  return unavailable(error);
#endif
}
}  // namespace annc::aarch64::gemm::codegen
