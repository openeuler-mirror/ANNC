#include "Target/aarch64/Gemm/Epilogue/EpilogueEmitter.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace annc::aarch64::gemm::epilogue {
namespace {
std::string vRegister(unsigned index) {
  return (llvm::Twine("v") + llvm::utostr(index) + ".4s").str();
}

std::string registerName(llvm::StringRef prefix, unsigned index,
                         llvm::StringRef suffix) {
  return (llvm::Twine(prefix) + llvm::utostr(index) + suffix).str();
}

mlir::FailureOr<EmissionResult> emitNeonSigmoid(const EmitRequest &request) {
  if (request.opcode != atir::EpilogueOpcode::kSigmoid ||
      request.binding.accumulatorRegisters.empty() ||
      request.binding.scratchRegisters.size() < 8)
    return mlir::failure();
  EmissionResult result;
  const auto scratch = [&](unsigned slot) {
    return vRegister(request.binding.scratchRegisters[slot]);
  };
  const auto scratchBytes = [&](unsigned slot) {
    return registerName("v", request.binding.scratchRegisters[slot], ".16b");
  };
  for (unsigned accumulator : request.binding.accumulatorRegisters) {
    const std::string acc = vRegister(accumulator);
    const std::string accBytes = registerName("v", accumulator, ".16b");
    result.push_back("mov " + scratchBytes(4) + ", " + accBytes);
    result.push_back("movz w9, #0x0000");
    result.push_back("movk w9, #0xC2B0, lsl #16");
    result.push_back("dup " + scratch(5) + ", w9");
    result.push_back("fmaxnm " + acc + ", " + acc + ", " + scratch(5));
    result.push_back("movz w9, #0x0000");
    result.push_back("movk w9, #0x42B0, lsl #16");
    result.push_back("dup " + scratch(5) + ", w9");
    result.push_back("fminnm " + acc + ", " + acc + ", " + scratch(5));
    const char *sequence[] = {
        "fneg v1.4s, ",
        "movz w9, #0xAA3B",
        "movk w9, #0x3FB8, lsl #16",
        "dup v0.4s, w9",
        "fmul v1.4s, v1.4s, v0.4s",
        "frinta v2.4s, v1.4s",
        "fsub v3.4s, v1.4s, v2.4s",
        "movz w9, #0x8F4E",
        "movk w9, #0x3C1D, lsl #16",
        "dup v5.4s, w9",
        "movz w9, #0x5847",
        "movk w9, #0x3D63, lsl #16",
        "dup v7.4s, w9",
        "fmla v7.4s, v3.4s, v5.4s",
        "movz w9, #0xFDF0",
        "movk w9, #0x3E75, lsl #16",
        "dup v5.4s, w9",
        "fmla v5.4s, v3.4s, v7.4s",
        "movz w9, #0x7218",
        "movk w9, #0x3F31, lsl #16",
        "dup v7.4s, w9",
        "fmla v7.4s, v3.4s, v5.4s",
        "fmov v5.4s, #1.0",
        "fmla v5.4s, v3.4s, v7.4s",
        "fcvtzs v2.4s, v2.4s",
        "movi v7.4s, #127",
        "add v7.4s, v2.4s, v7.4s",
        "shl v7.4s, v7.4s, #23",
        "fmul v5.4s, v5.4s, v7.4s",
        "fmov v7.4s, #1.0",
        "fadd v7.4s, v5.4s, v7.4s",
        "fmov v6.4s, #1.0",
        "fdiv v6.4s, v6.4s, v7.4s",
    };
    result.push_back(remapScratchRegisters(
        sequence[0] + acc, request.binding.scratchRegisters, "v"));
    for (unsigned i = 1; i < sizeof(sequence) / sizeof(sequence[0]); ++i)
      result.push_back(remapScratchRegisters(
          sequence[i], request.binding.scratchRegisters, "v"));
    result.push_back("mov " + accBytes + ", " + scratchBytes(6));
    result.push_back("movi " + scratch(5) + ", #0xff");
    result.push_back("shl " + scratch(5) + ", " + scratch(5) + ", #23");
    result.push_back("fneg " + scratch(5) + ", " + scratch(5));
    result.push_back("fcmeq " + scratch(6) + ", " + scratch(4) + ", " +
                     scratch(5));
    result.push_back("fcmeq " + scratch(7) + ", " + scratch(4) + ", " +
                     scratch(4));
    result.push_back("movi " + scratch(5) + ", #0");
    result.push_back("fdiv " + scratch(5) + ", " + scratch(5) + ", " +
                     scratch(5));
    result.push_back("bif " + accBytes + ", " + scratchBytes(5) + ", " +
                     scratchBytes(7));
    result.push_back("movi " + scratch(5) + ", #0");
    result.push_back("bit " + accBytes + ", " + scratchBytes(5) + ", " +
                     scratchBytes(6));
  }
  return result;
}
}  // namespace

mlir::FailureOr<EmissionResult> emitNeonEpilogue(const EmitRequest &request) {
  switch (request.opcode) {
    case atir::EpilogueOpcode::kAdd:
      return emitBinaryCommon(request, "fadd", false);
    case atir::EpilogueOpcode::kMul:
      return emitBinaryCommon(request, "fmul", false);
    case atir::EpilogueOpcode::kRelu:
      return emitReluCommon(request, false);
    case atir::EpilogueOpcode::kSigmoid:
      return emitNeonSigmoid(request);
  }
  return mlir::failure();
}
}  // namespace annc::aarch64::gemm::epilogue
