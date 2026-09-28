#include <algorithm>

#include "Target/aarch64/Gemm/Epilogue/EpilogueEmitter.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace annc::aarch64::gemm::epilogue {
namespace {
std::string zRegister(unsigned index) {
  return (llvm::Twine("z") + llvm::utostr(index) + ".s").str();
}

std::string registerName(llvm::StringRef prefix, unsigned index,
                         llvm::StringRef suffix) {
  return (llvm::Twine(prefix) + llvm::utostr(index) + suffix).str();
}

std::string withPredicate(llvm::StringRef instruction,
                          llvm::StringRef predicate) {
  std::string result = instruction.str();
  for (size_t offset = 0;
       (offset = result.find("p0", offset)) != std::string::npos;
       offset += predicate.size())
    result.replace(offset, 2, predicate);
  return result;
}

mlir::FailureOr<EmissionResult> emitSveSigmoid(const EmitRequest &request) {
  if (request.opcode != atir::EpilogueOpcode::kSigmoid ||
      request.binding.accumulatorRegisters.empty() ||
      request.binding.scratchRegisters.size() < 8)
    return mlir::failure();
  EmissionResult result;
  const auto scratch = [&](unsigned slot) {
    return zRegister(request.binding.scratchRegisters[slot]);
  };
  const auto scratchD = [&](unsigned slot) {
    std::string result = scratch(slot);
    result.replace(result.find(".s"), 2, ".d");
    return result;
  };
  const unsigned predicateBase = request.binding.predicateRegister.value_or(0);
  const unsigned nanPredicate =
      predicateBase + std::max(1u, request.binding.nGroups);
  for (unsigned index = 0; index < request.binding.accumulatorRegisters.size();
       ++index) {
    unsigned accumulator = request.binding.accumulatorRegisters[index];
    const std::string acc = zRegister(accumulator);
    std::string predicate =
        "p" + llvm::utostr(predicateBase +
                           index % std::max(1u, request.binding.nGroups));
    result.push_back(
        registerName("mov z", request.binding.scratchRegisters[4], ".d, ") +
        registerName("z", accumulator, ".d"));
    result.push_back(registerName("fcmuo p", nanPredicate, ".s, ") + predicate +
                     "/z, " + acc + ", " + acc);
    result.push_back("movz w9, #0x0000");
    result.push_back("movk w9, #0xC2B0, lsl #16");
    result.push_back("dup " + scratch(5) + ", w9");
    result.push_back("fmaxnm " + acc + ", " + predicate + "/m, " + acc + ", " +
                     scratch(5));
    result.push_back("movz w9, #0x0000");
    result.push_back("movk w9, #0x42B0, lsl #16");
    result.push_back("dup " + scratch(5) + ", w9");
    result.push_back("fminnm " + acc + ", " + predicate + "/m, " + acc + ", " +
                     scratch(5));
    const char *sequence[] = {
        "fneg z1.s, p0/m, ",
        "movz w9, #0xAA3B",
        "movk w9, #0x3FB8, lsl #16",
        "dup z0.s, w9",
        "fmul z1.s, z1.s, z0.s",
        "frinta z2.s, p0/m, z1.s",
        "fsub z3.s, z1.s, z2.s",
        "movz w9, #0x8F4E",
        "movk w9, #0x3C1D, lsl #16",
        "dup z5.s, w9",
        "movz w9, #0x5847",
        "movk w9, #0x3D63, lsl #16",
        "dup z7.s, w9",
        "fmla z7.s, p0/m, z3.s, z5.s",
        "movz w9, #0xFDF0",
        "movk w9, #0x3E75, lsl #16",
        "dup z5.s, w9",
        "fmla z5.s, p0/m, z3.s, z7.s",
        "movz w9, #0x7218",
        "movk w9, #0x3F31, lsl #16",
        "dup z7.s, w9",
        "fmla z7.s, p0/m, z3.s, z5.s",
        "fmov z5.s, #1.0",
        "fmla z5.s, p0/m, z3.s, z7.s",
        "fcvtzs z2.s, p0/m, z2.s",
        "fscale z5.s, p0/m, z5.s, z2.s",
        "fmov z7.s, #1.0",
        "fadd z7.s, z5.s, z7.s",
        "fmax z7.s, p0/m, z7.s, z7.s",
        "fmov z6.s, #1.0",
        "fdiv z6.s, p0/m, z6.s, z7.s",
    };
    result.push_back(withPredicate(
        remapScratchRegisters(sequence[0] + acc,
                              request.binding.scratchRegisters, "z"),
        predicate));
    for (unsigned i = 1; i < sizeof(sequence) / sizeof(sequence[0]); ++i)
      result.push_back(
          withPredicate(remapScratchRegisters(
                            sequence[i], request.binding.scratchRegisters, "z"),
                        predicate));
    result.push_back(registerName("mov z", accumulator, ".d, ") + scratchD(6));
    result.push_back("fmov " + scratch(5) + ", #0.0");
    result.push_back("fdiv " + scratch(5) + ", p" + llvm::utostr(nanPredicate) +
                     "/m, " + scratch(5) + ", " + scratch(5));
    result.push_back("fmul " + acc + ", p" + llvm::utostr(nanPredicate) +
                     "/m, " + acc + ", " + scratch(5));
    result.push_back("movz w9, #0x0000");
    result.push_back("movk w9, #0x7F80, lsl #16");
    result.push_back("dup " + scratch(5) + ", w9");
    result.push_back("fneg " + scratch(5) + ", " + predicate + "/m, " +
                     scratch(5));
    result.push_back("fcmeq p" + llvm::utostr(nanPredicate) + ".s, " +
                     predicate + "/z, " + scratch(4) + ", " + scratch(5));
    result.push_back("fmov " + scratch(5) + ", #0.0");
    result.push_back("mov " + acc + ", p" + llvm::utostr(nanPredicate) +
                     "/m, " + scratch(5));
  }
  return result;
}
}  // namespace

mlir::FailureOr<EmissionResult> emitSveEpilogue(const EmitRequest &request) {
  switch (request.opcode) {
    case atir::EpilogueOpcode::kAdd:
      return emitBinaryCommon(request, "fadd", true);
    case atir::EpilogueOpcode::kMul:
      return emitBinaryCommon(request, "fmul", true);
    case atir::EpilogueOpcode::kRelu:
      return emitReluCommon(request, true);
    case atir::EpilogueOpcode::kSigmoid:
      return emitSveSigmoid(request);
  }
  return mlir::failure();
}
}  // namespace annc::aarch64::gemm::epilogue
