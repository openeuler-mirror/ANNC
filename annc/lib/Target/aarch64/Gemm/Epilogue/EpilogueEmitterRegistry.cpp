#include <algorithm>

#include "../GemmPlan.h"
#include "Target/aarch64/Gemm/Epilogue/EpilogueEmitter.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"

namespace annc::aarch64::gemm::epilogue {
namespace {
std::string vectorRegister(bool sve, unsigned index) {
  return (llvm::Twine(sve ? "z" : "v") + llvm::utostr(index) +
          (sve ? ".s" : ".4s"))
      .str();
}
std::string predicate(const EmitRequest &request, bool sve, unsigned group) {
  return sve && request.binding.predicateRegister
             ? "p" + llvm::utostr(*request.binding.predicateRegister + group) +
                   "/m, "
             : "";
}
void replaceAll(std::string &text, llvm::StringRef from, llvm::StringRef to) {
  for (size_t offset = 0;
       (offset = text.find(from.str(), offset)) != std::string::npos;
       offset += to.size())
    text.replace(offset, from.size(), to.str());
}
}  // namespace
mlir::FailureOr<EmissionResult> emitBinaryCommon(const EmitRequest &request,
                                                 llvm::StringRef mnemonic,
                                                 bool sve) {
  const unsigned groups = std::max(1u, request.binding.nGroups);
  if (request.binding.accumulatorRegisters.empty() ||
      request.binding.scratchRegisters.size() < groups ||
      (!sve && request.binding.predicateRegister))
    return mlir::failure();
  EmissionResult result;
  result.reserve(request.binding.accumulatorRegisters.size());
  for (unsigned index = 0; index < request.binding.accumulatorRegisters.size();
       ++index) {
    const unsigned group = index % groups;
    const std::string acc =
        vectorRegister(sve, request.binding.accumulatorRegisters[index]);
    const std::string pred = predicate(request, sve, group);
    result.push_back(
        mnemonic.str() + " " + acc + ", " + pred + acc + ", " +
        vectorRegister(sve, request.binding.scratchRegisters[group]));
  }
  return result;
}
mlir::FailureOr<EmissionResult> emitReluCommon(const EmitRequest &request,
                                               bool sve) {
  if (request.opcode != atir::EpilogueOpcode::kRelu ||
      request.binding.accumulatorRegisters.empty() ||
      request.binding.scratchRegisters.empty() ||
      (sve && !request.binding.predicateRegister) ||
      (!sve && request.binding.predicateRegister))
    return mlir::failure();
  EmissionResult result;
  const unsigned zero = request.binding.scratchRegisters[0];
  result.push_back((llvm::Twine(sve ? "mov z" : "movi v") + llvm::utostr(zero) +
                    (sve ? ".b, #0" : ".4s, #0"))
                       .str());
  const std::string zeroReg = vectorRegister(sve, zero);
  const unsigned groups = std::max(1u, request.binding.nGroups);
  auto emitClamp = [&](llvm::StringRef mnemonic) {
    for (unsigned index = 0;
         index < request.binding.accumulatorRegisters.size(); ++index) {
      const std::string acc =
          vectorRegister(sve, request.binding.accumulatorRegisters[index]);
      result.push_back(mnemonic.str() + " " + acc + ", " +
                       predicate(request, sve, index % groups) + acc + ", " +
                       zeroReg);
    }
  };
  emitClamp("fmaxnm");
  if (request.reluLimit && *request.reluLimit >= 0.0f) {
    uint64_t bits =
        llvm::APFloat(*request.reluLimit).bitcastToAPInt().getZExtValue();
    result.push_back("movz w9, #" + llvm::utostr(bits & 0xffff));
    result.push_back("movk w9, #" + llvm::utostr(bits >> 16) + ", lsl #16");
    result.push_back("dup " + zeroReg + ", w9");
    emitClamp("fminnm");
  }
  return result;
}
std::string remapScratchRegisters(llvm::StringRef instruction,
                                  llvm::ArrayRef<unsigned> registers,
                                  llvm::StringRef prefix) {
  std::string mapped = instruction.str();
  for (unsigned slot = 0; slot < registers.size(); ++slot)
    replaceAll(mapped, prefix.str() + llvm::utostr(slot),
               "@scratch" + llvm::utostr(slot) + "@");
  for (unsigned slot = 0; slot < registers.size(); ++slot)
    replaceAll(mapped, "@scratch" + llvm::utostr(slot) + "@",
               prefix.str() + llvm::utostr(registers[slot]));
  return mapped;
}
mlir::FailureOr<EmissionResult> emitSveEpilogue(const EmitRequest &request);
mlir::FailureOr<EmissionResult> emitNeonEpilogue(const EmitRequest &request);

mlir::FailureOr<StepResourceRequirement>
EpilogueEmitterRegistry::requirementFor(const atir::EpilogueStep &step) const {
  for (const Entry &entry : emitters_)
    if (entry.opcode == step.opcode) return entry.requirement;
  return mlir::failure();
}
mlir::FailureOr<EmissionResult> EpilogueEmitterRegistry::emit(
    const EmitRequest &request) const {
  for (const Entry &entry : emitters_)
    if (entry.opcode == request.opcode) return entry.emit(request);
  return mlir::failure();
}
void EpilogueEmitterRegistry::add(atir::EpilogueOpcode opcode,
                                  StepResourceRequirement requirement,
                                  EmitFunction emit) {
  emitters_.push_back({opcode, requirement, emit});
}
EpilogueEmitterRegistry createRegistry(
    EpilogueEmitterRegistry::EmitFunction emit, bool sve) {
  EpilogueEmitterRegistry registry;
  registry.add(atir::EpilogueOpcode::kAdd, {1, 0, 1}, emit);
  registry.add(atir::EpilogueOpcode::kMul, {1, 0, 1}, emit);
  registry.add(atir::EpilogueOpcode::kRelu, {1, 0, 1}, emit);
  registry.add(atir::EpilogueOpcode::kSigmoid, {8, sve ? 1u : 0u, 1}, emit);
  return registry;
}
const EpilogueEmitterRegistry &getEpilogueEmitterRegistry(GemmIsa isa) {
  static const EpilogueEmitterRegistry registries[] = {
      createRegistry(emitSveEpilogue, true),
      createRegistry(emitNeonEpilogue, false)};
  return registries[isa == GemmIsa::kSve ? 0 : 1];
}
mlir::FailureOr<unsigned> selectFusiblePrefix(
    const atir::EpilogueProgram &program, const GemmPlan &plan,
    const EpilogueEmitterRegistry &registry) {
  if (program.steps.empty() || plan.version != kPlanVersion ||
      plan.kernelTile.mr < 1 || plan.kernelTile.mr > 6 ||
      plan.kernelTile.panelLanes < 1 || plan.kernelTile.panelLanes > 4)
    return mlir::failure();
  const unsigned groups =
      static_cast<unsigned>(plan.kernelTile.panelLanes);
  const unsigned accumulators =
      static_cast<unsigned>(plan.kernelTile.mr) * groups;
  const unsigned maxPredicates = plan.isa == GemmIsa::kSve ? 8 : 0;
  const unsigned storePredicates = plan.isa == GemmIsa::kSve ? groups : 0;
  unsigned count = 0, inputSlots = 0, liveVectors = 0, livePredicates = 0,
           liveScalars = 0;
  for (const atir::EpilogueStep &step : program.steps) {
    if (atir::isBinaryEpilogueOpcode(step.opcode) &&
        (step.broadcast == atir::BroadcastKind::kM ||
         step.broadcast == atir::BroadcastKind::kMatrix))
      break;
    auto requirement = registry.requirementFor(step);
    if (mlir::failed(requirement)) break;
    const unsigned inputs = atir::isBinaryEpilogueOpcode(step.opcode) ? 1 : 0;
    if (inputs > kEpilogueArgsPointerSlots - inputSlots) break;
    liveVectors =
        std::max(liveVectors, atir::isBinaryEpilogueOpcode(step.opcode)
                                  ? groups * inputs
                                  : requirement->vectorFragments);
    livePredicates = std::max(livePredicates, requirement->predicateFragments);
    liveScalars = std::max(liveScalars, requirement->scalarRegisters);
    if (accumulators + liveVectors > 32 ||
        storePredicates + livePredicates > maxPredicates || liveScalars > 1)
      break;
    inputSlots += inputs;
    ++count;
  }
  return count;
}

}  // namespace annc::aarch64::gemm::epilogue
