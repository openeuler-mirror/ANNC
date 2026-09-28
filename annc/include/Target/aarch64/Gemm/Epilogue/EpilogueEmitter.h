#ifndef ANNC_TARGET_AARCH64_GEMM_EPILOGUE_EPILOGUE_EMITTER_H
#define ANNC_TARGET_AARCH64_GEMM_EPILOGUE_EPILOGUE_EMITTER_H

#include <optional>
#include <string>

#include "Dialect/Atir/Passes/GemmEpilogueCandidate.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LogicalResult.h"
namespace annc::aarch64::gemm {
struct GemmPlan;
enum class GemmIsa;
enum class KcMode;
}  // namespace annc::aarch64::gemm

namespace annc::aarch64::gemm::epilogue {

struct StepResourceRequirement {
  unsigned vectorFragments = 0;
  unsigned predicateFragments = 0;
  unsigned scalarRegisters = 0;
};

struct FragmentRegisterBinding {
  llvm::SmallVector<unsigned> accumulatorRegisters;
  llvm::SmallVector<unsigned> scratchRegisters;
  std::optional<unsigned> predicateRegister;  // SVE tail predicate.
  unsigned nGroups = 1;
};

struct EmitRequest {
  atir::EpilogueOpcode opcode;
  FragmentRegisterBinding binding;
  std::optional<float> reluLimit;
};

using EmissionResult = llvm::SmallVector<std::string>;

class EpilogueEmitterRegistry {
 public:
  using EmitFunction = mlir::FailureOr<EmissionResult> (*)(const EmitRequest &);
  mlir::FailureOr<StepResourceRequirement> requirementFor(
      const atir::EpilogueStep &step) const;
  mlir::FailureOr<EmissionResult> emit(const EmitRequest &request) const;
  void add(atir::EpilogueOpcode opcode, StepResourceRequirement requirement,
           EmitFunction emit);

 private:
  struct Entry {
    atir::EpilogueOpcode opcode;
    StepResourceRequirement requirement;
    EmitFunction emit;
  };
  llvm::SmallVector<Entry> emitters_;
};

struct MicrokernelRequest {
  GemmIsa isa;
  int64_t mr = 0, ng = 0;
  KcMode kcMode;
  bool rowMajor = false;
  int64_t microN = 0;
  std::string epilogueSuffix;
  llvm::SmallVector<std::string> epilogueInstructions;
  int64_t kResidue = 0;
  bool kHasGroups = false;
  bool matrixVector = false;
};

struct GeneratedMicrokernel {
  std::string symbol, assembly;
};

const EpilogueEmitterRegistry &getEpilogueEmitterRegistry(GemmIsa isa);
mlir::FailureOr<unsigned> selectFusiblePrefix(
    const atir::EpilogueProgram &program, const GemmPlan &plan,
    const EpilogueEmitterRegistry &registry);
mlir::FailureOr<GeneratedMicrokernel> generateMicrokernel(
    const MicrokernelRequest &request);

mlir::FailureOr<EmissionResult> emitBinaryCommon(const EmitRequest &request,
                                                 llvm::StringRef mnemonic,
                                                 bool sve);
mlir::FailureOr<EmissionResult> emitReluCommon(const EmitRequest &request,
                                               bool sve);

std::string remapScratchRegisters(llvm::StringRef instruction,
                                  llvm::ArrayRef<unsigned> registers,
                                  llvm::StringRef prefix);

}  // namespace annc::aarch64::gemm::epilogue

#endif  // ANNC_TARGET_AARCH64_GEMM_EPILOGUE_EPILOGUE_EMITTER_H
