#include <utility>

#include "../GemmPlan.h"
#include "Target/aarch64/Gemm/Epilogue/EpilogueEmitter.h"
#include "llvm/ADT/StringExtras.h"

namespace annc::aarch64::gemm::epilogue {

namespace {

struct KernelFamily {
  llvm::StringRef isaName;
  llvm::StringRef symbolKind;
  llvm::StringRef asmName;
  llvm::StringRef templateName;
  llvm::StringRef generatorName;
  bool sve;
  bool matrixVector;
};

KernelFamily selectKernelFamily(const MicrokernelRequest &request) {
  if (request.matrixVector)
    return {"neon", "_matvec_mr", "NEON_MATVEC", "neon_matvec_f32.S",
            "GENERATE_NEON_MATVEC", false, true};
  if (request.isa == GemmIsa::kSve)
    return {"sve", request.rowMajor ? "_kernel_rm_mr" : "_kernel_mr",
            request.rowMajor ? "SVE_RM" : "SVE",
            request.rowMajor ? "sve_smallshape_rm_f32.S" : "sve_kernel_f32.S",
            request.rowMajor ? "GENERATE_SVE_KERNEL_RM"
                             : "GENERATE_SVE_KERNEL",
            true, false};
  return {"neon", request.rowMajor ? "_kernel_rm_mr" : "_kernel_mr",
          request.rowMajor ? "NEON_RM" : "NEON",
          request.rowMajor ? "neon_smallshape_rm_f32.S" : "neon_kernel_f32.S",
          request.rowMajor ? "GENERATE_NEON_KERNEL_RM"
                           : "GENERATE_NEON_KERNEL",
          false, false};
}

std::pair<int64_t, int64_t> neonPanelAndTail(int64_t microN) {
  const int64_t panels = (microN + 3) / 4;
  const int64_t tail = microN % 4 ? microN % 4 : 4;
  return {panels, tail};
}

bool containsForbidden(llvm::StringRef instruction) {
  return instruction.find("ret") != llvm::StringRef::npos ||
         instruction.find("str") != llvm::StringRef::npos ||
         instruction.find("st1w") != llvm::StringRef::npos ||
         instruction.find("stp") != llvm::StringRef::npos ||
         instruction.find("bl ") != llvm::StringRef::npos;
}

std::string symbolName(const MicrokernelRequest &request) {
  const KernelFamily family = selectKernelFamily(request);
  std::string symbol = "annc_aarch64_";
  symbol += family.isaName.str();
  symbol += family.symbolKind.str();
  symbol += llvm::utostr(request.mr);
  if (!family.matrixVector) {
    symbol += "_n";
    symbol += llvm::utostr(family.sve ? request.ng : request.microN);
  }
  if (family.sve)
    symbol += "vl";
  else {
    symbol += request.kHasGroups ? "_kg_r" : "_k";
    symbol += llvm::utostr(request.kResidue);
  }
  symbol += "_generated_" + request.epilogueSuffix;
  symbol += request.kcMode == KcMode::kAccumulate ? "_acc_f32" : "_f32";
  return symbol;
}

void appendKernel(std::string &assembly, const MicrokernelRequest &request) {
  const KernelFamily family = selectKernelFamily(request);
  assembly += ".set ANNC_" + family.asmName.str() +
              "_KERNELS_EPILOGUE_DEFINED, 1\n";
  assembly += ".set ANNC_" + family.asmName.str() +
              "_KERNELS_SKIP_INSTANTIATION, 1\n";
  assembly += ".macro " + family.asmName.str() + "_EPILOGUE";
  if (family.matrixVector)
    assembly += " mr\n";
  else
    assembly += family.sve ? " mr, ng\n" : " mr, n, panels, tail\n";
  for (const std::string &instruction : request.epilogueInstructions)
    assembly += "    " + instruction + "\n";
  assembly += ".endm\n";
  assembly += ".include \"" + family.templateName.str() + "\"\n" +
              family.generatorName.str() + " ";
  assembly += llvm::utostr(request.mr) + ", ";
  if (family.matrixVector) {
    assembly += llvm::utostr(request.kResidue) + ", " +
                llvm::utostr(request.kHasGroups ? 1 : 0);
  } else if (family.sve) {
    assembly += llvm::utostr(request.ng);
  } else {
    auto [panels, tail] = neonPanelAndTail(request.microN);
    assembly += llvm::utostr(request.microN) + ", " + llvm::utostr(panels) +
                ", " + llvm::utostr(tail);
    assembly += ", " + llvm::utostr(request.kResidue);
  }
  assembly += ", _generated_" + request.epilogueSuffix + "\n";
}

}  // namespace

mlir::FailureOr<GeneratedMicrokernel> generateMicrokernel(
    const MicrokernelRequest &request) {
  if (request.mr < 1 || request.mr > 6 || request.ng < 1 || request.ng > 4 ||
      request.epilogueSuffix.empty() ||
      llvm::StringRef(request.epilogueSuffix).find_first_of(" \n\r\t") !=
          llvm::StringRef::npos ||
      request.epilogueInstructions.empty())
    return mlir::failure();
  if (request.isa == GemmIsa::kNeon &&
      (request.microN < 1 || request.microN > 16))
    return mlir::failure();
  if (request.matrixVector &&
      (request.isa != GemmIsa::kNeon || request.mr > 4 || request.ng != 1 ||
       request.microN != 1 || !request.rowMajor))
    return mlir::failure();
  for (const std::string &instruction : request.epilogueInstructions)
    if (containsForbidden(instruction)) return mlir::failure();

  std::string assembly = ".altmacro\n";
  appendKernel(assembly, request);
  return GeneratedMicrokernel{symbolName(request), assembly};
}

}  // namespace annc::aarch64::gemm::epilogue
