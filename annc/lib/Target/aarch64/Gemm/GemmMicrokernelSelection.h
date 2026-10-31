#ifndef ANNC_LIB_TARGET_AARCH64_GEMM_GEMM_MICROKERNEL_SELECTION_H
#define ANNC_LIB_TARGET_AARCH64_GEMM_GEMM_MICROKERNEL_SELECTION_H

#include "Target/aarch64/Gemm/GemmCodegenAdapter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace annc::aarch64::gemm {
struct FinalMicrokernelSelection {
  std::string symbol;
  codegen::KernelDesc codegenKernel;
  bool supportsEpilogueCodegen = true;
};

inline constexpr llvm::StringLiteral kFinalMicrokernelSelectionAttrName =
    "annc.aarch64.final_microkernel_selection";

mlir::FailureOr<FinalMicrokernelSelection> selectFinalMicrokernel(
    mlir::func::CallOp call);
mlir::LogicalResult attachFinalMicrokernelSelection(
    mlir::func::CallOp call, const FinalMicrokernelSelection &selection);
mlir::FailureOr<FinalMicrokernelSelection> readFinalMicrokernelSelection(
    mlir::func::CallOp call);
}  // namespace annc::aarch64::gemm
#endif
