#ifndef ANNC_DIALECT_ATIR_PASSES_GEMM_EPILOGUE_CANDIDATE_H
#define ANNC_DIALECT_ATIR_PASSES_GEMM_EPILOGUE_CANDIDATE_H

#include <cstdint>
#include <optional>
#include <string>

#include "Dialect/Atir/AtirOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace atir {

enum class EpilogueOpcode { kAdd, kMul, kRelu, kSigmoid };
enum class BroadcastKind { kScalar, kM, kN, kMatrix };
enum class OutputView { kIdentity, kTerminalInsertUnitDimension };

inline bool isBinaryEpilogueOpcode(EpilogueOpcode opcode) {
  return opcode == EpilogueOpcode::kAdd || opcode == EpilogueOpcode::kMul;
}

struct EpilogueStep {
  EpilogueOpcode opcode;
  unsigned inputIndex = 0;
  BroadcastKind broadcast = BroadcastKind::kMatrix;
  std::optional<float> reluLimit;
  mlir::Operation *source = nullptr;
  mlir::Value result;
};

struct EpilogueProgram {
  llvm::SmallVector<EpilogueStep> steps;
  OutputView terminalView = OutputView::kIdentity;
  mlir::Operation *terminalViewSource = nullptr;
  mlir::Value result;
};

inline constexpr llvm::StringLiteral kGemmEpilogueCandidateAttr =
    "annc.gemm.epilogue.candidate";
inline constexpr llvm::StringLiteral kGemmEpilogueAttr =
    "annc.gemm.epilogue";
inline constexpr int64_t kGemmEpilogueCandidateVersion = 1;

mlir::FailureOr<EpilogueProgram> discoverOrderedEpilogue(MatMulOp matmul);
std::optional<BroadcastKind> classifyEpilogueBroadcastShape(
    llvm::ArrayRef<int64_t> outputShape, llvm::ArrayRef<int64_t> inputShape);
mlir::FailureOr<EpilogueProgram> parseEpilogueCandidatePlan(
    mlir::DictionaryAttr candidate);
mlir::LogicalResult bindEpilogueProgramToSource(MatMulOp matmul,
                                                EpilogueProgram &program);
std::string staticSymbolSuffix(const EpilogueProgram &program, bool accumulate);
mlir::DictionaryAttr buildEpilogueCandidateAttr(mlir::Builder &builder,
                                                const EpilogueProgram &program);

}  // namespace atir

#endif  // ANNC_DIALECT_ATIR_PASSES_GEMM_EPILOGUE_CANDIDATE_H
