#ifndef ANNC_LIB_TARGET_AARCH64_GEMM_SUPPORT_GEMMSTRATEGYCONFIG_H
#define ANNC_LIB_TARGET_AARCH64_GEMM_SUPPORT_GEMMSTRATEGYCONFIG_H

#include <nlohmann/json.hpp>

#include "../GemmStrategy.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace annc::aarch64::gemm {

llvm::Expected<GemmCacheStrategy> parseGemmCacheStrategyConfig(
    llvm::StringRef name, const nlohmann::json &parameters,
    llvm::StringRef configPath);
llvm::Expected<GemmKernelStrategy> parseGemmKernelStrategyConfig(
    llvm::StringRef name, const nlohmann::json &parameters,
    llvm::StringRef configPath);

mlir::DictionaryAttr encodeCacheStrategyParameters(
    mlir::Builder &builder, const GemmCacheStrategy &strategy);
mlir::DictionaryAttr encodeKernelStrategyParameters(
    mlir::Builder &builder, const GemmKernelStrategy &strategy);
mlir::FailureOr<GemmCacheStrategy> decodeCacheStrategyParameters(
    mlir::Operation *op, GemmCacheStrategyKind kind,
    mlir::DictionaryAttr parameters);
mlir::FailureOr<GemmKernelStrategy> decodeKernelStrategyParameters(
    mlir::Operation *op, GemmKernelStrategyKind kind,
    mlir::DictionaryAttr parameters);

}  // namespace annc::aarch64::gemm

#endif  // ANNC_LIB_TARGET_AARCH64_GEMM_SUPPORT_GEMMSTRATEGYCONFIG_H
