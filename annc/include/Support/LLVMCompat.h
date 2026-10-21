// ANNC LLVM/MLIR 版本兼容层。
//
// ANNC 使用发行版 llvm-toolset-20（LLVM/MLIR 20.1.x）构建；部分 API 在 LLVM 21
// 有变化，统一收敛到本头文件，使同一份源码也能在新版本工具链上编译：
//   - GreedyRewriteConfig 的区域简化开关：
//       LLVM 21: 成员函数 setRegionSimplificationLevel(level)
//       LLVM 20: 公开字段 enableRegionSimplification
//   - bufferization 桥接算子：
//       LLVM 21: bufferization.to_buffer (ToBufferOp)
//       LLVM 20: bufferization.to_memref (ToMemrefOp)
//   - one-shot bufferize 调用：
//       LLVM 21: runOneShotBufferize(op, options, state) 显式 BufferizationState
//       LLVM 20: runOneShotBufferize(op, options) 无状态参数
#ifndef ANNC_SUPPORT_LLVMCOMPAT_H
#define ANNC_SUPPORT_LLVMCOMPAT_H

#include "llvm/Config/llvm-config.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace annc {

/// tensor -> memref 桥接算子（LLVM 21 起 ToMemrefOp 更名为 ToBufferOp）。
#if LLVM_VERSION_MAJOR >= 21
using ToBufferOp = ::mlir::bufferization::ToBufferOp;
#else
using ToBufferOp = ::mlir::bufferization::ToMemrefOp;
#endif

/// 设置 GreedyRewriteConfig 的区域简化级别（LLVM 20/21 兼容）。
inline void setRegionSimplificationLevel(
    mlir::GreedyRewriteConfig &config,
    mlir::GreedySimplifyRegionLevel level) {
#if LLVM_VERSION_MAJOR >= 21
  config.setRegionSimplificationLevel(level);
#else
  config.enableRegionSimplification = level;
#endif
}

/// 运行 one-shot bufferize（LLVM 20/21 兼容）。
inline mlir::LogicalResult runOneShotBufferize(
    mlir::Operation *op,
    const mlir::bufferization::OneShotBufferizationOptions &options) {
#if LLVM_VERSION_MAJOR >= 21
  mlir::bufferization::BufferizationState state;
  return mlir::bufferization::runOneShotBufferize(op, options, state);
#else
  return mlir::bufferization::runOneShotBufferize(op, options);
#endif
}

} // namespace annc

#endif // ANNC_SUPPORT_LLVMCOMPAT_H
