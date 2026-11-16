#ifndef ANNC_FRONTENDS_STABLEHLO_HLOGRAPH_H
#define ANNC_FRONTENDS_STABLEHLO_HLOGRAPH_H

#include <map>
#include <set>
#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace annc::stablehlo {

// Stable references are derived from function/block order, never from SSA
// print names, source locations, or framework attributes. The HLO is immutable.
struct HloGraph {
  mlir::ModuleOp module;
  std::map<std::string, mlir::Value> values;
  std::map<std::string, mlir::Operation *> operations;
  llvm::DenseMap<mlir::Value, std::string> names;
  llvm::DenseMap<mlir::Operation *, std::string> nodes;
  std::set<mlir::Operation *> claimed;

  explicit HloGraph(mlir::ModuleOp module);
  std::string name(mlir::Value value) const;
  std::string node(mlir::Operation *op) const;
  // Publish one directly fused StableHLO custom_call with ordinary tensor
  // boundaries and the shared FusionMetadata/context builtin schema.
  bool fuse(llvm::StringRef pattern, llvm::StringRef builtin,
            mlir::ValueRange inputs, mlir::ValueRange outputs,
            llvm::ArrayRef<int64_t> attrs,
            mlir::SmallVectorImpl<mlir::Attribute> &rejections);
  bool publish(mlir::func::FuncOp kernel,
               mlir::SmallVectorImpl<mlir::Attribute> &rejections);
  static bool shapeUse(mlir::Operation *op);
  static mlir::Value identitySource(mlir::Value value);
  static bool shapeBoundaryUse(mlir::Operation *op, mlir::ValueRange inputs);
};

}  // namespace annc::stablehlo
#endif
