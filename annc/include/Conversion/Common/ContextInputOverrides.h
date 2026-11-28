#ifndef ANNC_CONVERSION_COMMON_CONTEXTINPUTOVERRIDES_H
#define ANNC_CONVERSION_COMMON_CONTEXTINPUTOVERRIDES_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

namespace atir {
// Materialize private folded constants in a call-local context. Empty entries
// keep their boundary input; outputs, attributes and services remain shared.
mlir::Value createContextInputOverrides(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::ModuleOp module,
    llvm::StringRef symbolPrefix, mlir::Value context,
    llvm::ArrayRef<mlir::DenseElementsAttr> overrides);
}  // namespace atir

#endif
