#ifndef DIALECT_ATIR_OPVERIFY_RUNORIGRAPH_H
#define DIALECT_ATIR_OPVERIFY_RUNORIGRAPH_H

#include "Dialect/Atir/OpVerify/IoDef.h"
#include "mlir/IR/BuiltinOps.h"

namespace atir {

/// Run the original graph by interpreting operations in topological order
void runOriGraph(mlir::ModuleOp root, const std::string &funcName,
                 const IoTensorDef *inputs, IoTensorDef *outputs);

} // namespace atir

#endif // DIALECT_ATIR_OPVERIFY_RUNORIGRAPH_H
