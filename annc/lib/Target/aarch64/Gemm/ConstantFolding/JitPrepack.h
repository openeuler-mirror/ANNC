#ifndef ANNC_JIT_PREPACK_H
#define ANNC_JIT_PREPACK_H
#include "mlir/IR/BuiltinOps.h"
namespace annc {
mlir::LogicalResult prepareJitRhs(mlir::ModuleOp module,
                                  llvm::StringRef manifest,
                                  llvm::StringRef output);
}
#endif
