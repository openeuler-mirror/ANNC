#ifndef ANNC_DIALECT_ATIR_TEMPLATE_FINGERPRINT_H
#define ANNC_DIALECT_ATIR_TEMPLATE_FINGERPRINT_H

#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace atir {

std::string computeAtirTemplateFingerprint(mlir::ModuleOp module,
                                           mlir::func::FuncOp function);

}  // namespace atir

#endif  // ANNC_DIALECT_ATIR_TEMPLATE_FINGERPRINT_H
