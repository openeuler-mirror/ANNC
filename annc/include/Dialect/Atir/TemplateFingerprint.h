#ifndef ANNC_DIALECT_ATIR_TEMPLATE_FINGERPRINT_H
#define ANNC_DIALECT_ATIR_TEMPLATE_FINGERPRINT_H

#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace atir {

std::string computeAtirTemplateFingerprint(mlir::func::FuncOp function);

}  // namespace atir

#endif  // ANNC_DIALECT_ATIR_TEMPLATE_FINGERPRINT_H
