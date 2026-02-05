#ifndef ATIR_C_DIALECTS_H
#define ATIR_C_DIALECTS_H

#include "mlir-c/IR.h"
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Atir, atir);

#ifdef __cplusplus
}
#endif

#endif // ATIR_C_DIALECTS_H