#ifndef PIMP_C_DIALECTS_H
#define PIMP_C_DIALECTS_H

#include "mlir-c/IR.h"
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Pimp, pimp);

#ifdef __cplusplus
}
#endif

#endif // PIMP_C_DIALECTS_H