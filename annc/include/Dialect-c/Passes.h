#ifndef ATIR_C_PASSES_H
#define ATIR_C_PASSES_H

#include "mlir-c/Support.h"

#include "Dialect/Atir/Passes/AtirPasses.capi.h.inc"
#include "Conversion/AtirConversionPasses.capi.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void anncRegisterAllTargetPasses(void);

#ifdef __cplusplus
}
#endif

#endif // ATIR_C_PASSES_H
