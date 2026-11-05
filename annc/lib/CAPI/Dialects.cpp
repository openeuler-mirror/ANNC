#include "mlir/CAPI/Registration.h"
#include "Dialect/Atir/AtirOps.h"
#include "Helper.h"
#include "Dialect-c/Dialects.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Atir, atir, atir::AtirDialect)
