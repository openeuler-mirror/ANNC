#include "mlir/CAPI/Registration.h"
#include "Dialect/Pimp/PimpOps.h"
#include "Builder/Builder.h"
#include "Helper.h"
#include "Dialect-c/Dialects.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Pimp, pimp, pimp::PimpDialect)
