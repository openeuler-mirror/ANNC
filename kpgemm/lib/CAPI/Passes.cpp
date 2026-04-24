#include "mlir/CAPI/Pass.h"
#include "mlir/Pass/Pass.h"
#include "Dialect/Pimp/Passes/Passes.h"
#include "Conversion/Passes.h"
#include "Dialect-c/Passes.h"

using namespace pimp;

#ifdef __cplusplus
extern "C" {
#endif

#include "Dialect/Pimp/Passes/PimpPasses.capi.cpp.inc"
#include "Conversion/PimpConversionPasses.capi.cpp.inc"

#ifdef __cplusplus
}
#endif