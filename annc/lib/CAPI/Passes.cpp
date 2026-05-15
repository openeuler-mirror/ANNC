#include "mlir/CAPI/Pass.h"
#include "mlir/Pass/Pass.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Conversion/Passes.h"
#include "Target/Passes.h"
#include "Dialect-c/Passes.h"

using namespace atir;
using namespace annc;

#ifdef __cplusplus
extern "C" {
#endif

#include "Dialect/Atir/Passes/AtirPasses.capi.cpp.inc"
#include "Conversion/AtirConversionPasses.capi.cpp.inc"
#include "Target/aarch64/AArch64CodeGenPasses.capi.cpp.inc"

#ifdef __cplusplus
}
#endif