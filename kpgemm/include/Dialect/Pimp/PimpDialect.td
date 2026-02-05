//===----------------------------------------------------------------------===//
// Define Pimp Dialect
//===----------------------------------------------------------------------===//
#ifndef PIMP_DIALECT_TD
#define PIMP_DIALECT_TD
include "mlir/IR/OpBase.td"
include "Dialect/Pimp/PimpEnums.td"

def Pimp_Dialect : Dialect {
  let name = "pimp";
  let summary = "Pimp Dialect";
  let description = [{}];
  let cppNamespace = "::pimp";

  let dependentDialects = [];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

include "Dialect/Pimp/PimpTypes.td"
include "Dialect/Pimp/PimpAttrs.td"

#endif // PIMP_DIALECT_TD