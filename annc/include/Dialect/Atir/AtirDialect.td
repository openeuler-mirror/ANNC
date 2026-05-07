//===----------------------------------------------------------------------===//
// Define Atir Dialect
//===----------------------------------------------------------------------===//
#ifndef ATIR_DIALECT_TD
#define ATIR_DIALECT_TD
include "mlir/IR/OpBase.td"
include "Dialect/Atir/AtirEnums.td"

def Atir_Dialect : Dialect {
  let name = "atir";
  let summary = "Atir Dialect";
  let description = [{}];
  let cppNamespace = "::atir";

  let dependentDialects = [];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

include "Dialect/Atir/AtirTypes.td"
include "Dialect/Atir/AtirAttrs.td"

#endif // ATIR_DIALECT_TD